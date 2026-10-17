"""
app.py - Main Flask Application & Auto-Initialization Orchestrator
Initializes project directories, checks/downloads dataset, trains/loads models,
sets up SQLite database, and exposes both web dashboard routes, live social feed,
and REST API endpoints.
"""

import io
import os
from pathlib import Path
import sys
from typing import Optional
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
import pandas as pd

# Ensure UTF-8 standard output for Windows command prompts
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from config import (
    ALLOWED_EXTENSIONS,
    BASE_DIR,
    DATABASE_FILE,
    MAX_CONTENT_LENGTH,
    METADATA_FILE,
    METRICS_FILE,
    MODEL_FILE,
    PROCESSED_DATA_FILE,
    RAW_DATA_FILE,
    SECRET_KEY,
    STATIC_DIR,
    TEMPLATES_DIR,
    UPLOADS_DIR,
    VECTORIZER_FILE,
)
from src.analytics import generate_wordclouds, get_dataset_analytics
from src.data_loader import get_dataset
from src.database import (
    clear_history,
    get_history,
    get_history_stats,
    get_live_posts_history,
    init_db,
    save_live_posts_batch,
    save_prediction,
    save_predictions_batch,
)
from src.predict import load_artifacts, predict_batch, predict_sentiment
from src.social_media_fetcher import fetch_social_posts
from src.train import train_and_compare_models
from src.utils import allowed_file, load_json, logger, sanitize_filename

# Initialize Flask application
app = Flask(
    __name__,
    template_folder=str(TEMPLATES_DIR),
    static_folder=str(STATIC_DIR),
)
app.secret_key = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Global in-memory cache for latest bulk CSV analysis result
_LATEST_BULK_DF = None


def auto_initialize_system() -> None:
    """
    Automated startup sequence:
    1. Verify & create required directories
    2. Check/download dataset and verify processed dataset
    3. Check/train ML models
    4. Initialize SQLite database
    5. Cache word clouds
    """
    logger.info("=" * 60)
    logger.info("[INFO] Initializing Sentiment Analysis System...")
    logger.info("=" * 60)

    # 1. Directory verification
    for d in [UPLOADS_DIR, STATIC_DIR / "images"]:
        d.mkdir(parents=True, exist_ok=True)

    # 2. Dataset verification
    logger.info("[INFO] Checking dataset...")
    if not PROCESSED_DATA_FILE.exists():
        logger.info("[INFO] Processed dataset not found. Acquiring and preprocessing data...")
        df = get_dataset()
        logger.info(f"[INFO] Dataset ready with {len(df)} rows.")
    else:
        logger.info(f"[INFO] Processed dataset found at {PROCESSED_DATA_FILE}.")

    # 3. Model verification
    logger.info("[INFO] Checking sentiment models...")
    if not MODEL_FILE.exists() or not VECTORIZER_FILE.exists():
        logger.info("[INFO] Trained model artifacts not found. Starting automatic training...")
        train_and_compare_models()
        logger.info("[INFO] Model training completed and artifacts saved.")
    else:
        logger.info("[INFO] Model artifacts found.")

    # 4. Load models into memory
    logger.info("[INFO] Loading sentiment model into memory...")
    load_artifacts()
    logger.info("[INFO] Model loaded successfully.")

    # 5. SQLite database verification
    logger.info("[INFO] Initializing SQLite database...")
    init_db()

    # 6. Ensure word clouds exist
    wc_pos = STATIC_DIR / "images" / "wordcloud_positive.png"
    if not wc_pos.exists():
        logger.info("[INFO] Generating initial word clouds for analytics dashboard...")
        try:
            generate_wordclouds()
        except Exception as e:
            logger.warning(f"Wordcloud generation deferred: {e}")

    logger.info("[INFO] System auto-initialization sequence complete.")
    logger.info("[INFO] Web server ready to accept requests.")
    logger.info("=" * 60)


# ============================================================================
# WEB INTERFACE ROUTES
# ============================================================================

@app.route("/")
def index():
    """Main dashboard displaying KPIs, distribution chart, model status, and analysis history."""
    keyword = request.args.get("keyword", "").strip()
    sentiment = request.args.get("sentiment", "all").strip()

    stats = get_history_stats()
    history = get_history(limit=50, keyword=keyword, sentiment=sentiment)
    metadata = load_json(METADATA_FILE)

    return render_template(
        "index.html",
        active_page="dashboard",
        history_stats=stats,
        history=history,
        metadata=metadata,
    )


@app.route("/analyze")
def analyze_view():
    """Live interactive single-post sentiment classification page."""
    metadata = load_json(METADATA_FILE)
    return render_template(
        "analyze.html",
        active_page="analyze",
        metadata=metadata,
    )


@app.route("/bulk")
def bulk_view():
    """Bulk CSV upload and tabular batch classification page."""
    metadata = load_json(METADATA_FILE)
    return render_template(
        "bulk.html",
        active_page="bulk",
        results=None,
        batch_stats=None,
        metadata=metadata,
    )


@app.route("/bulk-upload", methods=["POST"])
def bulk_upload():
    """Process uploaded CSV file for bulk sentiment inference."""
    global _LATEST_BULK_DF

    if "csv_file" not in request.files:
        return redirect(url_for("bulk_view"))

    file = request.files["csv_file"]
    if file.filename == "" or not allowed_file(file.filename, ALLOWED_EXTENSIONS):
        return redirect(url_for("bulk_view"))

    text_col = request.form.get("text_column", "").strip()
    max_rows_val = int(request.form.get("max_rows", "500"))

    try:
        # Read CSV
        df = pd.read_csv(file)

        # Auto-detect text column if not specified
        if not text_col or text_col not in df.columns:
            candidate_cols = ["text", "tweet", "post", "comment", "review", "content", "message"]
            matched_col = None
            for col in df.columns:
                if col.lower() in candidate_cols:
                    matched_col = col
                    break
            text_col = matched_col if matched_col else df.columns[0]

        # Subsample rows if configured
        if max_rows_val > 0 and len(df) > max_rows_val:
            df = df.head(max_rows_val)

        texts = df[text_col].dropna().astype(str).tolist()
        predictions = predict_batch(texts)

        # Persist to database
        save_predictions_batch(predictions, source=f"CSV ({Path(file.filename).name})")

        # Compile results for display and export
        results_df = pd.DataFrame(predictions)
        _LATEST_BULK_DF = results_df

        # Calculate batch statistics
        total = len(predictions)
        pos = sum(1 for p in predictions if p["sentiment"].lower() == "positive")
        neg = sum(1 for p in predictions if p["sentiment"].lower() == "negative")
        neu = sum(1 for p in predictions if p["sentiment"].lower() == "neutral")

        batch_stats = {
            "total": total,
            "positive": pos,
            "negative": neg,
            "neutral": neu,
            "positive_pct": round((pos / total) * 100, 1) if total else 0.0,
            "negative_pct": round((neg / total) * 100, 1) if total else 0.0,
            "neutral_pct": round((neu / total) * 100, 1) if total else 0.0,
        }

        metadata = load_json(METADATA_FILE)
        return render_template(
            "bulk.html",
            active_page="bulk",
            results=predictions,
            batch_stats=batch_stats,
            metadata=metadata,
        )

    except Exception as e:
        logger.error(f"Error processing CSV: {e}")
        metadata = load_json(METADATA_FILE)
        return render_template(
            "bulk.html",
            active_page="bulk",
            error=f"Error parsing CSV file: {str(e)}",
            results=None,
            metadata=metadata,
        )


@app.route("/download-bulk-csv")
def download_bulk_csv():
    """Download the most recently analyzed bulk prediction results as a CSV file."""
    global _LATEST_BULK_DF
    if _LATEST_BULK_DF is None or _LATEST_BULK_DF.empty:
        return redirect(url_for("bulk_view"))

    output = io.StringIO()
    _LATEST_BULK_DF.to_csv(output, index=False)
    output.seek(0)

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=sentiment_analysis_results.csv"},
    )


@app.route("/live-feed")
def live_feed_view():
    """Real-time social media data collection and live sentiment streaming page."""
    metadata = load_json(METADATA_FILE)
    recent_posts = get_live_posts_history(limit=10)

    if not recent_posts:
        # Fetch initial batch so the feed is active on first visit
        raw_posts = fetch_social_posts(source="all", limit=6)
        initial_analyzed = []
        for p in raw_posts:
            res = predict_sentiment(p["text"])
            initial_analyzed.append({
                "text": p["text"],
                "source": p["source"],
                "sentiment": res["sentiment"],
                "confidence": res["confidence"],
                "timestamp": p.get("timestamp") or p.get("created_at"),
                "url": p.get("url", ""),
            })
        save_live_posts_batch(initial_analyzed)
        recent_posts = initial_analyzed

    return render_template(
        "live_feed.html",
        active_page="live",
        initial_posts=recent_posts,
        metadata=metadata,
    )


@app.route("/analytics")
def analytics_view():
    """Exploratory Data Analysis and corpus word clouds."""
    metadata = load_json(METADATA_FILE)
    analytics = get_dataset_analytics()
    return render_template(
        "analytics.html",
        active_page="analytics",
        analytics=analytics,
        metadata=metadata,
    )


@app.route("/performance")
def performance_view():
    """Model comparison table, confusion matrix, and benchmark metrics."""
    metrics = load_json(METRICS_FILE)
    metadata = load_json(METADATA_FILE)
    return render_template(
        "performance.html",
        active_page="performance",
        metrics=metrics,
        metadata=metadata,
    )


@app.route("/about")
def about_view():
    """Final-year project documentation, system architecture, DFDs, and viva guide."""
    metadata = load_json(METADATA_FILE)
    return render_template(
        "about.html",
        active_page="about",
        metadata=metadata,
    )


# ============================================================================
# REST API ENDPOINTS
# ============================================================================

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    POST /api/predict
    Request Body: {"text": "I love this!", "source": "Optional Source"}
    Returns: {"sentiment": "Positive", "confidence": 0.94, ...}
    """
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in JSON request payload."}), 400

    text = data.get("text", "")
    source = data.get("source", "API Request")

    if not str(text).strip():
        return jsonify({"error": "Text field cannot be empty."}), 400

    result = predict_sentiment(text)

    # Save prediction in history database
    save_prediction(
        original_text=result["text"],
        cleaned_text=result["cleaned_text"],
        sentiment=result["sentiment"],
        confidence=result["confidence"],
        probabilities=result.get("probabilities", {}),
        source=source,
    )

    return jsonify(result), 200


@app.route("/api/bulk-predict", methods=["POST"])
def api_bulk_predict():
    """
    POST /api/bulk-predict
    Request Body: {"texts": ["Great phone", "Terrible update"]}
    Returns: {"count": 2, "predictions": [...]}
    """
    data = request.get_json(silent=True)
    if not data or "texts" not in data or not isinstance(data["texts"], list):
        return jsonify({"error": "Missing or invalid 'texts' array in JSON request."}), 400

    texts = data["texts"]
    if not texts:
        return jsonify({"count": 0, "predictions": []}), 200

    predictions = predict_batch(texts)
    save_predictions_batch(predictions, source="API Bulk")

    return jsonify({"count": len(predictions), "predictions": predictions}), 200


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """GET /api/stats - Retrieve current database sentiment distribution counts."""
    stats = get_history_stats()
    return jsonify(stats), 200


@app.route("/api/history", methods=["GET"])
def api_history():
    """
    GET /api/history
    Query params: limit, sentiment, keyword, min_confidence, source
    """
    limit = int(request.args.get("limit", "50"))
    sentiment = request.args.get("sentiment")
    keyword = request.args.get("keyword")
    min_confidence = request.args.get("min_confidence", type=float)
    source = request.args.get("source")

    records = get_history(
        limit=limit,
        sentiment=sentiment,
        keyword=keyword,
        min_confidence=min_confidence,
        source=source,
    )
    return jsonify({"count": len(records), "records": records}), 200


@app.route("/api/fetch-live", methods=["GET"])
def api_fetch_live():
    """
    GET /api/fetch-live
    Query params: source (reddit, mastodon, youtube, all), limit, subreddit, video_id
    Retrieves real-time social posts, executes sentiment analysis, stores them
    in the SQLite live_posts table, and returns formatted JSON.
    """
    source = request.args.get("source", "all").strip().lower()
    limit = int(request.args.get("limit", "10"))
    subreddit = request.args.get("subreddit", "technology").strip()
    video_id = request.args.get("video_id")

    # 1. Fetch raw social posts
    raw_posts = fetch_social_posts(
        source=source,
        limit=limit,
        subreddit=subreddit,
        video_id=video_id,
    )

    # 2. Analyze sentiment for each post
    analyzed_posts = []
    summary = {"positive": 0, "negative": 0, "neutral": 0}

    for p in raw_posts:
        pred = predict_sentiment(p["text"])
        sentiment = pred["sentiment"]
        conf = pred["confidence"]

        snt_key = sentiment.lower()
        if snt_key in summary:
            summary[snt_key] += 1

        analyzed_posts.append({
            "text": p["text"],
            "source": p["source"],
            "sentiment": sentiment,
            "confidence": conf,
            "timestamp": p.get("timestamp") or p.get("created_at"),
            "url": p.get("url", ""),
        })

    # 3. Store in SQLite live_posts table
    if analyzed_posts:
        save_live_posts_batch(analyzed_posts)

    return jsonify({
        "source": source,
        "count": len(analyzed_posts),
        "summary": summary,
        "posts": analyzed_posts,
    }), 200


@app.route("/api/clear-history", methods=["POST"])
def api_clear_history():
    """POST /api/clear-history - Clear prediction history database records."""
    success = clear_history()
    return jsonify({"success": success, "message": "History cleared."}), 200


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def page_not_found(e):
    return render_template("base.html", content="<div class='text-center py-5'><h2>404 - Page Not Found</h2><p class='text-muted'>The requested page does not exist.</p><a href='/' class='btn btn-primary-custom'>Return to Dashboard</a></div>"), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template("base.html", content="<div class='text-center py-5'><h2>500 - Internal Server Error</h2><p class='text-muted'>An unexpected error occurred. Please verify logs.</p><a href='/' class='btn btn-primary-custom'>Return to Dashboard</a></div>"), 500


# Application Entrypoint
if __name__ == "__main__":
    auto_initialize_system()
    port = int(os.getenv("PORT", "5000"))
    print("\n" + "=" * 65)
    print(f"🚀 Sentiment Analysis Web Application Live at: http://127.0.0.1:{port}")
    print("=" * 65 + "\n")
    app.run(host="0.0.0.0", port=port, debug=False)
