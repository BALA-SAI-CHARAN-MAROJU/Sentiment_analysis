"""
src/database.py - SQLite Database Storage Service
Handles storage, retrieval, filtering, and statistics calculation
for sentiment analysis predictions, history logs, and real-time live social media posts.
"""

from datetime import datetime
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional

from config import DATABASE_FILE
from src.utils import logger


def get_connection() -> sqlite3.Connection:
    """Create and return a thread-safe connection to the SQLite database."""
    DATABASE_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DATABASE_FILE), timeout=15)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize database tables and indexes if they do not already exist."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        # 1. Main predictions history table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_text TEXT NOT NULL,
                cleaned_text TEXT,
                sentiment TEXT NOT NULL,
                confidence REAL NOT NULL,
                probabilities TEXT,
                source TEXT DEFAULT 'Manual Input',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sentiment ON predictions(sentiment);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON predictions(created_at);")

        # 2. Real-time live social posts table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS live_posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                source TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                url TEXT
            );
            """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_live_source ON live_posts(source);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_live_sentiment ON live_posts(sentiment);")

        conn.commit()
        logger.info(f"[INFO] SQLite database initialized at {DATABASE_FILE}")
    except Exception as e:
        logger.error(f"[ERROR] Database initialization failed: {e}")
        conn.rollback()
        raise e
    finally:
        conn.close()


def save_prediction(
    original_text: str,
    cleaned_text: str,
    sentiment: str,
    confidence: float,
    probabilities: Optional[Dict[str, float]] = None,
    source: str = "Manual Input",
) -> int:
    """Save a single prediction entry to the database."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        prob_json = json.dumps(probabilities or {})
        cursor.execute(
            """
            INSERT INTO predictions (original_text, cleaned_text, sentiment, confidence, probabilities, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                original_text,
                cleaned_text,
                sentiment.capitalize(),
                round(float(confidence), 4),
                prob_json,
                source,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        conn.commit()
        return cursor.lastrowid
    except Exception as e:
        logger.error(f"Error saving prediction to database: {e}")
        conn.rollback()
        return -1
    finally:
        conn.close()


def save_predictions_batch(records: List[Dict[str, Any]], source: str = "CSV Upload") -> int:
    """Save multiple prediction records in a single database transaction."""
    if not records:
        return 0
    conn = get_connection()
    try:
        cursor = conn.cursor()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tuples = [
            (
                r.get("original_text", ""),
                r.get("cleaned_text", ""),
                r.get("sentiment", "Neutral").capitalize(),
                round(float(r.get("confidence", 0.0)), 4),
                json.dumps(r.get("probabilities", {})),
                source,
                now_str,
            )
            for r in records
        ]
        cursor.executemany(
            """
            INSERT INTO predictions (original_text, cleaned_text, sentiment, confidence, probabilities, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            tuples,
        )
        conn.commit()
        return len(tuples)
    except Exception as e:
        logger.error(f"Error saving batch predictions: {e}")
        conn.rollback()
        return 0
    finally:
        conn.close()


def get_history(
    limit: int = 100,
    sentiment: Optional[str] = None,
    keyword: Optional[str] = None,
    min_confidence: Optional[float] = None,
    source: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve filtered prediction records from SQLite."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        query = "SELECT * FROM predictions WHERE 1=1"
        params: List[Any] = []

        if sentiment and sentiment.lower() != "all":
            query += " AND LOWER(sentiment) = ?"
            params.append(sentiment.lower())

        if keyword and keyword.strip():
            query += " AND (original_text LIKE ? OR cleaned_text LIKE ?)"
            term = f"%{keyword.strip()}%"
            params.extend([term, term])

        if min_confidence is not None and min_confidence > 0:
            query += " AND confidence >= ?"
            params.append(float(min_confidence))

        if source and source.lower() != "all":
            query += " AND LOWER(source) = ?"
            params.append(source.lower())

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        results = []
        for row in rows:
            row_dict = dict(row)
            try:
                row_dict["probabilities"] = json.loads(row_dict.get("probabilities") or "{}")
            except Exception:
                row_dict["probabilities"] = {}
            results.append(row_dict)
        return results
    finally:
        conn.close()


def get_history_stats() -> Dict[str, Any]:
    """Calculate aggregated sentiment counts and metrics for the dashboard."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT LOWER(sentiment), COUNT(*), AVG(confidence)
            FROM predictions
            GROUP BY LOWER(sentiment)
            """
        )
        rows = cursor.fetchall()

        stats = {
            "total": total,
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "positive_pct": 0.0,
            "negative_pct": 0.0,
            "neutral_pct": 0.0,
            "avg_confidence": 0.0,
        }

        conf_sum = 0.0
        conf_count = 0
        for s, count, avg_conf in rows:
            if s in stats:
                stats[s] = count
            if avg_conf is not None:
                conf_sum += avg_conf * count
                conf_count += count

        if total > 0:
            stats["positive_pct"] = round((stats["positive"] / total) * 100, 1)
            stats["negative_pct"] = round((stats["negative"] / total) * 100, 1)
            stats["neutral_pct"] = round((stats["neutral"] / total) * 100, 1)
            if conf_count > 0:
                stats["avg_confidence"] = round(conf_sum / conf_count, 3)

        return stats
    finally:
        conn.close()


def save_live_post(
    text: str,
    source: str,
    sentiment: str,
    confidence: float,
    timestamp: Optional[str] = None,
    url: Optional[str] = None,
) -> int:
    """Save a single analyzed real-time live post to the live_posts table."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            """
            INSERT INTO live_posts (text, source, sentiment, confidence, timestamp, url)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                text,
                source,
                sentiment.capitalize(),
                round(float(confidence), 4),
                ts,
                url or "",
            ),
        )
        conn.commit()
        return cursor.lastrowid
    except Exception as e:
        logger.error(f"Error saving live post: {e}")
        conn.rollback()
        return -1
    finally:
        conn.close()


def save_live_posts_batch(posts: List[Dict[str, Any]]) -> int:
    """Save multiple analyzed live posts to the live_posts table."""
    if not posts:
        return 0
    conn = get_connection()
    try:
        cursor = conn.cursor()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tuples = [
            (
                p.get("text", ""),
                p.get("source", "Live Stream"),
                p.get("sentiment", "Neutral").capitalize(),
                round(float(p.get("confidence", 0.0)), 4),
                p.get("timestamp") or p.get("created_at") or now_str,
                p.get("url", ""),
            )
            for p in posts
        ]
        cursor.executemany(
            """
            INSERT INTO live_posts (text, source, sentiment, confidence, timestamp, url)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            tuples,
        )
        conn.commit()
        return len(tuples)
    except Exception as e:
        logger.error(f"Error saving batch live posts: {e}")
        conn.rollback()
        return 0
    finally:
        conn.close()


def get_live_posts_history(limit: int = 50, source: Optional[str] = None, sentiment: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve saved live posts from SQLite."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        query = "SELECT * FROM live_posts WHERE 1=1"
        params: List[Any] = []

        if source and source.lower() != "all":
            query += " AND LOWER(source) LIKE ?"
            params.append(f"%{source.lower()}%")

        if sentiment and sentiment.lower() != "all":
            query += " AND LOWER(sentiment) = ?"
            params.append(sentiment.lower())

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def clear_history() -> bool:
    """Clear all records from predictions table."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions")
        conn.commit()
        return True
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    print("Database schema successfully verified with live_posts support.")
