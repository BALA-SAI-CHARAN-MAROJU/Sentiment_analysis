"""
src/predict.py - Sentiment Prediction Service
Loads the trained ML model and TF-IDF vectorizer, performs preprocessing,
computes sentiment predictions and confidence probabilities, and extracts
interpretability/explainability tokens. Supports optional Hugging Face transformers.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np

from config import (
    CLASSES,
    METADATA_FILE,
    MODEL_FILE,
    TRANSFORMER_MODEL_NAME,
    USE_TRANSFORMER,
    VECTORIZER_FILE,
)
from src.preprocessing import clean_text
from src.utils import load_json, logger
from src.neural_network import PyTorchSentimentClassifier  # Supports unpickling PyTorch models

# Module-level cache for loaded model and vectorizer
_MODEL = None
_VECTORIZER = None
_METADATA = None
_TRANSFORMER_PIPELINE = None


def load_artifacts():
    """Load model, vectorizer, and metadata from disk into memory."""
    global _MODEL, _VECTORIZER, _METADATA, _TRANSFORMER_PIPELINE

    if _MODEL is None or _VECTORIZER is None:
        if not MODEL_FILE.exists() or not VECTORIZER_FILE.exists():
            from src.train import train_and_compare_models
            logger.info("[INFO] Trained models not found. Triggering automated training pipeline...")
            _MODEL, _VECTORIZER, _ = train_and_compare_models()
        else:
            logger.info(f"[INFO] Loading sentiment model from {MODEL_FILE}...")
            _MODEL = joblib.load(MODEL_FILE)
            logger.info(f"[INFO] Loading vectorizer from {VECTORIZER_FILE}...")
            _VECTORIZER = joblib.load(VECTORIZER_FILE)

    if _METADATA is None and METADATA_FILE.exists():
        _METADATA = load_json(METADATA_FILE)

    if USE_TRANSFORMER and _TRANSFORMER_PIPELINE is None:
        try:
            from transformers import pipeline
            logger.info(f"[INFO] Initializing Transformer pipeline ({TRANSFORMER_MODEL_NAME})...")
            _TRANSFORMER_PIPELINE = pipeline(
                "sentiment-analysis",
                model=TRANSFORMER_MODEL_NAME,
                top_k=None,
                truncation=True,
            )
        except Exception as e:
            logger.warning(
                f"[WARNING] Could not initialize transformer pipeline ({e}). Falling back to ML."
            )
            _TRANSFORMER_PIPELINE = None

    return _MODEL, _VECTORIZER, _METADATA


def explain_prediction(cleaned_text: str, predicted_class: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Extract key influential words from the cleaned text that contributed to the prediction.
    Computes TF-IDF weight and feature alignment for the words present in vocabulary.
    """
    model, vectorizer, _ = load_artifacts()
    if not cleaned_text or vectorizer is None:
        return []

    tokens = cleaned_text.split()
    vocab = vectorizer.vocabulary_
    feature_names = vectorizer.get_feature_names_out()

    # Check words and bigrams present in vocab
    found_ngrams = []
    # 1. unigrams
    for token in tokens:
        if token in vocab:
            found_ngrams.append(token)
    # 2. bigrams
    for i in range(len(tokens) - 1):
        bigram = f"{tokens[i]} {tokens[i+1]}"
        if bigram in vocab:
            found_ngrams.append(bigram)

    if not found_ngrams:
        return []

    # Transform the single text to get TF-IDF weights
    vec = vectorizer.transform([cleaned_text]).toarray()[0]
    scored_terms = []
    for term in set(found_ngrams):
        idx = vocab[term]
        tfidf_val = float(vec[idx])
        if tfidf_val > 0.0:
            scored_terms.append({"word": term, "score": round(tfidf_val, 4)})

    # Sort descending by TF-IDF prominence
    scored_terms.sort(key=lambda x: x["score"], reverse=True)
    return scored_terms[:top_k]


def predict_sentiment(text: str) -> Dict[str, Any]:
    """
    Predict sentiment for a single social media post string.
    Returns:
    {
        "text": original text,
        "cleaned_text": processed text,
        "sentiment": "Positive" | "Negative" | "Neutral",
        "confidence": float (0.0 to 1.0),
        "probabilities": {"negative": float, "neutral": float, "positive": float},
        "explainability": {"top_words": [{"word": str, "score": float}, ...]},
        "model_used": str
    }
    """
    model, vectorizer, metadata = load_artifacts()

    # Validate input
    if text is None or not str(text).strip():
        return {
            "text": "",
            "cleaned_text": "",
            "sentiment": "Neutral",
            "confidence": 0.0,
            "probabilities": {"negative": 0.0, "neutral": 1.0, "positive": 0.0},
            "explainability": {"top_words": []},
            "model_used": metadata.get("best_model_name", "Linear SVM") if metadata else "Linear SVM",
            "status": "empty_input",
        }

    original_text = str(text)
    cleaned = clean_text(original_text)

    # If cleaned text is empty (e.g. only punctuation or noise)
    if not cleaned:
        return {
            "text": original_text,
            "cleaned_text": "",
            "sentiment": "Neutral",
            "confidence": 0.33,
            "probabilities": {"negative": 0.33, "neutral": 0.34, "positive": 0.33},
            "explainability": {"top_words": []},
            "model_used": metadata.get("best_model_name", "Linear SVM") if metadata else "Linear SVM",
            "status": "uninformative_input",
        }

    # Optional Transformer path
    if USE_TRANSFORMER and _TRANSFORMER_PIPELINE is not None:
        try:
            res = _TRANSFORMER_PIPELINE(original_text)[0]
            # Map transformer labels (e.g. LABEL_0, negative, etc.)
            label_map = {"positive": "positive", "negative": "negative", "neutral": "neutral"}
            probs = {}
            for item in res:
                lbl = item["label"].lower()
                probs[lbl] = round(float(item["score"]), 4)
            best_label = max(probs, key=probs.get)
            return {
                "text": original_text,
                "cleaned_text": cleaned,
                "sentiment": best_label.capitalize(),
                "confidence": probs.get(best_label, 0.5),
                "probabilities": probs,
                "explainability": {"top_words": explain_prediction(cleaned, best_label)},
                "model_used": f"Transformer ({TRANSFORMER_MODEL_NAME})",
            }
        except Exception as e:
            logger.warning(f"Transformer inference failed: {e}. Falling back to ML model.")

    # Standard ML path
    vec = vectorizer.transform([cleaned])
    predicted_label = model.predict(vec)[0]

    # Calculate probabilities
    if hasattr(model, "predict_proba"):
        probs_array = model.predict_proba(vec)[0]
        # model.classes_ may be in any order, map cleanly
        class_order = list(model.classes_)
        probabilities = {
            cls: round(float(probs_array[class_order.index(cls)]), 4)
            for cls in ["negative", "neutral", "positive"]
            if cls in class_order
        }
        confidence = probabilities.get(predicted_label, float(np.max(probs_array)))
    elif hasattr(model, "decision_function"):
        # Softmax over decision function values
        decision = model.decision_function(vec)[0]
        exp_d = np.exp(decision - np.max(decision))
        softmax_probs = exp_d / exp_d.sum()
        class_order = list(model.classes_)
        probabilities = {
            cls: round(float(softmax_probs[class_order.index(cls)]), 4)
            for cls in ["negative", "neutral", "positive"]
            if cls in class_order
        }
        confidence = probabilities.get(predicted_label, float(np.max(softmax_probs)))
    else:
        confidence = 0.85
        probabilities = {predicted_label: 0.85}

    top_words = explain_prediction(cleaned, predicted_label)

    return {
        "text": original_text,
        "cleaned_text": cleaned,
        "sentiment": str(predicted_label).capitalize(),
        "confidence": round(float(confidence), 4),
        "probabilities": probabilities,
        "explainability": {"top_words": top_words},
        "model_used": metadata.get("best_model_name", "Linear SVM") if metadata else "Linear SVM",
    }


def predict_batch(texts: List[str]) -> List[Dict[str, Any]]:
    """
    Perform fast batch prediction for a list of text posts.
    """
    model, vectorizer, metadata = load_artifacts()
    cleaned_texts = [clean_text(t) for t in texts]

    # Handle cases where cleaned text is empty
    safe_cleaned = [t if t.strip() else "neutral statement" for t in cleaned_texts]
    vecs = vectorizer.transform(safe_cleaned)

    predictions = model.predict(vecs)
    has_proba = hasattr(model, "predict_proba")
    if has_proba:
        probs = model.predict_proba(vecs)
        class_order = list(model.classes_)

    results = []
    for i, orig_text in enumerate(texts):
        pred = predictions[i]
        cln = cleaned_texts[i]
        if has_proba:
            cls_probs = {
                cls: round(float(probs[i][class_order.index(cls)]), 4)
                for cls in ["negative", "neutral", "positive"]
                if cls in class_order
            }
            conf = cls_probs.get(pred, round(float(np.max(probs[i])), 4))
        else:
            cls_probs = {pred: 0.85}
            conf = 0.85

        results.append({
            "original_text": orig_text,
            "cleaned_text": cln,
            "sentiment": str(pred).capitalize(),
            "confidence": conf,
            "probabilities": cls_probs,
        })

    return results


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    demo_samples = [
        "I absolutely love this new smartphone! The camera is stunning 😍✨ #happy",
        "This service is terrible and I am very disappointed. Never again! 😡👎",
        "The train is scheduled to arrive at platform 3 at 10:15 AM.",
    ]
    for s in demo_samples:
        print("\nInput:", s)
        res = predict_sentiment(s)
        print("Result:", res)
