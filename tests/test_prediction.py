"""
tests/test_prediction.py - Unit tests for sentiment inference and explainability.
"""

import pytest
from src.predict import load_artifacts, predict_batch, predict_sentiment


def test_artifacts_loaded():
    model, vectorizer, metadata = load_artifacts()
    assert model is not None
    assert vectorizer is not None
    assert hasattr(model, "predict")
    assert hasattr(vectorizer, "transform")


def test_positive_prediction():
    text = "I absolutely love this new smartphone! The camera is stunning 😍✨ #happy"
    res = predict_sentiment(text)
    assert res["sentiment"] in ["Positive", "Neutral"]
    assert 0.0 <= res["confidence"] <= 1.0
    assert "probabilities" in res
    assert "explainability" in res
    assert isinstance(res["explainability"]["top_words"], list)


def test_negative_prediction():
    text = "This service is terrible and I am very disappointed. Never again! 😡👎"
    res = predict_sentiment(text)
    assert res["sentiment"] in ["Negative", "Neutral"]
    assert 0.0 <= res["confidence"] <= 1.0
    assert "negative" in res["probabilities"]


def test_empty_input():
    res = predict_sentiment("")
    assert res["sentiment"] == "Neutral"
    assert res["confidence"] == 0.0


def test_batch_prediction():
    samples = [
        "Brilliant performance and great battery life! 😍",
        "Worst customer experience ever. Total waste of money. 😡",
        "The train departs at 3 PM from platform 2.",
    ]
    results = predict_batch(samples)
    assert len(results) == 3
    for r in results:
        assert "sentiment" in r
        assert "confidence" in r
        assert r["sentiment"] in ["Positive", "Negative", "Neutral"]


def test_probability_consistency_and_confidence():
    """Verify that predicted sentiment matches the highest probability and confidence matches it."""
    test_inputs = [
        "I absolutely love this new smartphone! The camera is stunning 😍✨ #happy",
        "This service is terrible and I am very disappointed. Never using this company again! 😡👎",
        "The software update version 4.2.1 is now available for download from the official portal.",
        "advances in synthetic biology growing organs all this shit genetically modified humans as an example look at all those hundreds of breeds of dogs we have created that might happen to humans"
    ]
    for text in test_inputs:
        res = predict_sentiment(text)
        probs = res["probabilities"]
        top_prob_class = max(probs, key=probs.get)

        # Classified sentiment must match the class with highest probability
        assert res["sentiment"].lower() == top_prob_class, (
            f"Sentiment mismatch for '{text}': {res['sentiment']} != {top_prob_class}"
        )
        # Confidence score must match the predicted class probability
        assert round(res["confidence"], 4) == round(probs[top_prob_class], 4), (
            f"Confidence mismatch: {res['confidence']} != {probs[top_prob_class]}"
        )
        # Probabilities sum to 1.0 (100%)
        assert abs(sum(probs.values()) - 1.0) < 0.01
