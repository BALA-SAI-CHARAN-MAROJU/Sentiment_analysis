"""
src/train.py - Model Training & Selection Pipeline
Trains and evaluates three baseline NLP models:
1. Logistic Regression
2. Multinomial Naive Bayes
3. Linear Support Vector Machine (with Probability Calibration)
Extracts features using TF-IDF, selects the best model based on weighted F1 score,
and serializes artifacts to disk.
"""

from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from config import (
    CLASSES,
    MAX_FEATURES,
    METADATA_FILE,
    METRICS_FILE,
    MODEL_FILE,
    NGRAM_RANGE,
    PYTORCH_MODEL_FILE,
    RANDOM_STATE,
    TEST_SIZE,
    VECTORIZER_FILE,
)
from src.data_loader import get_dataset
from src.evaluate import evaluate_model
from src.neural_network import PyTorchSentimentClassifier
from src.utils import logger, save_json


def train_and_compare_models(force_reload_data: bool = False):
    """
    Main training routine:
    1. Load preprocessed dataset
    2. Perform stratified 80/20 train/test split
    3. Fit TfidfVectorizer on training set only (prevent data leakage)
    4. Train Logistic Regression, Naive Bayes, Linear SVM
    5. Evaluate all models
    6. Select and save the best performing model
    """
    logger.info("=" * 60)
    logger.info("[INFO] Starting Sentiment Analysis Model Training Pipeline")
    logger.info("=" * 60)

    # 1. Load Dataset
    df = get_dataset(force_reload=force_reload_data)
    if "cleaned_text" not in df.columns or "sentiment" not in df.columns:
        raise ValueError("Dataset missing 'cleaned_text' or 'sentiment' column.")

    X_texts = df["cleaned_text"].astype(str).tolist()
    y_labels = df["sentiment"].astype(str).tolist()

    logger.info(f"[INFO] Total dataset size: {len(X_texts)} samples across {len(CLASSES)} classes.")

    # 2. Train / Test Split (Stratified)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_texts,
        y_labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_labels,
    )
    logger.info(f"[INFO] Train size: {len(X_train_raw)}, Test size: {len(X_test_raw)}")

    # 3. Feature Extraction: TF-IDF
    logger.info(
        f"[INFO] Extracting TF-IDF features (max_features={MAX_FEATURES}, ngrams={NGRAM_RANGE})..."
    )
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        sublinear_tf=True,
        min_df=2,
    )
    X_train_vec = vectorizer.fit_transform(X_train_raw)
    X_test_vec = vectorizer.transform(X_test_raw)
    logger.info(f"[INFO] TF-IDF Vocabulary size: {len(vectorizer.vocabulary_)} features.")

    # 4. Define candidate models
    # Includes Linear Models, Probabilistic Model, Multi-Layer Perceptron, and PyTorch Deep Neural Network
    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "Multinomial Naive Bayes": MultinomialNB(
            alpha=0.5,
        ),
        "Linear SVM": CalibratedClassifierCV(
            LinearSVC(random_state=RANDOM_STATE, max_iter=2000, class_weight="balanced"),
            cv=3,
        ),
        "Multi-Layer Perceptron (NN)": MLPClassifier(
            hidden_layer_sizes=(64,),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            max_iter=25,
            random_state=RANDOM_STATE,
            early_stopping=True,
        ),
        "PyTorch Deep Neural Network": PyTorchSentimentClassifier(
            hidden_dim1=128,
            hidden_dim2=64,
            epochs=5,
            batch_size=64,
            random_state=RANDOM_STATE,
        ),
    }

    results = {}
    best_model_name = None
    best_f1 = -1.0
    best_model = None

    print("\n" + "=" * 75)
    print(f"{'Model Name':<28} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("=" * 75)

    # 5. Train and evaluate each model
    for name, model in models.items():
        logger.info(f"[INFO] Training {name}...")
        model.fit(X_train_vec, y_train)

        metrics = evaluate_model(model, X_test_vec, y_test, CLASSES)
        results[name] = metrics

        print(
            f"{name:<28} | "
            f"{metrics['accuracy'] * 100:>8.2f}% | "
            f"{metrics['precision_weighted'] * 100:>8.2f}% | "
            f"{metrics['recall_weighted'] * 100:>8.2f}% | "
            f"{metrics['f1_weighted'] * 100:>8.2f}%"
        )

        if metrics["f1_weighted"] > best_f1:
            best_f1 = metrics["f1_weighted"]
            best_model_name = name
            best_model = model

    print("=" * 75)
    logger.info(f"[INFO] Best Model Selected: {best_model_name} with Weighted F1: {best_f1:.4f}")

    # 6. Save Artifacts
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    logger.info(f"[INFO] Best model saved to: {MODEL_FILE}")
    logger.info(f"[INFO] Vectorizer saved to: {VECTORIZER_FILE}")

    # Save dedicated PyTorch checkpoint
    if "PyTorch Deep Neural Network" in models:
        try:
            models["PyTorch Deep Neural Network"].save_checkpoint(PYTORCH_MODEL_FILE)
            logger.info(f"[INFO] PyTorch model checkpoint saved to: {PYTORCH_MODEL_FILE}")
        except Exception as e:
            logger.warning(f"Could not save PyTorch checkpoint: {e}")

    # Save metadata
    metadata = {
        "best_model_name": best_model_name,
        "classes": CLASSES,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "max_features": MAX_FEATURES,
        "ngram_range": list(NGRAM_RANGE),
        "total_samples": len(X_texts),
        "train_samples": len(X_train_raw),
        "test_samples": len(X_test_raw),
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_metrics": results[best_model_name],
    }
    save_json(metadata, METADATA_FILE)

    # Save all comparison metrics for dashboard
    comparison_summary = {
        "best_model": best_model_name,
        "models": results,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(comparison_summary, METRICS_FILE)

    logger.info("[INFO] Model training and evaluation successfully completed!")
    return best_model, vectorizer, comparison_summary


if __name__ == "__main__":
    train_and_compare_models()
