"""
src/evaluate.py - Model Evaluation Module
Calculates comprehensive metrics including Accuracy, Precision, Recall,
F1 Score (Macro & Weighted), Confusion Matrix, and full Classification Report.
"""

from typing import Any, Dict, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from src.utils import logger


def evaluate_model(
    model: Any,
    X_test,
    y_test: List[str],
    classes: List[str]
) -> Dict[str, Any]:
    """
    Evaluate a trained classifier on test data and compute standardized metrics.
    """
    y_pred = model.predict(X_test)

    # Standard metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_list = cm.tolist()

    # Per-class classification report
    report = classification_report(
        y_test, y_pred, labels=classes, output_dict=True, zero_division=0
    )

    per_class_metrics = {}
    for cls in classes:
        if cls in report:
            per_class_metrics[cls] = {
                "precision": round(float(report[cls]["precision"]), 4),
                "recall": round(float(report[cls]["recall"]), 4),
                "f1_score": round(float(report[cls]["f1-score"]), 4),
                "support": int(report[cls]["support"]),
            }

    metrics = {
        "accuracy": round(acc, 4),
        "precision_macro": round(prec_macro, 4),
        "precision_weighted": round(prec_weighted, 4),
        "recall_macro": round(rec_macro, 4),
        "recall_weighted": round(rec_weighted, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "confusion_matrix": cm_list,
        "classes": classes,
        "per_class": per_class_metrics,
    }

    logger.info(
        f"Evaluation -> Accuracy: {acc:.4f} | Precision: {prec_weighted:.4f} | "
        f"Recall: {rec_weighted:.4f} | F1: {f1_weighted:.4f}"
    )
    return metrics
