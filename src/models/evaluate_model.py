from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def predict_positive_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return (scores - scores.min()) / (scores.max() - scores.min())

    raise TypeError("Model must expose predict_proba or decision_function.")


def evaluate_classifier(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_proba = predict_positive_probability(model, X_test)
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_default": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall_default": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_default": float(f1_score(y_test, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def metrics_to_frame(metrics_by_model: dict[str, dict[str, Any]]) -> pd.DataFrame:
    return (
        pd.DataFrame.from_dict(metrics_by_model, orient="index")
        .sort_values(["pr_auc", "roc_auc"], ascending=False)
        .reset_index(names="model")
    )
