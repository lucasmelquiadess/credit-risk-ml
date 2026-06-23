from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def predict_positive_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        if probabilities.shape[1] == 1:
            return probabilities[:, 0]
        return probabilities[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        score_range = scores.max() - scores.min()
        if score_range == 0:
            return np.full(shape=scores.shape, fill_value=0.5, dtype=float)
        return (scores - scores.min()) / score_range

    raise TypeError("Model must expose predict_proba or decision_function.")


def calculate_binary_classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def evaluate_classifier(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    y_proba = predict_positive_probability(model, X_test)
    return calculate_binary_classification_metrics(y_test, y_proba, threshold)


def metrics_to_frame(metrics_by_model: dict[str, dict[str, Any]]) -> pd.DataFrame:
    return (
        pd.DataFrame.from_dict(metrics_by_model, orient="index")
        .sort_values(["pr_auc", "roc_auc"], ascending=False)
        .reset_index(names="model")
    )


def save_confusion_matrix_plot(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    output_path: Path,
    threshold: float = 0.5,
    title: str = "Confusion Matrix",
) -> None:
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1]).plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format="d",
    )
    ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve_plot(
    y_true: pd.Series | np.ndarray,
    probabilities_by_model: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    for model_name, y_proba in probabilities_by_model.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        ax.plot(fpr, tpr, label=f"{model_name} (ROC-AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_precision_recall_curve_plot(
    y_true: pd.Series | np.ndarray,
    probabilities_by_model: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    for model_name, y_proba in probabilities_by_model.items():
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
        ax.plot(recall, precision, label=f"{model_name} (PR-AUC={pr_auc:.3f})")

    baseline = float(np.mean(y_true))
    ax.axhline(
        baseline,
        linestyle="--",
        color="gray",
        label=f"Positive class baseline={baseline:.3f}",
    )
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
