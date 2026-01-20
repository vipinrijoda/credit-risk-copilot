"""Model evaluation: metrics, confusion matrix, ROC/PR curves, and
threshold optimization.

Accuracy alone is never treated as sufficient for imbalanced credit-risk
data — ROC-AUC, precision, recall and F1 are always reported alongside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class EvaluationResult:
    roc_auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: np.ndarray
    threshold: float
    roc_curve: dict = field(default_factory=dict)
    pr_curve: dict = field(default_factory=dict)
    class_distribution: dict = field(default_factory=dict)


def optimize_threshold(y_true: np.ndarray, y_proba: np.ndarray, strategy: str = "f1") -> float:
    """Find the classification threshold that optimizes the chosen metric
    on validation data. This is a MODELING decision, not a lending rule.
    """
    thresholds = np.linspace(0.01, 0.99, 197)
    best_threshold = 0.5
    best_score = -1.0

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        if preds.sum() == 0 or preds.sum() == len(preds):
            continue
        if strategy == "f1":
            score = f1_score(y_true, preds, zero_division=0)
        elif strategy == "recall":
            score = recall_score(y_true, preds, zero_division=0)
        elif strategy == "precision":
            score = precision_score(y_true, preds, zero_division=0)
        else:
            raise ValueError(f"Unknown threshold strategy: {strategy}")

        if score > best_score:
            best_score = score
            best_threshold = float(t)

    return round(best_threshold, 3)


def evaluate_model(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> EvaluationResult:
    y_pred = (y_proba >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_true, y_proba)) if len(set(y_true)) > 1 else float("nan")
    fpr, tpr, _ = (roc_curve(y_true, y_proba) if len(set(y_true)) > 1 else ([], [], []))
    prec_curve, rec_curve, _ = (
        precision_recall_curve(y_true, y_proba) if len(set(y_true)) > 1 else ([], [], [])
    )

    return EvaluationResult(
        roc_auc=roc_auc,
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        confusion=confusion_matrix(y_true, y_pred),
        threshold=threshold,
        roc_curve={"fpr": list(map(float, fpr)), "tpr": list(map(float, tpr))},
        pr_curve={"precision": list(map(float, prec_curve)), "recall": list(map(float, rec_curve))},
        class_distribution={int(k): int(v) for k, v in zip(*np.unique(y_true, return_counts=True))},
    )
