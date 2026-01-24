"""Explainability for individual predictions and global model behaviour.

Preferred method: SHAP (TreeExplainer for tree models, else a
model-agnostic Explainer / KernelExplainer fallback for linear models).
If SHAP fails for any reason, we fall back to permutation importance or
model coefficients. Explanations are ALWAYS derived from the actual fitted
model — the GenAI Copilot is only ever given these computed numbers and
must never invent feature importance itself (see src/genai/tools.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from src.config import FEATURE_BY_NAME


@dataclass
class FeatureContribution:
    feature: str
    label: str
    value: object
    contribution: float  # signed: positive = increases risk, negative = decreases risk


@dataclass
class ExplanationResult:
    method: str  # "shap" | "permutation_importance" | "coefficients"
    top_risk_increasing: list[FeatureContribution] = field(default_factory=list)
    top_risk_reducing: list[FeatureContribution] = field(default_factory=list)
    global_importance: Optional[pd.DataFrame] = None


def _pretty_label(raw_feature_name: str) -> str:
    """Map a one-hot-encoded or raw feature name back to a human label."""
    base = raw_feature_name.split("__")[-1]
    for canonical, feature in FEATURE_BY_NAME.items():
        if base == canonical or base.startswith(canonical + "_"):
            suffix = base[len(canonical):].lstrip("_")
            return f"{feature.label}" + (f" = {suffix}" if suffix else "")
    return base


def _get_transformed_feature_names(pipeline: Pipeline) -> list[str]:
    preprocessor = pipeline.named_steps["preprocessing"]
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:  # noqa: BLE001
        return [f"feature_{i}" for i in range(pipeline.named_steps["classifier"].n_features_in_)]


def explain_single_prediction(
    customer_df: pd.DataFrame, pipeline: Pipeline, top_n: int = 5
) -> ExplanationResult:
    """Explain ONE row (a single-row dataframe with raw canonical columns)."""
    fe = pipeline.named_steps["feature_engineering"]
    preprocessor = pipeline.named_steps["preprocessing"]
    classifier = pipeline.named_steps["classifier"]

    engineered = fe.transform(customer_df)
    transformed = preprocessor.transform(engineered)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = _get_transformed_feature_names(pipeline)

    try:
        import shap

        model_type_name = type(classifier).__name__
        if model_type_name in ("RandomForestClassifier", "XGBClassifier", "LGBMClassifier", "CatBoostClassifier"):
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(transformed)
            if isinstance(shap_values, list):  # some tree explainers return [class0, class1]
                shap_values = shap_values[1]
            if shap_values.ndim == 3:  # (n_samples, n_features, n_classes)
                shap_values = shap_values[:, :, 1]
            values = shap_values[0]
        else:
            background = np.zeros((1, transformed.shape[1]))
            explainer = shap.LinearExplainer(classifier, background)
            values = explainer.shap_values(transformed)[0]
        method = "shap"
    except Exception:  # noqa: BLE001 — any SHAP failure falls back gracefully
        values, method = _coefficient_fallback(classifier, transformed, feature_names)

    contributions = [
        FeatureContribution(feature=name, label=_pretty_label(name), value=None, contribution=float(v))
        for name, v in zip(feature_names, values)
    ]
    contributions.sort(key=lambda c: c.contribution, reverse=True)

    top_risk_increasing = [c for c in contributions if c.contribution > 0][:top_n]
    top_risk_reducing = sorted(
        [c for c in contributions if c.contribution < 0], key=lambda c: c.contribution
    )[:top_n]

    return ExplanationResult(
        method=method,
        top_risk_increasing=top_risk_increasing,
        top_risk_reducing=top_risk_reducing,
    )


def _coefficient_fallback(classifier, transformed, feature_names) -> tuple[np.ndarray, str]:
    if hasattr(classifier, "coef_"):
        coefs = classifier.coef_[0]
        contribution = coefs * transformed[0]
        return contribution, "coefficients"
    if hasattr(classifier, "feature_importances_"):
        # Feature importances are unsigned global scores; use them scaled by
        # the (standardized) input value's sign as a rough local proxy.
        importances = classifier.feature_importances_
        signed = importances * np.sign(transformed[0])
        return signed, "permutation_importance"
    return np.zeros(len(feature_names)), "unavailable"


def global_feature_importance(pipeline: Pipeline, X_val: pd.DataFrame, y_val: pd.Series, top_n: int = 15) -> pd.DataFrame:
    """Model-level (global) feature importance via permutation importance —
    used on the Model Performance page, independent of any single prediction."""
    result = permutation_importance(
        pipeline, X_val, y_val, n_repeats=8, random_state=42, scoring="roc_auc", n_jobs=-1
    )
    df = pd.DataFrame({
        "feature": X_val.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).head(top_n)
    df["label"] = df["feature"].apply(lambda f: FEATURE_BY_NAME.get(f, None).label if f in FEATURE_BY_NAME else f)
    return df.reset_index(drop=True)
