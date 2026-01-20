"""Training orchestration for both the built-in demo model and
user-uploaded datasets (Type A: dataset WITH a target column).

This module is deliberately the ONLY place that fits a model. The
Streamlit UI never trains anything itself — it calls `train_model()` and
renders the returned result.
"""

from __future__ import annotations

import json
import platform
import sklearn
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    CATEGORICAL_RAW_FEATURES,
    ENGINEERED_FEATURES,
    NUMERIC_RAW_FEATURES,
)
from src.modeling.evaluate import EvaluationResult, evaluate_model, optimize_threshold
from src.modeling.feature_engineering import FinancialFeatureEngineer
from src.modeling.pipeline_builder import SchemaColumns, build_pipeline


class TrainingError(Exception):
    """Raised when training cannot proceed safely."""


@dataclass
class TrainingResult:
    pipeline: Pipeline
    evaluation: EvaluationResult
    metadata: dict
    warnings: list[str] = field(default_factory=list)


def _detect_schema_columns(df: pd.DataFrame) -> SchemaColumns:
    """Determine which numeric/categorical columns (raw + engineered) are
    actually present, so the ColumnTransformer only references real data.
    """
    numeric = [c for c in NUMERIC_RAW_FEATURES if c in df.columns]
    numeric += [f.name for f in ENGINEERED_FEATURES]  # always computed by the transformer
    categorical = [c for c in CATEGORICAL_RAW_FEATURES if c in df.columns]
    # de-duplicate while preserving order
    numeric = list(dict.fromkeys(numeric))
    categorical = list(dict.fromkeys(categorical))
    return SchemaColumns(numeric=numeric, categorical=categorical)


def pre_training_checks(df: pd.DataFrame, target_col: str) -> list[str]:
    """Return a list of human-readable warnings about dataset suitability.
    Raises TrainingError for conditions that make training impossible."""
    warnings: list[str] = []

    if target_col not in df.columns:
        raise TrainingError(f"Target column '{target_col}' not found in dataset.")

    y = df[target_col].dropna()
    n_classes = y.nunique()
    if n_classes < 2:
        raise TrainingError(
            "The target column has only one class present. A binary default "
            "classifier cannot be trained on a single-class outcome."
        )

    n_rows = len(df)
    if n_rows < 50:
        raise TrainingError(
            f"Dataset has only {n_rows} rows after removing missing targets — "
            "too few to train a reliable model. At least ~200 rows are recommended."
        )
    if n_rows < 200:
        warnings.append(
            f"Dataset is small ({n_rows} rows). Metrics may be unstable; "
            "treat results as indicative only."
        )

    proportions = y.value_counts(normalize=True)
    minority = proportions.min()
    if minority < 0.03:
        warnings.append(
            f"Extremely imbalanced target: minority class is {minority:.1%} of rows. "
            "Model may struggle to learn minority-class patterns; class-weighting is applied."
        )
    elif minority < 0.15:
        warnings.append(f"Imbalanced target: minority class is {minority:.1%} of rows.")

    available_features = [c for c in NUMERIC_RAW_FEATURES + CATEGORICAL_RAW_FEATURES if c in df.columns]
    if len(available_features) < 3:
        warnings.append(
            f"Only {len(available_features)} recognized feature(s) found in this dataset "
            "after column mapping. Model quality may be limited."
        )

    return warnings


def _build_distribution_reference(X_train: pd.DataFrame, schema: SchemaColumns) -> dict:
    """Capture training-data distribution reference points used later for
    out-of-distribution detection at prediction time (see src/modeling/predict.py).
    Only RAW numeric/categorical columns are captured here (engineered
    features are recomputed from raw values, so OOD is checked on raw inputs).
    """
    reference: dict = {"numeric": {}, "categorical": {}}
    raw_numeric = [c for c in NUMERIC_RAW_FEATURES if c in X_train.columns]
    raw_categorical = [c for c in CATEGORICAL_RAW_FEATURES if c in X_train.columns]

    for col in raw_numeric:
        series = pd.to_numeric(X_train[col], errors="coerce").dropna()
        if len(series) == 0:
            continue
        reference["numeric"][col] = {
            "p1": float(np.percentile(series, 1)),
            "p99": float(np.percentile(series, 99)),
            "median": float(np.median(series)),
        }

    for col in raw_categorical:
        reference["categorical"][col] = sorted(X_train[col].dropna().astype(str).unique().tolist())

    return reference


def train_model(
    df: pd.DataFrame,
    target_col: str,
    model_name: str = "Random Forest",
    threshold_strategy: str = "f1",
    test_size: float = 0.2,
    excluded_columns: Optional[list[str]] = None,
    data_source_type: str = "user_uploaded",
    random_state: int = 42,
) -> TrainingResult:
    """Train, validate, and package a credit-default model.

    Parameters
    ----------
    df: canonical-schema dataframe (post column-mapping) INCLUDING the target column.
    target_col: name of the binary outcome column.
    excluded_columns: columns to drop before training (e.g. leakage/PII flagged).
    """
    excluded_columns = excluded_columns or []
    working_df = df.drop(columns=[c for c in excluded_columns if c in df.columns]).copy()

    warnings = pre_training_checks(working_df, target_col)

    working_df = working_df.dropna(subset=[target_col])
    y = pd.to_numeric(working_df[target_col], errors="coerce")
    if y.isna().any():
        raise TrainingError(
            "Target column contains values that cannot be interpreted as 0/1. "
            "Please map default values to 0 (no default) / 1 (default)."
        )
    y = y.astype(int)
    X = working_df.drop(columns=[target_col])

    schema = _detect_schema_columns(X)
    if not schema.numeric and not schema.categorical:
        raise TrainingError(
            "No recognized canonical features found in this dataset after column "
            "mapping. Please map at least a few columns (income, loan amount, etc.) "
            "before training."
        )

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    pipeline = build_pipeline(schema, model_name=model_name)
    try:
        pipeline.fit(X_train, y_train)
    except Exception as exc:  # noqa: BLE001
        raise TrainingError(f"Model training failed: {exc}") from exc

    y_proba_val = pipeline.predict_proba(X_val)[:, 1]
    threshold = optimize_threshold(y_val.to_numpy(), y_proba_val, strategy=threshold_strategy)
    evaluation = evaluate_model(y_val.to_numpy(), y_proba_val, threshold)

    distribution_reference = _build_distribution_reference(X_train, schema)

    metadata = {
        "model_type": type(pipeline.named_steps["classifier"]).__name__,
        "model_name": model_name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_rows": int(len(df)),
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "features_numeric": schema.numeric,
        "features_categorical": schema.categorical,
        "target_column": target_col,
        "threshold": threshold,
        "threshold_strategy": threshold_strategy,
        "metrics": {
            "roc_auc": evaluation.roc_auc,
            "accuracy": evaluation.accuracy,
            "precision": evaluation.precision,
            "recall": evaluation.recall,
            "f1": evaluation.f1,
        },
        "is_demo_model": data_source_type == "synthetic",
        "data_source_type": data_source_type,  # "synthetic" | "public_dataset" | "user_uploaded"
        "excluded_columns": excluded_columns,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "distribution_reference": distribution_reference,
    }

    return TrainingResult(pipeline=pipeline, evaluation=evaluation, metadata=metadata, warnings=warnings)


def save_model(result: TrainingResult, model_path: str, metadata_path: str) -> None:
    joblib.dump(result.pipeline, model_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(result.metadata, f, indent=2, default=str)


def load_model(model_path: str) -> Pipeline:
    return joblib.load(model_path)


def load_metadata(metadata_path: str) -> dict:
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)
