"""Dataset profiling utilities: shape, dtypes, missingness, duplicates,
descriptive statistics and class-imbalance checks.

These functions calculate FACTS about a dataframe. They are used both by
the Streamlit "Data Quality" page and as grounded inputs for the GenAI
Copilot (never let an LLM invent these numbers).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    inferred_type: str  # "numeric" | "categorical" | "datetime" | "boolean" | "unknown"
    n_missing: int
    pct_missing: float
    n_unique: int
    sample_values: list


@dataclass
class DataProfile:
    n_rows: int
    n_cols: int
    n_duplicate_rows: int
    columns: list[ColumnProfile]
    numeric_columns: list[str]
    categorical_columns: list[str]
    describe_numeric: pd.DataFrame
    describe_categorical: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def _infer_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    # Heuristic: object column that is actually numeric-like (e.g. "12,345")
    non_null = series.dropna().astype(str).str.replace(",", "", regex=False)
    if len(non_null) > 0:
        numeric_like = pd.to_numeric(non_null, errors="coerce")
        if numeric_like.notna().mean() > 0.9:
            return "numeric"
    return "categorical"


def profile_dataset(df: pd.DataFrame) -> DataProfile:
    """Compute a comprehensive profile of a raw uploaded dataset."""
    warnings: list[str] = []
    n_rows, n_cols = df.shape
    n_dupes = int(df.duplicated().sum())
    if n_dupes > 0:
        warnings.append(f"{n_dupes} duplicate row(s) detected ({n_dupes / n_rows:.1%} of rows).")

    columns: list[ColumnProfile] = []
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in df.columns:
        series = df[col]
        inferred = _infer_type(series)
        n_missing = int(series.isna().sum())
        pct_missing = n_missing / n_rows if n_rows else 0.0
        n_unique = int(series.nunique(dropna=True))
        sample_values = series.dropna().unique()[:5].tolist()

        columns.append(
            ColumnProfile(
                name=col,
                dtype=str(series.dtype),
                inferred_type=inferred,
                n_missing=n_missing,
                pct_missing=pct_missing,
                n_unique=n_unique,
                sample_values=sample_values,
            )
        )

        if inferred == "numeric":
            numeric_cols.append(col)
        elif inferred in ("categorical", "boolean"):
            categorical_cols.append(col)

        if pct_missing > 0.4:
            warnings.append(f"Column '{col}' has {pct_missing:.1%} missing values — consider dropping or imputing carefully.")
        if inferred == "categorical" and n_unique == n_rows and n_rows > 20:
            warnings.append(f"Column '{col}' looks like a unique identifier ({n_unique} unique values) — exclude it from modeling.")

    describe_numeric = df[numeric_cols].describe().T if numeric_cols else pd.DataFrame()
    describe_categorical = (
        df[categorical_cols].describe().T if categorical_cols else pd.DataFrame()
    )

    if n_rows < 200:
        warnings.append(f"Dataset is small ({n_rows} rows). Model training results may be unstable.")

    return DataProfile(
        n_rows=n_rows,
        n_cols=n_cols,
        n_duplicate_rows=n_dupes,
        columns=columns,
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        describe_numeric=describe_numeric,
        describe_categorical=describe_categorical,
        warnings=warnings,
    )


def class_balance_report(df: pd.DataFrame, target_col: str) -> dict:
    """Report class distribution and imbalance warnings for a target column."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    counts = df[target_col].value_counts(dropna=False)
    proportions = df[target_col].value_counts(normalize=True, dropna=False)
    n_classes = counts.shape[0]

    warnings: list[str] = []
    if n_classes < 2:
        warnings.append("Target column has only one class — a classifier cannot be trained.")
    elif proportions.min() < 0.05:
        warnings.append(
            f"Severe class imbalance detected: minority class is only "
            f"{proportions.min():.1%} of rows. Consider class weighting or "
            f"gathering more minority-class examples."
        )
    elif proportions.min() < 0.15:
        warnings.append(
            f"Moderate class imbalance detected: minority class is "
            f"{proportions.min():.1%} of rows."
        )

    return {
        "counts": counts.to_dict(),
        "proportions": proportions.to_dict(),
        "n_classes": n_classes,
        "warnings": warnings,
    }
