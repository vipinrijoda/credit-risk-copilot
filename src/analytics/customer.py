"""Per-customer analytics: profile lookups and portfolio comparisons.

Like `portfolio.py`, every function here returns calculated facts that can
be safely handed to the GenAI Copilot for narration.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.config import FEATURE_BY_NAME


def get_customer_row(df: pd.DataFrame, customer_id: str, id_col: str = "customer_id") -> Optional[pd.Series]:
    if id_col not in df.columns:
        return None
    matches = df[df[id_col].astype(str) == str(customer_id)]
    if matches.empty:
        return None
    return matches.iloc[0]


def customer_profile_dict(row: pd.Series) -> dict:
    """Human-readable profile: {label: value} for known canonical fields present."""
    profile = {}
    for name, feature in FEATURE_BY_NAME.items():
        if name in row.index and pd.notna(row[name]):
            profile[feature.label] = row[name]
    return profile


def compare_to_portfolio(row: pd.Series, portfolio_df: pd.DataFrame, numeric_cols: list[str]) -> dict:
    """For each numeric column, compare the customer's value to the
    portfolio mean/median and report the percentile rank."""
    comparison = {}
    for col in numeric_cols:
        if col not in row.index or col not in portfolio_df.columns:
            continue
        value = row[col]
        if pd.isna(value):
            continue
        series = pd.to_numeric(portfolio_df[col], errors="coerce").dropna()
        if series.empty:
            continue
        percentile = float((series < value).mean() * 100)
        comparison[col] = {
            "customer_value": float(value),
            "portfolio_mean": float(series.mean()),
            "portfolio_median": float(series.median()),
            "percentile_rank": round(percentile, 1),
        }
    return comparison
