"""Portfolio-level analytics.

All numbers here are calculated in pandas/numpy. The GenAI Copilot is only
ever given the OUTPUT of these functions (see src/genai/tools.py) — it
never computes statistics itself, per the project's hallucination-prevention
requirement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class PortfolioKPIs:
    total_customers: int
    avg_predicted_default_probability: float
    high_risk_customers: int
    very_high_risk_customers: int
    avg_loan_amount: float
    total_loan_exposure: float
    observed_default_rate: Optional[float] = None
    predicted_default_rate: Optional[float] = None


def compute_kpis(scored_df: pd.DataFrame, threshold: float, target_col: Optional[str] = None) -> PortfolioKPIs:
    """`scored_df` must already contain `probability_of_default` and `risk_category`."""
    total = len(scored_df)
    kpis = PortfolioKPIs(
        total_customers=total,
        avg_predicted_default_probability=float(scored_df["probability_of_default"].mean()) if total else 0.0,
        high_risk_customers=int((scored_df["risk_category"] == "HIGH").sum()),
        very_high_risk_customers=int((scored_df["risk_category"] == "VERY HIGH").sum()),
        avg_loan_amount=float(scored_df["loan_amount"].mean()) if "loan_amount" in scored_df.columns and total else 0.0,
        total_loan_exposure=float(scored_df["loan_amount"].sum()) if "loan_amount" in scored_df.columns else 0.0,
        predicted_default_rate=float((scored_df["probability_of_default"] >= threshold).mean()) if total else 0.0,
    )
    if target_col and target_col in scored_df.columns:
        kpis.observed_default_rate = float(pd.to_numeric(scored_df[target_col], errors="coerce").mean())
    return kpis


def risk_distribution(scored_df: pd.DataFrame) -> pd.DataFrame:
    order = ["LOW", "MEDIUM", "HIGH", "VERY HIGH"]
    counts = scored_df["risk_category"].value_counts().reindex(order, fill_value=0)
    return pd.DataFrame({"risk_category": counts.index, "count": counts.values})


def default_rate_by_group(scored_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in scored_df.columns:
        return pd.DataFrame(columns=[group_col, "avg_probability_of_default", "count"])
    grouped = (
        scored_df.groupby(group_col)["probability_of_default"]
        .agg(avg_probability_of_default="mean", count="count")
        .reset_index()
        .sort_values("avg_probability_of_default", ascending=False)
    )
    return grouped


def numeric_distribution_summary(scored_df: pd.DataFrame, col: str) -> dict:
    if col not in scored_df.columns:
        return {}
    series = pd.to_numeric(scored_df[col], errors="coerce").dropna()
    if series.empty:
        return {}
    return {
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max()),
        "p25": float(series.quantile(0.25)),
        "p75": float(series.quantile(0.75)),
    }


def portfolio_summary_text(kpis: PortfolioKPIs) -> dict:
    """Structured, LLM-ready summary dict (facts only, no narrative)."""
    summary = {
        "total_customers": kpis.total_customers,
        "avg_predicted_default_probability_pct": round(kpis.avg_predicted_default_probability * 100, 2),
        "high_risk_customers": kpis.high_risk_customers,
        "very_high_risk_customers": kpis.very_high_risk_customers,
        "avg_loan_amount_inr": round(kpis.avg_loan_amount, 2),
        "total_loan_exposure_inr": round(kpis.total_loan_exposure, 2),
        "predicted_default_rate_pct": round((kpis.predicted_default_rate or 0) * 100, 2),
    }
    if kpis.observed_default_rate is not None:
        summary["observed_default_rate_pct"] = round(kpis.observed_default_rate * 100, 2)
    return summary
