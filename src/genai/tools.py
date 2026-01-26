"""Structured analytics functions ("tools") that ground the GenAI Copilot.

CRITICAL ARCHITECTURE RULE:
The LLM NEVER sees raw customer-level rows or the raw uploaded file, and it
NEVER computes statistics itself. Every number the copilot can talk about
must come from one of these functions, which wrap the already-tested
analytics/explainability/evaluation modules. This is what prevents the
copilot from inventing metrics, SHAP values, or portfolio facts.

Uploaded dataset TEXT VALUES are treated strictly as data: they are only
ever summarized into aggregates (counts, percentages, lists of column
names) before being placed in a prompt — never inserted verbatim as
free text that an LLM would "read", which is what defends against prompt
injection via malicious cell contents (see src/genai/prompts.py).
"""

from __future__ import annotations

import json
from typing import Optional

import pandas as pd

from src.analytics.customer import compare_to_portfolio, customer_profile_dict, get_customer_row
from src.analytics.portfolio import compute_kpis, default_rate_by_group, portfolio_summary_text, risk_distribution
from src.data.pii_detector import detect_pii_and_sensitive_columns
from src.data.profiler import profile_dataset
from src.explainability.explain import explain_single_prediction


def get_portfolio_summary(scored_df: pd.DataFrame, threshold: float, target_col: Optional[str] = None) -> dict:
    kpis = compute_kpis(scored_df, threshold, target_col)
    summary = portfolio_summary_text(kpis)
    dist = risk_distribution(scored_df)
    summary["risk_distribution"] = dict(zip(dist["risk_category"], dist["count"].astype(int)))
    return summary


def get_customer_summary(scored_df: pd.DataFrame, customer_id: str, id_col: str = "customer_id") -> dict:
    row = get_customer_row(scored_df, customer_id, id_col)
    if row is None:
        return {"error": f"No customer found with id '{customer_id}'."}
    profile = customer_profile_dict(row)
    result = {
        "customer_id": customer_id,
        "profile": {k: (float(v) if isinstance(v, (int, float)) else str(v)) for k, v in profile.items()},
    }
    if "probability_of_default" in row.index:
        result["probability_of_default_pct"] = round(float(row["probability_of_default"]) * 100, 2)
    if "risk_category" in row.index:
        result["risk_category"] = str(row["risk_category"])
    if "confidence" in row.index:
        result["confidence"] = str(row["confidence"])
    return result


def get_customer_explanation(row: pd.Series, pipeline, raw_feature_cols: list[str]) -> dict:
    single_df = pd.DataFrame([row[raw_feature_cols].to_dict()])
    explanation = explain_single_prediction(single_df, pipeline)
    return {
        "method": explanation.method,
        "top_risk_increasing_factors": [c.label for c in explanation.top_risk_increasing],
        "top_risk_reducing_factors": [c.label for c in explanation.top_risk_reducing],
    }


def get_model_metrics(metadata: dict) -> dict:
    return {
        "model_type": metadata.get("model_type"),
        "is_demo_model": metadata.get("is_demo_model"),
        "data_source_type": metadata.get("data_source_type"),
        "trained_at": metadata.get("trained_at"),
        "threshold": metadata.get("threshold"),
        "threshold_strategy": metadata.get("threshold_strategy"),
        "metrics": metadata.get("metrics"),
        "training_rows": metadata.get("training_rows"),
        "validation_rows": metadata.get("validation_rows"),
    }


def get_risk_distribution(scored_df: pd.DataFrame) -> dict:
    dist = risk_distribution(scored_df)
    return dict(zip(dist["risk_category"], dist["count"].astype(int)))


def get_risk_by_group(scored_df: pd.DataFrame, group_col: str) -> list[dict]:
    grouped = default_rate_by_group(scored_df, group_col)
    grouped = grouped.copy()
    grouped["avg_probability_of_default_pct"] = (grouped["avg_probability_of_default"] * 100).round(2)
    return grouped[[group_col, "avg_probability_of_default_pct", "count"]].to_dict(orient="records")


def get_data_quality_report(df: pd.DataFrame) -> dict:
    profile = profile_dataset(df)
    pii_flags = detect_pii_and_sensitive_columns(df)
    return {
        "n_rows": profile.n_rows,
        "n_cols": profile.n_cols,
        "n_duplicate_rows": profile.n_duplicate_rows,
        "numeric_columns": profile.numeric_columns,
        "categorical_columns": profile.categorical_columns,
        "columns_with_missing_values": [
            {"column": c.name, "pct_missing": round(c.pct_missing * 100, 2)}
            for c in profile.columns if c.n_missing > 0
        ],
        "warnings": profile.warnings,
        "pii_or_sensitive_columns": [{"column": f.column, "kind": f.kind, "reason": f.reason} for f in pii_flags],
    }


TOOL_REGISTRY = {
    "get_portfolio_summary": get_portfolio_summary,
    "get_customer_summary": get_customer_summary,
    "get_customer_explanation": get_customer_explanation,
    "get_model_metrics": get_model_metrics,
    "get_risk_distribution": get_risk_distribution,
    "get_risk_by_group": get_risk_by_group,
    "get_data_quality_report": get_data_quality_report,
}


def facts_to_json(facts: dict) -> str:
    """Serialize computed facts safely for prompt insertion."""
    return json.dumps(facts, indent=2, default=str)
