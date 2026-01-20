"""Feature engineering as a proper sklearn-compatible transformer.

This transformer computes financially meaningful ratios (loan-to-income,
EMI-to-income, debt-to-income, estimated EMI) using the standard amortized
loan payment formula. It is DELIBERATELY robust to missing optional inputs:
if a required raw column for a derived feature is absent, that derived
feature is simply skipped (filled with NaN, which downstream imputers
handle) rather than crashing the pipeline.

Being a transformer (not ad-hoc pandas code inside Streamlit) guarantees
the exact same feature engineering logic runs during training AND at
prediction time for built-in, uploaded, and manually-entered customers —
eliminating training/inference skew.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    return numerator / denom


def compute_estimated_emi(loan_amount: pd.Series, interest_rate: pd.Series,
                           loan_term_months: pd.Series) -> pd.Series:
    """Standard amortized EMI formula, protected against division-by-zero,
    zero/negative terms, and invalid rates."""
    principal = pd.to_numeric(loan_amount, errors="coerce")
    annual_rate = pd.to_numeric(interest_rate, errors="coerce")
    term = pd.to_numeric(loan_term_months, errors="coerce")

    r = (annual_rate / 100.0) / 12.0
    r = r.where((r.notna()) & (r > 0), np.nan)
    term_safe = term.where((term.notna()) & (term > 0), np.nan)

    factor = (1 + r) ** term_safe
    denom = factor - 1
    emi = principal * r * factor / denom

    # Fallback for zero/near-zero interest rate: simple amortization.
    zero_rate_mask = (r.isna() | (r == 0)) & term_safe.notna() & (term_safe > 0)
    emi = emi.where(~zero_rate_mask, principal / term_safe)

    return emi


class FinancialFeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds engineered financial ratio features to a canonical-schema dataframe."""

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        df = X.copy()

        has = lambda col: col in df.columns  # noqa: E731

        if has("loan_amount") and has("annual_income"):
            df["loan_to_income_ratio"] = _safe_div(
                pd.to_numeric(df["loan_amount"], errors="coerce"),
                pd.to_numeric(df["annual_income"], errors="coerce"),
            )
        else:
            df["loan_to_income_ratio"] = np.nan

        if has("loan_amount") and has("interest_rate") and has("loan_term_months"):
            df["estimated_monthly_emi"] = compute_estimated_emi(
                df["loan_amount"], df["interest_rate"], df["loan_term_months"]
            )
        else:
            df["estimated_monthly_emi"] = np.nan

        if has("monthly_income"):
            monthly_income = pd.to_numeric(df["monthly_income"], errors="coerce")
        elif has("annual_income"):
            monthly_income = pd.to_numeric(df["annual_income"], errors="coerce") / 12.0
        else:
            monthly_income = pd.Series(np.nan, index=df.index)

        existing_emi = (
            pd.to_numeric(df["existing_emi"], errors="coerce") if has("existing_emi")
            else pd.Series(0.0, index=df.index)
        )
        total_emi = df["estimated_monthly_emi"].fillna(0) + existing_emi.fillna(0)
        df["emi_to_income_ratio"] = _safe_div(total_emi, monthly_income)

        if has("existing_debt") and has("annual_income"):
            df["debt_to_income_ratio"] = _safe_div(
                pd.to_numeric(df["existing_debt"], errors="coerce"),
                pd.to_numeric(df["annual_income"], errors="coerce"),
            )
        else:
            df["debt_to_income_ratio"] = np.nan

        return df

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else []
        engineered = [
            "loan_to_income_ratio", "estimated_monthly_emi",
            "emi_to_income_ratio", "debt_to_income_ratio",
        ]
        return np.array(base + [f for f in engineered if f not in base])
