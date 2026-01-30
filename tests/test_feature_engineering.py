import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modeling.feature_engineering import FinancialFeatureEngineer, compute_estimated_emi


def test_full_schema_produces_all_engineered_columns():
    df = pd.DataFrame([{
        "annual_income": 600000, "monthly_income": 50000, "loan_amount": 300000,
        "interest_rate": 12.0, "loan_term_months": 36, "existing_debt": 100000,
        "existing_emi": 5000,
    }])
    out = FinancialFeatureEngineer().fit_transform(df)
    for col in ("loan_to_income_ratio", "estimated_monthly_emi", "emi_to_income_ratio", "debt_to_income_ratio"):
        assert col in out.columns
        assert not pd.isna(out[col].iloc[0])


def test_missing_optional_column_does_not_crash():
    df = pd.DataFrame([{"annual_income": 600000, "loan_amount": 300000}])
    out = FinancialFeatureEngineer().fit_transform(df)
    # debt_to_income_ratio needs existing_debt, which is absent -> NaN, no crash
    assert pd.isna(out["debt_to_income_ratio"].iloc[0])
    assert out["loan_to_income_ratio"].iloc[0] == 0.5


def test_zero_income_does_not_divide_by_zero_crash():
    df = pd.DataFrame([{"annual_income": 0, "loan_amount": 300000, "existing_debt": 1000}])
    out = FinancialFeatureEngineer().fit_transform(df)
    assert pd.isna(out["loan_to_income_ratio"].iloc[0]) or np.isinf(out["loan_to_income_ratio"].iloc[0]) is False


def test_emi_formula_reasonable():
    emi = compute_estimated_emi(pd.Series([100000]), pd.Series([12.0]), pd.Series([12]))
    # A 1-year, 12%-p.a. loan of 100000 should have EMI roughly 8880-8900
    assert 8500 < emi.iloc[0] < 9200


def test_emi_zero_rate_uses_simple_division():
    emi = compute_estimated_emi(pd.Series([120000]), pd.Series([0.0]), pd.Series([12]))
    assert emi.iloc[0] == 10000


def test_emi_invalid_term_returns_nan():
    emi = compute_estimated_emi(pd.Series([100000]), pd.Series([12.0]), pd.Series([0]))
    assert pd.isna(emi.iloc[0])
