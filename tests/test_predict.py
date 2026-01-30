import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modeling.train import train_model
from src.modeling.predict import predict_single, predict_batch


def _toy_dataset(n=300, seed=0):
    rng = np.random.default_rng(seed)
    annual_income = rng.normal(600000, 200000, n).clip(100000)
    loan_amount = rng.normal(300000, 150000, n).clip(10000)
    interest_rate = rng.normal(13, 3, n).clip(6, 30)
    loan_term_months = rng.choice([12, 24, 36, 60], n)
    employment_type = rng.choice(["Salaried", "Self-Employed", "Business Owner"], n)
    loan_purpose = rng.choice(["Personal Loan", "Home Loan", "Vehicle Loan"], n)
    credit_utilization = rng.normal(40, 20, n).clip(0, 100)
    z = -2 + 2.5 * (loan_amount / annual_income) + 0.02 * credit_utilization + rng.normal(0, 0.5, n)
    prob = 1 / (1 + np.exp(-z))
    default = (rng.uniform(0, 1, n) < prob).astype(int)
    return pd.DataFrame({
        "annual_income": annual_income, "loan_amount": loan_amount, "interest_rate": interest_rate,
        "loan_term_months": loan_term_months, "employment_type": employment_type,
        "loan_purpose": loan_purpose, "credit_utilization": credit_utilization, "default": default,
    })


def test_train_produces_reasonable_auc():
    df = _toy_dataset()
    result = train_model(df, target_col="default", model_name="Logistic Regression")
    assert result.evaluation.roc_auc > 0.6
    assert 0 < result.metadata["threshold"] < 1


def test_predict_single_matches_batch_for_same_row():
    """Guards against training/inference skew: scoring one customer via
    predict_single must equal the value produced by predict_batch for the
    same row, since both paths go through the identical fitted pipeline."""
    df = _toy_dataset()
    result = train_model(df, target_col="default", model_name="Logistic Regression")
    row = df.drop(columns=["default"]).iloc[0].to_dict()

    single_result = predict_single(row, result.pipeline, result.metadata["threshold"])
    batch_df = predict_batch(df.drop(columns=["default"]).iloc[[0]], result.pipeline, result.metadata["threshold"])

    assert abs(single_result.probability_of_default - batch_df["probability_of_default"].iloc[0]) < 1e-9


def test_ood_flag_for_extreme_value():
    df = _toy_dataset()
    result = train_model(df, target_col="default", model_name="Logistic Regression")
    extreme_customer = df.drop(columns=["default"]).iloc[0].to_dict()
    extreme_customer["annual_income"] = 50_000_000  # absurdly high

    pred = predict_single(
        extreme_customer, result.pipeline, result.metadata["threshold"],
        result.metadata["distribution_reference"],
    )
    assert pred.ood_report.has_warnings
    assert pred.confidence in ("Moderate Confidence", "Low Confidence")


def test_single_class_target_raises():
    import pytest
    df = _toy_dataset()
    df["default"] = 0
    with pytest.raises(Exception):
        train_model(df, target_col="default")
