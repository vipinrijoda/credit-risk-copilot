"""Generate the built-in SYNTHETIC demonstration dataset.

This dataset is NOT real customer data and is NOT sourced from any credit
bureau. It is built with causally sensible relationships so the ML
pipeline, SHAP explanations, and portfolio analytics behave realistically.

Run:
    python scripts/generate_sample_data.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    EMPLOYMENT_TYPES,
    LOAN_PURPOSES,
    RESIDENTIAL_STATUS,
    SAMPLE_DATA_PATH,
)

RNG = np.random.default_rng(42)
N = 6000


def generate() -> pd.DataFrame:
    employment_type = RNG.choice(
        EMPLOYMENT_TYPES, size=N, p=[0.42, 0.20, 0.10, 0.12, 0.10, 0.04, 0.02]
    )
    # Base annual income (INR) varies by employment type.
    income_base = {
        "Salaried": (450000, 260000),
        "Self-Employed": (500000, 400000),
        "Business Owner": (700000, 600000),
        "Professional": (900000, 550000),
        "Government Employee": (550000, 200000),
        "Retired": (300000, 150000),
        "Other": (350000, 200000),
    }
    annual_income = np.array(
        [max(RNG.normal(*income_base[e]), 90000) for e in employment_type]
    )
    annual_income = np.round(annual_income, -2)
    monthly_income = np.round(annual_income / 12, 0)

    employment_duration_years = np.clip(RNG.exponential(4.5, N), 0, 40)
    # Retired people show long historical duration
    employment_duration_years = np.where(
        employment_type == "Retired", np.clip(RNG.normal(25, 6, N), 5, 40), employment_duration_years
    )

    loan_purpose = RNG.choice(LOAN_PURPOSES, size=N,
                               p=[0.28, 0.18, 0.14, 0.06, 0.10, 0.08, 0.08, 0.06, 0.02])
    residential_status = RNG.choice(RESIDENTIAL_STATUS, size=N, p=[0.30, 0.38, 0.22, 0.07, 0.03])

    purpose_loan_multiplier = {
        "Personal Loan": (0.35, 0.15), "Home Loan": (3.5, 1.2), "Vehicle Loan": (0.6, 0.25),
        "Education Loan": (0.9, 0.4), "Business Loan": (1.4, 0.9), "Medical Expenses": (0.25, 0.15),
        "Debt Consolidation": (0.5, 0.25), "Consumer Durable": (0.1, 0.06), "Other": (0.3, 0.2),
    }
    loan_amount = np.array([
        max(annual_income[i] * RNG.normal(*purpose_loan_multiplier[loan_purpose[i]]), 15000)
        for i in range(N)
    ])
    loan_amount = np.round(loan_amount, -3)

    term_choices_by_purpose = {
        "Home Loan": [120, 180, 240, 300, 360], "Vehicle Loan": [24, 36, 48, 60, 84],
        "Education Loan": [36, 60, 84, 120], "Business Loan": [12, 24, 36, 60],
    }
    loan_term_months = np.array([
        RNG.choice(term_choices_by_purpose.get(p, [12, 24, 36, 48, 60]))
        for p in loan_purpose
    ]).astype(float)

    interest_rate = np.clip(RNG.normal(13.0, 3.2, N), 7.0, 32.0)
    interest_rate = np.round(interest_rate, 2)

    credit_history_years = np.clip(RNG.exponential(4.0, N), 0, 35)
    credit_utilization = np.clip(RNG.normal(38, 22, N), 0, 100)
    total_credit_accounts = np.clip(RNG.poisson(3.2, N), 0, 25)
    previous_delinquencies = np.clip(
        RNG.poisson(0.35 + 0.02 * credit_utilization / 10, N), 0, 12
    )
    existing_debt = np.clip(annual_income * RNG.beta(2, 6, N) * 1.5, 0, None)
    existing_emi = np.clip(monthly_income * RNG.beta(2, 8, N) * 1.3, 0, None)

    # --- Derive a latent default-risk score from causally-sensible drivers ---
    dti = existing_debt / np.maximum(annual_income, 1)
    lti = loan_amount / np.maximum(annual_income, 1)
    r_monthly = interest_rate / 1200
    emi_est = np.where(
        r_monthly > 0,
        loan_amount * r_monthly * (1 + r_monthly) ** loan_term_months
        / (((1 + r_monthly) ** loan_term_months) - 1),
        loan_amount / loan_term_months,
    )
    emi_to_income = (emi_est + existing_emi) / np.maximum(monthly_income, 1)

    z = (
        -6.1
        + 2.6 * np.clip(emi_to_income, 0, 3)
        + 1.8 * np.clip(dti, 0, 3)
        + 1.1 * np.clip(lti, 0, 5)
        + 0.022 * credit_utilization
        + 0.55 * previous_delinquencies
        - 0.05 * credit_history_years
        - 0.03 * employment_duration_years
        + 0.045 * (interest_rate - 13.0)
        + RNG.normal(0, 0.65, N)  # noise so the signal isn't perfectly separable
    )
    prob_default = 1 / (1 + np.exp(-z))
    default = (RNG.uniform(0, 1, N) < prob_default).astype(int)

    df = pd.DataFrame({
        "customer_id": [f"CUST{100000 + i}" for i in range(N)],
        "annual_income": annual_income,
        "monthly_income": monthly_income,
        "employment_type": employment_type,
        "employment_duration_years": np.round(employment_duration_years, 1),
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "interest_rate": interest_rate,
        "loan_purpose": loan_purpose,
        "residential_status": residential_status,
        "existing_debt": np.round(existing_debt, -2),
        "existing_emi": np.round(existing_emi, 0),
        "credit_history_years": np.round(credit_history_years, 1),
        "credit_utilization": np.round(credit_utilization, 1),
        "total_credit_accounts": total_credit_accounts,
        "previous_delinquencies": previous_delinquencies,
        "default": default,
    })
    return df


def main() -> None:
    df = generate()
    os.makedirs(os.path.dirname(SAMPLE_DATA_PATH), exist_ok=True)
    df.to_csv(SAMPLE_DATA_PATH, index=False)
    print(f"Wrote {len(df)} rows to {SAMPLE_DATA_PATH}")
    print(f"Default rate: {df['default'].mean():.2%}")


if __name__ == "__main__":
    main()
