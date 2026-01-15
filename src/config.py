"""
Central schema & configuration for the Indian Credit Risk Copilot.

This module is the SINGLE SOURCE OF TRUTH for the application's feature
schema. Every other part of the system (dataset column mapping, feature
engineering, model training, prediction, the manual "new customer" form,
input validation and model metadata) reads from the definitions here.

Do NOT duplicate feature lists elsewhere. If a field needs to change,
change it here only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class FieldType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


@dataclass(frozen=True)
class FeatureField:
    """Definition of a single canonical feature used by the application."""

    name: str                      # canonical column name, e.g. "annual_income"
    label: str                     # human readable label, e.g. "Annual Income (₹)"
    field_type: FieldType
    required: bool                 # required for training a NEW model on uploaded data
    aliases: tuple[str, ...] = field(default_factory=tuple)
    categories: tuple[str, ...] = field(default_factory=tuple)  # for categorical fields
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    default_value: Any = None
    help_text: str = ""
    is_engineered: bool = False    # True if this is derived, not raw-input


# ---------------------------------------------------------------------------
# Domain vocabularies (India-first lending context)
# ---------------------------------------------------------------------------

EMPLOYMENT_TYPES: tuple[str, ...] = (
    "Salaried",
    "Self-Employed",
    "Business Owner",
    "Professional",
    "Government Employee",
    "Retired",
    "Other",
)

LOAN_PURPOSES: tuple[str, ...] = (
    "Personal Loan",
    "Home Loan",
    "Vehicle Loan",
    "Education Loan",
    "Business Loan",
    "Medical Expenses",
    "Debt Consolidation",
    "Consumer Durable",
    "Other",
)

RESIDENTIAL_STATUS: tuple[str, ...] = (
    "Owned",
    "Rented",
    "Living with Family",
    "Company Provided",
    "Other",
)

TARGET_COLUMN = "default"
TARGET_ALIASES: tuple[str, ...] = (
    "default",
    "loan_default",
    "bad_loan",
    "loan_status",
    "target",
    "is_default",
    "defaulted",
    "class",
    "label",
)

# ---------------------------------------------------------------------------
# Canonical raw-input schema
# ---------------------------------------------------------------------------
# NOTE: `annual_income` is treated as the primary income field. If a dataset
# only has monthly income, the column mapper / adapters derive annual_income
# = monthly_income * 12 (see src/data_adapters/generic_adapter.py).

CANONICAL_FEATURES: tuple[FeatureField, ...] = (
    FeatureField(
        name="annual_income",
        label="Annual Income (₹)",
        field_type=FieldType.NUMERIC,
        required=True,
        aliases=("annualincome", "annual_inc", "income", "applicantincome",
                  "gross_annual_income", "yearly_income"),
        min_value=0,
        max_value=1e8,
        default_value=600000.0,
        help_text="Total gross annual income of the applicant, in INR.",
    ),
    FeatureField(
        name="monthly_income",
        label="Monthly Income (₹)",
        field_type=FieldType.NUMERIC,
        required=False,
        aliases=("monthlyincome", "monthly_inc", "net_monthly_income"),
        min_value=0,
        max_value=1e7,
        default_value=None,
        help_text="Optional. If provided without annual_income, annual income is derived as monthly x 12.",
    ),
    FeatureField(
        name="employment_type",
        label="Employment Type",
        field_type=FieldType.CATEGORICAL,
        required=True,
        aliases=("employmenttype", "employment_status", "occupation_type", "job_type"),
        categories=EMPLOYMENT_TYPES,
        default_value="Salaried",
    ),
    FeatureField(
        name="employment_duration_years",
        label="Employment Duration (Years)",
        field_type=FieldType.NUMERIC,
        required=True,
        aliases=("employmentduration", "years_employed", "job_tenure", "emp_length"),
        min_value=0,
        max_value=50,
        default_value=3.0,
    ),
    FeatureField(
        name="loan_amount",
        label="Loan Amount (₹)",
        field_type=FieldType.NUMERIC,
        required=True,
        aliases=("loanamount", "loan_amnt", "principal", "sanctioned_amount"),
        min_value=1000,
        max_value=1e8,
        default_value=300000.0,
    ),
    FeatureField(
        name="loan_term_months",
        label="Loan Tenure (Months)",
        field_type=FieldType.NUMERIC,
        required=True,
        aliases=("loanterm", "term", "tenure_months", "loan_tenure"),
        min_value=1,
        max_value=480,
        default_value=36.0,
    ),
    FeatureField(
        name="interest_rate",
        label="Interest Rate (% p.a.)",
        field_type=FieldType.NUMERIC,
        required=True,
        aliases=("interestrate", "int_rate", "rate_of_interest", "roi"),
        min_value=0.1,
        max_value=60.0,
        default_value=12.5,
    ),
    FeatureField(
        name="loan_purpose",
        label="Loan Purpose",
        field_type=FieldType.CATEGORICAL,
        required=True,
        aliases=("loanpurpose", "purpose", "loan_reason"),
        categories=LOAN_PURPOSES,
        default_value="Personal Loan",
    ),
    FeatureField(
        name="residential_status",
        label="Residential Status",
        field_type=FieldType.CATEGORICAL,
        required=False,
        aliases=("residentialstatus", "home_ownership", "housing"),
        categories=RESIDENTIAL_STATUS,
        default_value="Rented",
    ),
    FeatureField(
        name="existing_debt",
        label="Total Outstanding Debt (₹)",
        field_type=FieldType.NUMERIC,
        required=False,
        aliases=("existingdebt", "total_outstanding_debt", "current_debt", "outstanding_balance"),
        min_value=0,
        max_value=1e8,
        default_value=0.0,
    ),
    FeatureField(
        name="existing_emi",
        label="Existing EMI Obligations (₹/month)",
        field_type=FieldType.NUMERIC,
        required=False,
        aliases=("existingemi", "monthly_emi", "current_emi"),
        min_value=0,
        max_value=1e6,
        default_value=0.0,
    ),
    FeatureField(
        name="credit_history_years",
        label="Credit History Length (Years)",
        field_type=FieldType.NUMERIC,
        required=False,
        aliases=("credithistory", "credit_history_length", "cred_hist_years"),
        min_value=0,
        max_value=60,
        default_value=3.0,
    ),
    FeatureField(
        name="credit_utilization",
        label="Credit Utilization (%)",
        field_type=FieldType.NUMERIC,
        required=False,
        aliases=("creditutilization", "revol_util", "utilization"),
        min_value=0,
        max_value=100,
        default_value=30.0,
    ),
    FeatureField(
        name="total_credit_accounts",
        label="Number of Credit Accounts",
        field_type=FieldType.NUMERIC,
        required=False,
        aliases=("totalcreditaccounts", "num_credit_accounts", "open_acc", "total_acc"),
        min_value=0,
        max_value=100,
        default_value=3.0,
    ),
    FeatureField(
        name="previous_delinquencies",
        label="Previous Delinquencies (count)",
        field_type=FieldType.NUMERIC,
        required=False,
        aliases=("previousdelinquencies", "delinq_2yrs", "num_delinquencies"),
        min_value=0,
        max_value=50,
        default_value=0.0,
    ),
)

# ---------------------------------------------------------------------------
# Engineered features (produced by src/modeling/feature_engineering.py)
# ---------------------------------------------------------------------------

ENGINEERED_FEATURES: tuple[FeatureField, ...] = (
    FeatureField(
        name="loan_to_income_ratio", label="Loan-to-Income Ratio",
        field_type=FieldType.NUMERIC, required=False, is_engineered=True,
        help_text="loan_amount / annual_income",
    ),
    FeatureField(
        name="estimated_monthly_emi", label="Estimated Monthly EMI (₹)",
        field_type=FieldType.NUMERIC, required=False, is_engineered=True,
        help_text="Amortized EMI computed from loan_amount, interest_rate, loan_term_months",
    ),
    FeatureField(
        name="emi_to_income_ratio", label="EMI-to-Income Ratio",
        field_type=FieldType.NUMERIC, required=False, is_engineered=True,
        help_text="(estimated_monthly_emi + existing_emi) / monthly_income",
    ),
    FeatureField(
        name="debt_to_income_ratio", label="Debt-to-Income Ratio",
        field_type=FieldType.NUMERIC, required=False, is_engineered=True,
        help_text="existing_debt / annual_income",
    ),
)

ALL_FEATURES: tuple[FeatureField, ...] = CANONICAL_FEATURES + ENGINEERED_FEATURES

NUMERIC_RAW_FEATURES = tuple(f.name for f in CANONICAL_FEATURES if f.field_type == FieldType.NUMERIC)
CATEGORICAL_RAW_FEATURES = tuple(f.name for f in CANONICAL_FEATURES if f.field_type == FieldType.CATEGORICAL)
REQUIRED_RAW_FEATURES = tuple(f.name for f in CANONICAL_FEATURES if f.required)
FEATURE_BY_NAME = {f.name: f for f in ALL_FEATURES}

# Sensitive attributes that must NEVER be used as model features by default.
SENSITIVE_ATTRIBUTE_KEYWORDS: tuple[str, ...] = (
    "religion", "caste", "race", "ethnicity", "ethnic", "political",
    "political_affiliation", "gender", "sex", "nationality", "tribe",
)

# Direct-identifier PII keywords that must never be collected/trained on.
PII_KEYWORDS: tuple[str, ...] = (
    "name", "email", "phone", "mobile", "pan", "aadhaar", "aadhar",
    "account_number", "acc_no", "card_number", "cardnumber", "password",
    "otp", "address", "ifsc", "passport", "voter_id",
)

# Columns that typically encode information only known AFTER loan issuance /
# after repayment behaviour, and therefore risk target leakage.
LEAKAGE_KEYWORDS: tuple[str, ...] = (
    "final_payment", "recovery_amount", "collection_amount",
    "loan_status_after_default", "total_received", "last_payment_date",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee", "out_prncp",
    "settlement_amount", "charged_off_amount", "days_past_due_final",
)

# ---------------------------------------------------------------------------
# Risk bands (configurable). Probabilities are of DEFAULT.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RiskBand:
    name: str
    lower: float  # inclusive
    upper: float  # exclusive (1.0 for last band)
    color: str


RISK_BANDS: tuple[RiskBand, ...] = (
    RiskBand("LOW", 0.00, 0.15, "#2E7D32"),
    RiskBand("MEDIUM", 0.15, 0.35, "#F9A825"),
    RiskBand("HIGH", 0.35, 0.60, "#EF6C00"),
    RiskBand("VERY HIGH", 0.60, 1.01, "#C62828"),
)


def get_risk_band(probability: float) -> RiskBand:
    for band in RISK_BANDS:
        if band.lower <= probability < band.upper:
            return band
    return RISK_BANDS[-1]


# ---------------------------------------------------------------------------
# Confidence configuration
# ---------------------------------------------------------------------------

CONFIDENCE_LEVELS = ("High Confidence", "Moderate Confidence", "Low Confidence")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_SAMPLE_DIR = os.path.join(BASE_DIR, "data", "sample")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
DEFAULT_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "default_model.joblib")
DEFAULT_MODEL_METADATA_PATH = os.path.join(ARTIFACTS_DIR, "default_model_metadata.json")
SAMPLE_DATA_PATH = os.path.join(DATA_SAMPLE_DIR, "credit_data.csv")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Disclaimers (single source of truth for legal/regulatory text)
# ---------------------------------------------------------------------------

DISCLAIMER_SHORT = (
    "This is a decision-support analytics tool, not an automated approval "
    "system. Predictions are model estimates, not certainties, and must not "
    "replace professional lending judgement."
)

DISCLAIMER_FULL = (
    "This application is an educational and analytical demonstration. It is "
    "not a regulated credit scoring system, does not access CIBIL, Experian "
    "India, Equifax India, CRIF High Mark, or any other credit bureau, bank, "
    "PAN, Aadhaar, or GST system, and should not be used as the sole basis "
    "for lending decisions. Production deployment would require legal, "
    "regulatory, privacy, security, validation, and governance review."
)

PRIVACY_NOTICE = (
    "Uploaded datasets are processed locally within this application "
    "session/environment and are not permanently stored by default. Please "
    "do not upload PAN numbers, Aadhaar numbers, bank account numbers, card "
    "numbers, passwords, or OTPs."
)

DEMO_DATA_DISCLAIMER = (
    "The included demonstration dataset is SYNTHETIC. It is generated to "
    "resemble Indian retail-lending applications for teaching purposes only. "
    "It is not real customer data, is not sourced from any credit bureau, "
    "and must not be interpreted as a model trained on real Indian credit "
    "bureau data."
)
