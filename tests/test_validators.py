import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.validators import validate_customer_input


VALID_INPUT = {
    "annual_income": 600000,
    "employment_type": "Salaried",
    "employment_duration_years": 3,
    "loan_amount": 300000,
    "loan_term_months": 36,
    "interest_rate": 12.5,
    "loan_purpose": "Personal Loan",
}


def test_valid_input_passes():
    result = validate_customer_input(VALID_INPUT)
    assert result.is_valid
    assert not result.errors


def test_missing_required_field_fails():
    incomplete = {k: v for k, v in VALID_INPUT.items() if k != "loan_amount"}
    result = validate_customer_input(incomplete)
    assert not result.is_valid
    assert any("loan_amount" in e.lower() or "loan amount" in e.lower() for e in result.errors)


def test_non_numeric_value_fails():
    bad = dict(VALID_INPUT)
    bad["annual_income"] = "not a number"
    result = validate_customer_input(bad)
    assert not result.is_valid


def test_negative_income_fails():
    bad = dict(VALID_INPUT)
    bad["annual_income"] = -5000
    result = validate_customer_input(bad)
    assert not result.is_valid


def test_zero_interest_rate_fails():
    bad = dict(VALID_INPUT)
    bad["interest_rate"] = 0
    result = validate_customer_input(bad)
    assert not result.is_valid


def test_high_loan_to_income_generates_warning_not_error():
    bad = dict(VALID_INPUT)
    bad["loan_amount"] = 8000000  # >10x income
    result = validate_customer_input(bad)
    assert result.is_valid
    assert any("annual income" in w.lower() for w in result.warnings)


def test_unusual_category_generates_warning():
    bad = dict(VALID_INPUT)
    bad["employment_type"] = "Freelance Astronaut"
    result = validate_customer_input(bad)
    assert result.is_valid
    assert result.warnings
