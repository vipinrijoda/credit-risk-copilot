import sys
import os

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.leakage_detector import detect_keyword_leakage, detect_leakage
from src.data.pii_detector import detect_pii_and_sensitive_columns


def test_keyword_leakage_detected():
    flags = detect_keyword_leakage(["annual_income", "recoveries", "loan_amount", "final_payment"])
    flagged_cols = {f.column for f in flags}
    assert "recoveries" in flagged_cols
    assert "final_payment" in flagged_cols
    assert "annual_income" not in flagged_cols


def test_correlation_leakage_detected():
    df = pd.DataFrame({
        "default": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "suspicious_col": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],  # identical to target
        "annual_income": [500000, 300000, 700000, 250000, 600000, 200000, 550000, 280000, 620000, 310000],
    })
    flags = detect_leakage(df, target_col="default")
    flagged_cols = {f.column for f in flags}
    assert "suspicious_col" in flagged_cols


def test_pii_keyword_detection():
    df = pd.DataFrame({"customer_name": ["A"], "pan_number": ["ABCDE1234F"], "annual_income": [500000]})
    flags = detect_pii_and_sensitive_columns(df)
    flagged_cols = {f.column for f in flags}
    assert "customer_name" in flagged_cols
    assert "pan_number" in flagged_cols
    assert "annual_income" not in flagged_cols


def test_sensitive_attribute_detection():
    df = pd.DataFrame({"religion": ["A"], "annual_income": [500000]})
    flags = detect_pii_and_sensitive_columns(df)
    kinds = {f.column: f.kind for f in flags}
    assert kinds.get("religion") == "sensitive_attribute"
