import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.column_mapper import suggest_mapping_for_column, suggest_mapping_for_dataset


def test_exact_match():
    s = suggest_mapping_for_column("annual_income")
    assert s.suggested_target == "annual_income"
    assert s.method == "exact"
    assert s.confidence == 1.0


def test_alias_match_case_insensitive():
    s = suggest_mapping_for_column("AnnualIncome")
    assert s.suggested_target == "annual_income"
    assert s.method in ("alias", "fuzzy", "exact")


def test_alias_match_applicant_income():
    s = suggest_mapping_for_column("ApplicantIncome")
    assert s.suggested_target == "annual_income"


def test_target_alias_match():
    s = suggest_mapping_for_column("loan_status")
    assert s.suggested_target == "default"


def test_unrecognized_column_returns_none():
    s = suggest_mapping_for_column("xyz_totally_unknown_col_123")
    assert s.suggested_target is None


def test_dataset_level_conflict_resolution():
    suggestions = suggest_mapping_for_dataset(["annual_income", "AnnualIncome", "loan_amnt"])
    # Only one of the two income-like columns should keep the mapping
    income_mapped = [s for s in suggestions if s.suggested_target == "annual_income"]
    assert len(income_mapped) == 1
