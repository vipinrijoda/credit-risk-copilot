"""Input validation utilities.

Used by the "Assess New Customer" form and by dataset-quality checks. Keeps
all range/type checks against the central schema in `src/config.py` so
validation rules cannot drift from the training-time feature definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.config import CANONICAL_FEATURES, FEATURE_BY_NAME, FieldType


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cleaned: dict = field(default_factory=dict)


def validate_customer_input(raw_input: dict) -> ValidationResult:
    """Validate and lightly clean a manually-entered customer's fields
    against the canonical schema's declared ranges/categories."""
    errors: list[str] = []
    warnings: list[str] = []
    cleaned: dict = {}

    for feature in CANONICAL_FEATURES:
        if feature.name not in raw_input:
            continue
        value = raw_input[feature.name]

        if feature.field_type == FieldType.NUMERIC:
            try:
                value = float(value)
            except (TypeError, ValueError):
                errors.append(f"'{feature.label}' must be a number.")
                continue

            if feature.min_value is not None and value < feature.min_value:
                errors.append(
                    f"'{feature.label}' = {value} is below the minimum plausible value "
                    f"({feature.min_value})."
                )
                continue
            if feature.max_value is not None and value > feature.max_value:
                warnings.append(
                    f"'{feature.label}' = {value} is unusually high "
                    f"(above {feature.max_value}); please double-check."
                )
            cleaned[feature.name] = value

        elif feature.field_type == FieldType.CATEGORICAL:
            value_str = str(value)
            if feature.categories and value_str not in feature.categories:
                warnings.append(
                    f"'{feature.label}' = '{value_str}' is not one of the standard categories "
                    f"({', '.join(feature.categories)})."
                )
            cleaned[feature.name] = value_str

    # Cross-field sanity checks
    if "loan_amount" in cleaned and "annual_income" in cleaned and cleaned["annual_income"] > 0:
        ratio = cleaned["loan_amount"] / cleaned["annual_income"]
        if ratio > 10:
            warnings.append(
                f"Loan amount is {ratio:.1f}x the applicant's annual income — please verify this is correct."
            )

    if "interest_rate" in cleaned and cleaned["interest_rate"] <= 0:
        errors.append("Interest rate must be greater than zero.")

    if "loan_term_months" in cleaned and cleaned["loan_term_months"] <= 0:
        errors.append("Loan tenure must be greater than zero months.")

    missing_required = [
        f.label for f in CANONICAL_FEATURES if f.required and f.name not in cleaned
    ]
    if missing_required:
        errors.append(f"Missing required field(s): {', '.join(missing_required)}.")

    return ValidationResult(is_valid=(len(errors) == 0), errors=errors, warnings=warnings, cleaned=cleaned)
