"""Out-of-distribution (OOD) detection and confidence estimation.

OOD detection never blocks a prediction — it only lowers the reported
confidence and surfaces a warning, per project requirement #18/#19.
Confidence reflects "how similar is this input to the training data",
NOT "how certain are we that this person will default".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.config import CONFIDENCE_LEVELS


@dataclass
class OODReport:
    numeric_warnings: list[str] = field(default_factory=list)
    categorical_warnings: list[str] = field(default_factory=list)
    n_missing_inputs: int = 0
    ood_signal_count: int = 0

    @property
    def has_warnings(self) -> bool:
        return bool(self.numeric_warnings or self.categorical_warnings)


def check_out_of_distribution(customer: dict, distribution_reference: dict) -> OODReport:
    report = OODReport()
    numeric_ref = distribution_reference.get("numeric", {})
    categorical_ref = distribution_reference.get("categorical", {})

    for col, bounds in numeric_ref.items():
        value = customer.get(col)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            report.n_missing_inputs += 1
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if value < bounds["p1"] or value > bounds["p99"]:
            report.numeric_warnings.append(
                f"'{col}' = {value:,.2f} is outside the typical training-data range "
                f"({bounds['p1']:,.2f} – {bounds['p99']:,.2f})."
            )
            report.ood_signal_count += 1

    for col, known_categories in categorical_ref.items():
        value = customer.get(col)
        if value is None:
            report.n_missing_inputs += 1
            continue
        if str(value) not in known_categories:
            report.categorical_warnings.append(
                f"'{col}' = '{value}' was not seen during training "
                f"(known values: {', '.join(known_categories[:6])}"
                f"{'...' if len(known_categories) > 6 else ''})."
            )
            report.ood_signal_count += 1

    return report


def estimate_confidence(ood_report: OODReport, model_probability: float) -> str:
    """Combine OOD signals, missing inputs, and model-probability uncertainty
    (how close the probability is to the decision boundary) into a single
    human-readable confidence level."""
    score = 0  # higher score = lower confidence

    score += ood_report.ood_signal_count * 2
    score += ood_report.n_missing_inputs

    # Probabilities near 0.5 indicate the model itself is less decisive.
    distance_from_boundary = abs(model_probability - 0.5)
    if distance_from_boundary < 0.05:
        score += 3
    elif distance_from_boundary < 0.15:
        score += 1

    if score == 0:
        return CONFIDENCE_LEVELS[0]  # High
    if score <= 3:
        return CONFIDENCE_LEVELS[1]  # Moderate
    return CONFIDENCE_LEVELS[2]  # Low
