"""The ONE prediction function used for every customer source:
built-in dataset rows, uploaded-dataset rows, and manually-entered new
customers. This is what guarantees no training/inference skew (req #39).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import get_risk_band
from src.modeling.ood import OODReport, check_out_of_distribution, estimate_confidence


@dataclass
class PredictionResult:
    probability_of_default: float
    risk_category: str
    risk_color: str
    confidence: str
    ood_report: OODReport
    threshold_used: float


def predict_single(
    customer: dict,
    pipeline: Pipeline,
    threshold: float,
    distribution_reference: Optional[dict] = None,
) -> PredictionResult:
    """Score a single customer (given as a flat dict of canonical fields)."""
    df = pd.DataFrame([customer])
    proba = float(pipeline.predict_proba(df)[:, 1][0])

    ood_report = (
        check_out_of_distribution(customer, distribution_reference)
        if distribution_reference else OODReport()
    )
    confidence = estimate_confidence(ood_report, proba)
    band = get_risk_band(proba)

    return PredictionResult(
        probability_of_default=proba,
        risk_category=band.name,
        risk_color=band.color,
        confidence=confidence,
        ood_report=ood_report,
        threshold_used=threshold,
    )


def predict_batch(
    df: pd.DataFrame,
    pipeline: Pipeline,
    threshold: float,
    distribution_reference: Optional[dict] = None,
) -> pd.DataFrame:
    """Score every row of a dataframe; returns the input df with prediction
    columns appended. Used for the built-in portfolio and uploaded datasets."""
    result = df.copy()
    probabilities = pipeline.predict_proba(df)[:, 1]
    result["probability_of_default"] = probabilities
    result["risk_category"] = [get_risk_band(p).name for p in probabilities]

    if distribution_reference:
        confidences = []
        for pos, (_, row) in enumerate(df.iterrows()):
            ood = check_out_of_distribution(row.to_dict(), distribution_reference)
            confidences.append(estimate_confidence(ood, probabilities[pos]))
        result["confidence"] = confidences
    else:
        result["confidence"] = "Unknown"

    return result
