"""PII and sensitive-attribute detection for uploaded datasets.

This module never sees or stores real PAN/Aadhaar/bank data on purpose; it
only inspects COLUMN NAMES (and lightweight value patterns) to warn the
user, who must then decide whether to exclude those columns before
training. The application does not collect these fields anywhere else.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from src.config import PII_KEYWORDS, SENSITIVE_ATTRIBUTE_KEYWORDS

_PAN_PATTERN = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")
_AADHAAR_PATTERN = re.compile(r"^\d{4}\s?\d{4}\s?\d{4}$")


@dataclass
class PIIFlag:
    column: str
    kind: str  # "pii" | "sensitive_attribute"
    reason: str


def _matches_any_keyword(col_name: str, keywords: tuple[str, ...]) -> str | None:
    normalized = col_name.lower().replace(" ", "_")
    for kw in keywords:
        if kw in normalized:
            return kw
    return None


def detect_pii_and_sensitive_columns(df: pd.DataFrame) -> list[PIIFlag]:
    flags: list[PIIFlag] = []

    for col in df.columns:
        pii_kw = _matches_any_keyword(col, PII_KEYWORDS)
        if pii_kw:
            flags.append(PIIFlag(col, "pii", f"Column name suggests direct personal identifier ('{pii_kw}')."))
            continue  # a column is flagged once

        sensitive_kw = _matches_any_keyword(col, SENSITIVE_ATTRIBUTE_KEYWORDS)
        if sensitive_kw:
            flags.append(
                PIIFlag(
                    col, "sensitive_attribute",
                    f"Column name suggests a sensitive attribute ('{sensitive_kw}') that should not "
                    "be used as a default model feature.",
                )
            )
            continue

        # Lightweight value-pattern sniffing on a sample, only for object columns.
        if df[col].dtype == object:
            sample = df[col].dropna().astype(str).head(50)
            if len(sample) > 0:
                pan_ratio = sample.str.match(_PAN_PATTERN).mean()
                aadhaar_ratio = sample.str.match(_AADHAAR_PATTERN).mean()
                if pan_ratio > 0.5:
                    flags.append(PIIFlag(col, "pii", "Column values match the PAN number format."))
                elif aadhaar_ratio > 0.5:
                    flags.append(PIIFlag(col, "pii", "Column values match the Aadhaar number format."))

    return flags


def recommended_exclusions(flags: list[PIIFlag]) -> list[str]:
    """Columns recommended to exclude before training."""
    return [f.column for f in flags]
