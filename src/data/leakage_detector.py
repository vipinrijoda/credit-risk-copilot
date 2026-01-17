"""Target-leakage detection.

Flags columns that are likely to contain information only available AFTER
loan issuance or after repayment behaviour is known (e.g. `recoveries`,
`final_payment`). These columns must never be silently used for training —
the user is shown a warning and must explicitly exclude or keep them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import LEAKAGE_KEYWORDS


@dataclass
class LeakageFlag:
    column: str
    reason: str
    severity: str  # "high" | "medium"


def detect_keyword_leakage(columns: list[str]) -> list[LeakageFlag]:
    flags: list[LeakageFlag] = []
    for col in columns:
        normalized = col.lower().replace(" ", "_")
        for keyword in LEAKAGE_KEYWORDS:
            if keyword in normalized:
                flags.append(
                    LeakageFlag(
                        column=col,
                        reason=f"Column name matches known post-outcome pattern ('{keyword}').",
                        severity="high",
                    )
                )
                break
    return flags


def detect_correlation_leakage(
    df: pd.DataFrame, target_col: str, threshold: float = 0.95
) -> list[LeakageFlag]:
    """Flag numeric columns near-perfectly correlated with the target.

    A correlation this strong with the outcome usually signals that the
    column encodes the outcome itself (or a close proxy of it) rather than
    a genuine predictive signal available at application time.
    """
    flags: list[LeakageFlag] = []
    if target_col not in df.columns:
        return flags

    y = pd.to_numeric(df[target_col], errors="coerce")
    if y.nunique(dropna=True) < 2:
        return flags

    numeric_df = df.select_dtypes(include=[np.number])
    for col in numeric_df.columns:
        if col == target_col:
            continue
        x = numeric_df[col]
        if x.nunique(dropna=True) < 2:
            continue
        try:
            corr = x.corr(y)
        except Exception:  # noqa: BLE001
            continue
        if corr is not None and abs(corr) >= threshold:
            flags.append(
                LeakageFlag(
                    column=col,
                    reason=f"Near-perfect correlation with target (|r|={abs(corr):.2f}).",
                    severity="high",
                )
            )
    return flags


def detect_leakage(df: pd.DataFrame, target_col: str | None = None) -> list[LeakageFlag]:
    """Run all leakage checks and return a de-duplicated list of flags."""
    flags = detect_keyword_leakage(list(df.columns))
    if target_col:
        flags += detect_correlation_leakage(df, target_col)

    seen = set()
    unique_flags = []
    for f in flags:
        if f.column not in seen:
            unique_flags.append(f)
            seen.add(f.column)
    return unique_flags
