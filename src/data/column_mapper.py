"""Automatic (but user-controlled) column mapping.

Uploaded datasets may use arbitrary column names (e.g. `AnnualIncome`,
`annual_inc`, `ApplicantIncome` for the canonical `annual_income` field).
This module SUGGESTS mappings using exact / case-insensitive / alias /
fuzzy matching, but never silently applies them — the caller (Streamlit UI)
must always show the suggestion to the user for confirmation or correction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from rapidfuzz import fuzz

from src.config import ALL_FEATURES, TARGET_ALIASES, TARGET_COLUMN, FeatureField


@dataclass
class ColumnMappingSuggestion:
    source_column: str
    suggested_target: Optional[str]  # canonical field name, or None
    confidence: float  # 0.0 - 1.0
    method: str  # "exact" | "case_insensitive" | "alias" | "fuzzy" | "none"


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _build_alias_index() -> dict[str, str]:
    """Map every normalized alias/name/canonical-name to its canonical field name."""
    index: dict[str, str] = {}
    for feature in ALL_FEATURES:
        if feature.is_engineered:
            continue
        index[_normalize(feature.name)] = feature.name
        for alias in feature.aliases:
            index[_normalize(alias)] = feature.name
    for alias in TARGET_ALIASES:
        index[_normalize(alias)] = TARGET_COLUMN
    return index


_ALIAS_INDEX = _build_alias_index()
_CANONICAL_NORMALIZED = {_normalize(f.name): f.name for f in ALL_FEATURES if not f.is_engineered}
_CANONICAL_NORMALIZED[_normalize(TARGET_COLUMN)] = TARGET_COLUMN


def suggest_mapping_for_column(column_name: str) -> ColumnMappingSuggestion:
    """Suggest the best canonical field for a single uploaded column name."""
    normalized = _normalize(column_name)

    # 1. Exact match against canonical name
    if normalized in _CANONICAL_NORMALIZED:
        return ColumnMappingSuggestion(column_name, _CANONICAL_NORMALIZED[normalized], 1.0, "exact")

    # 2. Alias dictionary match (covers case-insensitive too, since normalized)
    if normalized in _ALIAS_INDEX:
        return ColumnMappingSuggestion(column_name, _ALIAS_INDEX[normalized], 0.95, "alias")

    # 3. Fuzzy matching against all known names + aliases
    best_target: Optional[str] = None
    best_score = 0.0
    for norm_alias, canonical in _ALIAS_INDEX.items():
        score = fuzz.ratio(normalized, norm_alias) / 100.0
        if score > best_score:
            best_score = score
            best_target = canonical

    if best_score >= 0.80:
        return ColumnMappingSuggestion(column_name, best_target, round(best_score, 2), "fuzzy")

    return ColumnMappingSuggestion(column_name, None, 0.0, "none")


def suggest_mapping_for_dataset(columns: list[str]) -> list[ColumnMappingSuggestion]:
    """Suggest mappings for every column in an uploaded dataset.

    If two source columns map to the same canonical target, only the
    higher-confidence one keeps the suggestion; the other is marked
    unmapped so the user must resolve the conflict manually.
    """
    suggestions = [suggest_mapping_for_column(c) for c in columns]

    best_for_target: dict[str, int] = {}  # canonical -> index of best suggestion so far
    for i, s in enumerate(suggestions):
        if s.suggested_target is None:
            continue
        target = s.suggested_target
        if target not in best_for_target:
            best_for_target[target] = i
        else:
            current_best = suggestions[best_for_target[target]]
            if s.confidence > current_best.confidence:
                # demote the previous best
                suggestions[best_for_target[target]] = ColumnMappingSuggestion(
                    current_best.source_column, None, 0.0, "none (conflict)"
                )
                best_for_target[target] = i
            else:
                suggestions[i] = ColumnMappingSuggestion(s.source_column, None, 0.0, "none (conflict)")

    return suggestions


def apply_mapping(df, mapping: dict[str, str]):
    """Apply a user-confirmed {source_column: canonical_name} mapping.

    Only columns present in `mapping` are renamed/kept; this returns a new
    dataframe containing exactly the mapped columns (renamed to canonical
    names). The caller is responsible for keeping any extra columns if
    desired.
    """
    import pandas as pd

    mapped_cols = {src: tgt for src, tgt in mapping.items() if tgt and src in df.columns}
    result = df[list(mapped_cols.keys())].rename(columns=mapped_cols)
    return result
