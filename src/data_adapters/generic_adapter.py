"""Generic adapter: transforms ANY user-uploaded dataset into the canonical
schema, using user-confirmed column mappings (see src/data/column_mapper.py).

This is the adapter used for the "Upload Dataset" flow in the Streamlit UI.
It performs light, well-defined unit derivations (e.g. monthly -> annual
income) but never invents values that aren't derivable from the data.
"""

from __future__ import annotations

import pandas as pd

from src.config import CANONICAL_FEATURES, TARGET_COLUMN
from src.data_adapters.base import BaseDatasetAdapter


class GenericAdapter(BaseDatasetAdapter):
    name = "generic_user_upload"

    def can_handle(self, df: pd.DataFrame) -> bool:
        return True  # generic adapter accepts anything; used as the fallback

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # In the app this is typically called AFTER apply_mapping() has
        # already renamed the user's columns to canonical names. This
        # method performs the remaining, well-defined derivations.
        result = df.copy()

        if "annual_income" not in result.columns and "monthly_income" in result.columns:
            result["annual_income"] = pd.to_numeric(result["monthly_income"], errors="coerce") * 12

        if "monthly_income" not in result.columns and "annual_income" in result.columns:
            result["monthly_income"] = pd.to_numeric(result["annual_income"], errors="coerce") / 12

        return result

    @staticmethod
    def transform_with_mapping(raw_df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
        """Convenience: apply a {source_col: canonical_name} mapping, then
        run the standard generic transform (unit derivations)."""
        from src.data.column_mapper import apply_mapping

        mapped = apply_mapping(raw_df, mapping)
        adapter = GenericAdapter()
        return adapter.transform(mapped)
