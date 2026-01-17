"""Base class for dataset adapters.

An adapter's job is to transform a raw external dataframe (arbitrary
column names/units) into the application's canonical schema defined in
`src/config.py`, WITHOUT inventing values. Adapters should be conservative:
if a canonical field cannot be derived, leave it absent rather than guess.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseDatasetAdapter(ABC):
    """Common interface for all dataset adapters."""

    name: str = "base"

    @abstractmethod
    def can_handle(self, df: pd.DataFrame) -> bool:
        """Return True if this adapter recognizes the dataframe's shape/columns."""

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a new dataframe with columns renamed/derived to canonical names.

        Implementations must NOT fabricate values for columns that cannot
        be derived from the source data; simply omit them.
        """

    def describe(self) -> str:
        return f"Adapter: {self.name}"
