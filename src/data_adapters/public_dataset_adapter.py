"""Adapter for the application's built-in demonstration dataset.

IMPORTANT / HONESTY NOTE
------------------------
This project ships a SYNTHETICALLY GENERATED dataset (see
`scripts/generate_sample_data.py`) rather than a redistributed third-party
dataset file, for two reasons:

1. This environment cannot reliably fetch or redistribute a licensed public
   dataset file (e.g. Lending Club, UCI, Kaggle exports) as part of the
   deliverable.
2. The application's UI and financial interpretation are India-focused, and
   the project's own instructions explicitly prohibit fabricating a
   dataset while claiming it represents real Indian borrowers.

The generator therefore builds a dataset with realistic, causally-sensible
relationships (income, DTI, credit utilization, delinquency history, etc.
driving default risk) so that the ML pipeline, SHAP explanations, and
analytics dashboards behave like a real portfolio would. Its provenance is
tracked explicitly as `data_source_type = "synthetic"` everywhere in the
app (model metadata, UI banners, GenAI copilot answers) — it is NEVER
described as real or as Indian bureau data.
"""

from __future__ import annotations

import pandas as pd

from src.config import DEMO_DATA_DISCLAIMER, SAMPLE_DATA_PATH
from src.data_adapters.base import BaseDatasetAdapter


class PublicDatasetAdapter(BaseDatasetAdapter):
    """Loads the bundled demonstration dataset. Already in canonical schema
    because it is generated directly against `src/config.py`."""

    name = "builtin_demo_dataset"
    data_source_type = "synthetic"
    disclaimer = DEMO_DATA_DISCLAIMER

    def can_handle(self, df: pd.DataFrame) -> bool:
        return False  # this adapter loads from a fixed path; not used for arbitrary uploads

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df  # already canonical

    def load(self) -> pd.DataFrame:
        return pd.read_csv(SAMPLE_DATA_PATH)
