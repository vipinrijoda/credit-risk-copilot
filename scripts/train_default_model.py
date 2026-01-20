"""Train and save the application's built-in demonstration model.

Run:
    python scripts/train_default_model.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.config import (
    DEFAULT_MODEL_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    SAMPLE_DATA_PATH,
)
from src.modeling.train import save_model, train_model


def main() -> None:
    df = pd.read_csv(SAMPLE_DATA_PATH)
    df = df.drop(columns=["customer_id"])  # identifier, not a model feature

    result = train_model(
        df,
        target_col="default",
        model_name="Random Forest",
        threshold_strategy="f1",
        data_source_type="synthetic",
    )

    print("Training warnings:", result.warnings or "None")
    print("Metrics:", result.metadata["metrics"])
    print("Threshold:", result.metadata["threshold"])

    save_model(result, DEFAULT_MODEL_PATH, DEFAULT_MODEL_METADATA_PATH)
    print(f"Saved model to {DEFAULT_MODEL_PATH}")
    print(f"Saved metadata to {DEFAULT_MODEL_METADATA_PATH}")


if __name__ == "__main__":
    main()
