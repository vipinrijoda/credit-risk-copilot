"""Shared session-state helpers for the Streamlit app.

Centralizes how the "active model" and "active dataset" are stored so
every page reads/writes state the same way. This is what lets the app
cleanly switch between:
  - Built-in demo dataset + built-in demo model
  - Uploaded dataset (with target) + user-trained model
  - Uploaded dataset (no target) + built-in model used for scoring only
"""

from __future__ import annotations

import streamlit as st

from src.config import (
    DEFAULT_MODEL_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    SAMPLE_DATA_PATH,
)
from src.data.loader import load_sample_dataset
from src.modeling.train import load_metadata, load_model


def init_session_state() -> None:
    defaults = {
        "active_dataset_source": "builtin",     # "builtin" | "uploaded"
        "active_dataset_df": None,               # canonical-schema dataframe (raw, pre-scoring)
        "active_dataset_name": "Built-in demonstration dataset",
        "active_dataset_has_target": True,
        "active_dataset_target_col": "default",
        "active_model_source": "builtin",        # "builtin" | "user_trained"
        "active_pipeline": None,
        "active_model_metadata": None,
        "raw_upload_df": None,                   # dataframe exactly as uploaded, pre-mapping
        "column_mapping": None,                  # confirmed {source_col: canonical_name}
        "excluded_columns": [],
        "scored_df": None,                       # cached scored version of active dataset
        "chat_history": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if st.session_state["active_dataset_df"] is None:
        st.session_state["active_dataset_df"] = load_sample_dataset(SAMPLE_DATA_PATH)

    if st.session_state["active_pipeline"] is None:
        try:
            st.session_state["active_pipeline"] = load_model(DEFAULT_MODEL_PATH)
            st.session_state["active_model_metadata"] = load_metadata(DEFAULT_MODEL_METADATA_PATH)
        except FileNotFoundError:
            st.session_state["active_pipeline"] = None
            st.session_state["active_model_metadata"] = None


def get_active_pipeline():
    return st.session_state.get("active_pipeline")


def get_active_metadata() -> dict:
    return st.session_state.get("active_model_metadata") or {}


def get_active_dataset():
    return st.session_state.get("active_dataset_df")


def set_user_trained_model(pipeline, metadata: dict) -> None:
    st.session_state["active_pipeline"] = pipeline
    st.session_state["active_model_metadata"] = metadata
    st.session_state["active_model_source"] = "user_trained"
    st.session_state["scored_df"] = None  # invalidate cache


def reset_to_builtin_model() -> None:
    st.session_state["active_pipeline"] = load_model(DEFAULT_MODEL_PATH)
    st.session_state["active_model_metadata"] = load_metadata(DEFAULT_MODEL_METADATA_PATH)
    st.session_state["active_model_source"] = "builtin"
    st.session_state["scored_df"] = None


def get_scored_active_dataset():
    """Return the active dataset scored with the active model, using a
    cached copy in session state to avoid re-scoring on every rerun."""
    if st.session_state.get("scored_df") is not None:
        return st.session_state["scored_df"]

    pipeline = get_active_pipeline()
    df = get_active_dataset()
    metadata = get_active_metadata()
    if pipeline is None or df is None:
        return None

    from src.modeling.predict import predict_batch

    target_col = st.session_state.get("active_dataset_target_col")
    feature_df = df.drop(columns=[c for c in [target_col] if c and c in df.columns])
    threshold = metadata.get("threshold", 0.5)
    dist_ref = metadata.get("distribution_reference")

    scored = predict_batch(feature_df, pipeline, threshold, dist_ref)
    # re-attach target/id columns dropped above, plus any id columns from df
    for col in df.columns:
        if col not in scored.columns:
            scored[col] = df[col].values

    st.session_state["scored_df"] = scored
    return scored


def reset_to_builtin_dataset() -> None:
    st.session_state["active_dataset_df"] = load_sample_dataset(SAMPLE_DATA_PATH)
    st.session_state["active_dataset_source"] = "builtin"
    st.session_state["active_dataset_name"] = "Built-in demonstration dataset"
    st.session_state["active_dataset_has_target"] = True
    st.session_state["active_dataset_target_col"] = "default"
    st.session_state["scored_df"] = None
