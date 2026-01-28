"""⚙️ Model Information page."""

from __future__ import annotations

import streamlit as st

from src.config import DISCLAIMER_FULL, SENSITIVE_ATTRIBUTE_KEYWORDS
from src.ui.state import (
    get_active_metadata,
    reset_to_builtin_dataset,
    reset_to_builtin_model,
)


def render() -> None:
    st.title("⚙️ Model Information")

    metadata = get_active_metadata()
    model_source = st.session_state.get("active_model_source", "builtin")

    if model_source == "builtin":
        st.success("**Active Model:** ✓ Built-in Demonstration Model")
    else:
        st.info("**Active Model:** ✓ User-Trained Model (this session only)")

    st.divider()
    st.subheader("Model Versioning / Metadata")
    st.json(metadata or {"note": "No model metadata available."})

    st.divider()
    st.subheader("Data Provenance")
    source_type = (metadata or {}).get("data_source_type", "unknown")
    labels = {
        "synthetic": "🧪 SYNTHETIC — generated for demonstration purposes, not real customer data.",
        "user_uploaded": "📤 USER-UPLOADED — trained on a dataset you uploaded this session.",
        "public_dataset": "🌐 PUBLIC DATASET — sourced from a public dataset.",
    }
    st.markdown(labels.get(source_type, f"Unknown source type: `{source_type}`"))

    st.divider()
    st.subheader("Responsible AI")
    st.markdown(
        f"""
- The model may contain **bias present in its training data**, whether synthetic or user-uploaded.
- Sensitive attributes are **never used as default model features**: {', '.join(SENSITIVE_ATTRIBUTE_KEYWORDS)}.
- If an uploaded dataset contains such columns, the app warns and lets you explicitly exclude them
  (see the Upload Dataset and Data Quality pages) — they are never silently included.
- The out-of-distribution and confidence indicators reflect **model reliability relative to
  training data**, not certainty about any individual's future behavior.
        """
    )

    st.divider()
    st.subheader("Regulatory & Legal Disclaimer")
    st.warning(DISCLAIMER_FULL)

    st.divider()
    st.subheader("Session Controls")
    col1, col2 = st.columns(2)
    if col1.button("🔄 Reset to Built-in Model"):
        reset_to_builtin_model()
        st.success("Reset to the built-in demonstration model.")
        st.rerun()
    if col2.button("🔄 Reset to Built-in Dataset"):
        reset_to_builtin_dataset()
        st.success("Reset to the built-in demonstration dataset.")
        st.rerun()
