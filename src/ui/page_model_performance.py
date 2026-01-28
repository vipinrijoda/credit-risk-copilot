"""📈 Model Performance page."""

from __future__ import annotations

import streamlit as st

from src.ui.state import get_active_metadata


def render() -> None:
    st.title("📈 Model Performance")

    metadata = get_active_metadata()
    if not metadata:
        st.info("No trained model is active yet.")
        return

    metrics = metadata.get("metrics") or {}

    if not metrics or all(v is None for v in metrics.values()):
        st.warning(
            "Validation metrics are unavailable for the currently active model/dataset "
            "combination (e.g. the active dataset has no observed outcome column, so it is "
            "used for scoring only, not evaluation). No fake metrics are shown."
        )
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("ROC-AUC", f"{metrics.get('roc_auc'):.3f}" if metrics.get("roc_auc") is not None else "n/a")
        c2.metric("Accuracy", f"{metrics.get('accuracy'):.3f}" if metrics.get("accuracy") is not None else "n/a")
        c3.metric("Precision", f"{metrics.get('precision'):.3f}" if metrics.get("precision") is not None else "n/a")
        c4.metric("Recall", f"{metrics.get('recall'):.3f}" if metrics.get("recall") is not None else "n/a")
        c5.metric("F1", f"{metrics.get('f1'):.3f}" if metrics.get("f1") is not None else "n/a")
        st.caption(
            "⚠️ Accuracy alone is not sufficient for imbalanced credit-risk data — always read "
            "it together with ROC-AUC, precision, and recall."
        )

    st.divider()
    st.subheader(f"Decision Threshold: {metadata.get('threshold', 'n/a')}")
    st.caption(
        f"Optimized for **{metadata.get('threshold_strategy', 'f1')}** on validation data. "
        "The threshold is a modeling decision and is NOT a real lending approval rule."
    )

    st.divider()
    st.subheader("Model Metadata")
    meta_display = {
        "Model type": metadata.get("model_type"),
        "Data source type": metadata.get("data_source_type"),
        "Is demonstration model": metadata.get("is_demo_model"),
        "Trained at (UTC)": metadata.get("trained_at"),
        "Training rows": metadata.get("training_rows"),
        "Validation rows": metadata.get("validation_rows"),
        "Numeric features used": metadata.get("features_numeric"),
        "Categorical features used": metadata.get("features_categorical"),
        "Excluded columns": metadata.get("excluded_columns"),
        "sklearn version": metadata.get("sklearn_version"),
    }
    st.json(meta_display)
