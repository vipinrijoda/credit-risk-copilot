"""➕ Assess New Customer page.

The form is generated DYNAMICALLY from `src/config.py`'s CANONICAL_FEATURES
— the same schema used for training, mapping, and prediction — so there is
never a separate, hand-maintained list of fields to keep in sync.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import CANONICAL_FEATURES, DISCLAIMER_SHORT, FieldType
from src.explainability.explain import explain_single_prediction
from src.modeling.ood import check_out_of_distribution
from src.modeling.predict import predict_single
from src.ui.state import get_active_metadata, get_active_pipeline
from src.utils.validators import validate_customer_input


def _render_dynamic_form() -> dict:
    raw_input: dict = {}
    with st.form("new_customer_form"):
        st.markdown("Fields marked **\\*** are required by the active model.")
        cols = st.columns(2)
        for i, feature in enumerate(CANONICAL_FEATURES):
            target_col = cols[i % 2]
            label = feature.label + (" \\*" if feature.required else "")
            if feature.field_type == FieldType.NUMERIC:
                default = float(feature.default_value) if feature.default_value is not None else 0.0
                raw_input[feature.name] = target_col.number_input(
                    label, value=default, help=feature.help_text, key=f"nc_{feature.name}",
                )
            else:
                options = list(feature.categories) if feature.categories else [""]
                default_index = options.index(feature.default_value) if feature.default_value in options else 0
                raw_input[feature.name] = target_col.selectbox(
                    label, options=options, index=default_index, key=f"nc_{feature.name}",
                )
        submitted = st.form_submit_button("Assess Customer", type="primary")
    return raw_input if submitted else None


def render() -> None:
    st.title("➕ Assess New Customer")
    st.caption("Assess a hypothetical applicant who does not need to exist in any dataset.")

    pipeline = get_active_pipeline()
    metadata = get_active_metadata()
    if pipeline is None:
        st.info("No active model available. Visit **Model Information** or train a model on **Upload Dataset**.")
        return

    raw_input = _render_dynamic_form()
    if raw_input is None:
        return

    validation = validate_customer_input(raw_input)

    for w in validation.warnings:
        st.warning(w)
    if not validation.is_valid:
        for e in validation.errors:
            st.error(e)
        st.stop()

    threshold = metadata.get("threshold", 0.5)
    dist_ref = metadata.get("distribution_reference")

    result = predict_single(validation.cleaned, pipeline, threshold, dist_ref)

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Probability of Default", f"{result.probability_of_default:.1%}")
    c2.metric("Risk Category", result.risk_category)
    c3.metric("Confidence", result.confidence)
    st.info(DISCLAIMER_SHORT)

    if result.ood_report.has_warnings:
        st.subheader("Out-of-Distribution Warnings")
        for w in result.ood_report.numeric_warnings + result.ood_report.categorical_warnings:
            st.warning(w)
        st.caption(
            "Confidence has been lowered because one or more inputs fall outside the "
            "model's training-data range. This affects model reliability, not certainty "
            "about whether this customer will default."
        )

    st.divider()
    st.subheader("Model Explanation")
    single_df = pd.DataFrame([validation.cleaned])
    try:
        explanation = explain_single_prediction(single_df, pipeline)
        st.caption(f"Explanation method: `{explanation.method}`")
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("**⬆️ Top Risk-Increasing Factors**")
            for c in explanation.top_risk_increasing:
                st.markdown(f"- {c.label}")
        with col_right:
            st.markdown("**⬇️ Top Risk-Reducing Factors**")
            for c in explanation.top_risk_reducing:
                st.markdown(f"- {c.label}")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Explanation could not be generated: {exc}")
