"""👤 Customer Risk Assessment page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analytics.customer import compare_to_portfolio, customer_profile_dict
from src.config import CANONICAL_FEATURES, DISCLAIMER_SHORT, NUMERIC_RAW_FEATURES
from src.explainability.explain import explain_single_prediction
from src.ui.state import get_active_metadata, get_active_pipeline, get_scored_active_dataset


def render() -> None:
    st.title("👤 Customer Risk Assessment")

    scored_df = get_scored_active_dataset()
    pipeline = get_active_pipeline()
    metadata = get_active_metadata()

    if scored_df is None or pipeline is None:
        st.info("No active model/dataset available. Visit **Upload Dataset** or check **Model Information**.")
        return

    id_col = "customer_id" if "customer_id" in scored_df.columns else None
    if id_col is None:
        st.warning("This dataset has no `customer_id` column, so individual lookup isn't available. "
                   "Use the Customer Explorer on the Portfolio Analytics page instead, or the row index below.")
        options = list(scored_df.index[:2000])
        selected = st.selectbox("Select a row (by index)", options)
        row = scored_df.loc[selected]
    else:
        customer_ids = scored_df[id_col].astype(str).tolist()
        selected_id = st.selectbox("Select a customer", customer_ids[:5000])
        row = scored_df[scored_df[id_col].astype(str) == selected_id].iloc[0]

    st.divider()

    proba = float(row["probability_of_default"])
    risk_category = row["risk_category"]
    confidence = row.get("confidence", "Unknown")

    c1, c2, c3 = st.columns(3)
    c1.metric("Probability of Default", f"{proba:.1%}")
    c2.metric("Risk Category", risk_category)
    c3.metric("Confidence", confidence)

    st.info(DISCLAIMER_SHORT)

    st.divider()
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Customer Profile")
        profile = customer_profile_dict(row)
        profile_display = {k: (f"{v:,.2f}" if isinstance(v, (int, float)) else str(v)) for k, v in profile.items()}
        st.table(pd.DataFrame(profile_display.items(), columns=["Field", "Value"]).set_index("Field"))

    with col_right:
        st.subheader("Model Explanation")
        raw_feature_cols = [f.name for f in CANONICAL_FEATURES if f.name in row.index]
        single_df = pd.DataFrame([row[raw_feature_cols].to_dict()])
        try:
            explanation = explain_single_prediction(single_df, pipeline)
            st.caption(f"Explanation method: `{explanation.method}`")

            st.markdown("**⬆️ Top Risk-Increasing Factors**")
            for c in explanation.top_risk_increasing:
                st.markdown(f"- {c.label}")
            if not explanation.top_risk_increasing:
                st.caption("No risk-increasing factors identified.")

            st.markdown("**⬇️ Top Risk-Reducing Factors**")
            for c in explanation.top_risk_reducing:
                st.markdown(f"- {c.label}")
            if not explanation.top_risk_reducing:
                st.caption("No risk-reducing factors identified.")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Explanation could not be generated: {exc}")

    st.divider()
    st.subheader("Out-of-Distribution Warnings")
    from src.modeling.ood import check_out_of_distribution

    dist_ref = metadata.get("distribution_reference")
    if dist_ref:
        ood_report = check_out_of_distribution(row.to_dict(), dist_ref)
        if ood_report.has_warnings:
            for w in ood_report.numeric_warnings + ood_report.categorical_warnings:
                st.warning(w)
        else:
            st.success("No out-of-distribution signals detected for this customer.")
    else:
        st.caption("No training-distribution reference available for this model.")

    st.divider()
    st.subheader("Portfolio Comparison")
    numeric_cols_present = [c for c in NUMERIC_RAW_FEATURES if c in scored_df.columns]
    comparison = compare_to_portfolio(row, scored_df, numeric_cols_present)
    if comparison:
        comp_rows = []
        for col, stats in comparison.items():
            comp_rows.append({
                "Field": col, "Customer Value": round(stats["customer_value"], 2),
                "Portfolio Mean": round(stats["portfolio_mean"], 2),
                "Percentile Rank": f"{stats['percentile_rank']:.0f}th",
            })
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No comparable numeric fields available.")
