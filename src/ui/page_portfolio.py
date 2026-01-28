"""📊 Portfolio Analytics page."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.analytics.portfolio import (
    compute_kpis,
    default_rate_by_group,
    numeric_distribution_summary,
    risk_distribution,
)
from src.config import RISK_BANDS
from src.ui.state import get_active_metadata, get_scored_active_dataset


def render() -> None:
    st.title("📊 Portfolio Analytics")

    scored_df = get_scored_active_dataset()
    if scored_df is None:
        st.info("No active model/dataset available yet. Visit **Upload Dataset** or check **Model Information**.")
        return

    metadata = get_active_metadata()
    target_col = st.session_state.get("active_dataset_target_col")
    has_target = st.session_state.get("active_dataset_has_target") and target_col in scored_df.columns

    kpis = compute_kpis(scored_df, metadata.get("threshold", 0.5), target_col if has_target else None)

    st.subheader("Portfolio KPIs")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{kpis.total_customers:,}")
    c2.metric("Avg. Predicted Default Probability", f"{kpis.avg_predicted_default_probability:.1%}")
    c3.metric("High Risk Customers", f"{kpis.high_risk_customers:,}")
    c4.metric("Very High Risk Customers", f"{kpis.very_high_risk_customers:,}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Avg. Loan Amount", f"₹{kpis.avg_loan_amount:,.0f}" if kpis.avg_loan_amount else "n/a")
    c6.metric("Total Loan Exposure", f"₹{kpis.total_loan_exposure:,.0f}" if kpis.total_loan_exposure else "n/a")
    c7.metric("Predicted Default Rate", f"{(kpis.predicted_default_rate or 0):.1%}")
    if kpis.observed_default_rate is not None:
        c8.metric("Observed Default Rate", f"{kpis.observed_default_rate:.1%}")
    else:
        c8.metric("Observed Default Rate", "n/a (no target column)")

    st.caption(
        "'Predicted' default rate uses the model's optimized classification threshold "
        f"({metadata.get('threshold', 'n/a')}); it is a modeling choice, not a lending rule."
    )

    st.divider()
    st.subheader("Visualizations")

    color_map = {b.name: b.color for b in RISK_BANDS}

    col1, col2 = st.columns(2)
    with col1:
        dist = risk_distribution(scored_df)
        fig = px.bar(dist, x="risk_category", y="count", color="risk_category",
                     color_discrete_map=color_map, title="Risk Category Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(scored_df, x="probability_of_default", nbins=30,
                            title="Predicted Default Probability Distribution")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        if "loan_amount" in scored_df.columns:
            fig = px.box(scored_df, x="risk_category", y="loan_amount", color="risk_category",
                         color_discrete_map=color_map, title="Loan Amount by Risk Category")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("`loan_amount` column not available in this dataset.")
    with col4:
        if "loan_purpose" in scored_df.columns:
            grouped = default_rate_by_group(scored_df, "loan_purpose")
            fig = px.bar(grouped, x="loan_purpose", y="avg_probability_of_default",
                         title="Avg. Predicted Default Probability by Loan Purpose")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("`loan_purpose` column not available in this dataset.")

    col5, col6 = st.columns(2)
    with col5:
        if "employment_type" in scored_df.columns:
            grouped = default_rate_by_group(scored_df, "employment_type")
            fig = px.bar(grouped, x="employment_type", y="avg_probability_of_default",
                         title="Avg. Predicted Default Probability by Employment Type")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("`employment_type` column not available in this dataset.")
    with col6:
        if "annual_income" in scored_df.columns:
            fig = px.histogram(scored_df, x="annual_income", nbins=30, title="Income Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("`annual_income` column not available in this dataset.")

    if "credit_utilization" in scored_df.columns:
        fig = px.histogram(scored_df, x="credit_utilization", nbins=30, title="Credit Utilization Distribution")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("🔍 Customer Explorer")
    _render_customer_explorer(scored_df)


def _render_customer_explorer(scored_df) -> None:
    id_col = "customer_id" if "customer_id" in scored_df.columns else None

    filter_cols = st.columns(4)
    search_id = filter_cols[0].text_input("Search Customer ID") if id_col else ""
    risk_filter = filter_cols[1].multiselect(
        "Filter Risk Category", options=[b.name for b in RISK_BANDS], default=[]
    )
    purpose_filter = (
        filter_cols[2].multiselect("Filter Loan Purpose", options=sorted(scored_df["loan_purpose"].dropna().unique()))
        if "loan_purpose" in scored_df.columns else []
    )
    sort_by = filter_cols[3].selectbox(
        "Sort by", options=["probability_of_default", "loan_amount"] if "loan_amount" in scored_df.columns
        else ["probability_of_default"]
    )

    filtered = scored_df.copy()
    if id_col and search_id:
        filtered = filtered[filtered[id_col].astype(str).str.contains(search_id, case=False, na=False)]
    if risk_filter:
        filtered = filtered[filtered["risk_category"].isin(risk_filter)]
    if purpose_filter:
        filtered = filtered[filtered["loan_purpose"].isin(purpose_filter)]
    filtered = filtered.sort_values(sort_by, ascending=False)

    display_cols = [c for c in [id_col, "probability_of_default", "risk_category", "confidence",
                                 "loan_amount", "loan_purpose", "employment_type"] if c and c in filtered.columns]
    st.dataframe(filtered[display_cols].head(500), use_container_width=True, height=350)
    st.caption(f"Showing up to 500 of {len(filtered):,} matching customers.")
