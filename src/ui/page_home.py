"""🏠 Home page."""

from __future__ import annotations

import streamlit as st

from src.config import DEMO_DATA_DISCLAIMER, DISCLAIMER_FULL, DISCLAIMER_SHORT, PRIVACY_NOTICE
from src.ui.state import get_active_dataset, get_active_metadata


def render() -> None:
    st.title("🏠 Indian Credit Risk Copilot")
    st.caption("ML + Explainable AI + GenAI Decision Support System")

    st.warning(DISCLAIMER_SHORT, icon="⚠️")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("What this application does")
        st.markdown(
            """
- Estimates **probability of default** for individual applicants
- Explains predictions with **SHAP / feature importance**
- Lets you **upload your own dataset** (CSV/Excel) and train or score against it
- Assesses **new, hypothetical customers** via a dynamic form
- Provides **portfolio-level risk analytics**
- Includes an **AI Risk Copilot** grounded in real, calculated numbers
            """
        )
    with col2:
        st.subheader("What this application does NOT do")
        st.markdown(
            """
- ❌ It does **not** automatically approve or reject loans
- ❌ It does **not** access CIBIL, Experian India, Equifax India, or CRIF High Mark
- ❌ It does **not** collect PAN, Aadhaar, bank account, or card numbers
- ❌ It does **not** claim regulatory (RBI) approval or compliance
- ❌ It does **not** let the AI Copilot invent metrics or SHAP values
            """
        )

    st.divider()
    st.subheader("Current session status")

    metadata = get_active_metadata()
    dataset = get_active_dataset()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Active dataset rows", len(dataset) if dataset is not None else 0)
    m2.metric("Active model", metadata.get("model_type", "None trained yet"))
    m3.metric("Data source type", metadata.get("data_source_type", "n/a"))
    m4.metric("Decision threshold", metadata.get("threshold", "n/a"))

    st.divider()
    with st.expander("📋 Full regulatory & privacy disclaimer", expanded=False):
        st.markdown(f"**Regulatory notice:** {DISCLAIMER_FULL}")
        st.markdown(f"**Privacy notice:** {PRIVACY_NOTICE}")
        st.markdown(f"**Demonstration dataset notice:** {DEMO_DATA_DISCLAIMER}")

    st.divider()
    st.subheader("Navigate")
    st.markdown(
        """
Use the sidebar to explore:
1. **📊 Portfolio Analytics** — KPIs and risk breakdowns across the active dataset
2. **👤 Customer Risk Assessment** — look up an existing customer's risk profile
3. **➕ Assess New Customer** — score a hypothetical applicant via a form
4. **📤 Upload Dataset** — bring your own CSV/Excel data
5. **🤖 AI Risk Copilot** — ask questions grounded in calculated facts
6. **📈 Model Performance** — ROC-AUC, confusion matrix, curves
7. **🔍 Data Quality** — missingness, duplicates, leakage, PII checks
8. **⚙️ Model Information** — active model metadata & versioning
        """
    )
