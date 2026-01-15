"""Indian Credit Risk Copilot — Streamlit application entry point.

Run with:
    python -m streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Indian Credit Risk Copilot",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.ui.state import get_active_metadata, init_session_state  # noqa: E402

init_session_state()

PAGES = {
    "🏠 Home": "page_home",
    "📊 Portfolio Analytics": "page_portfolio",
    "👤 Customer Risk Assessment": "page_customer",
    "➕ Assess New Customer": "page_new_customer",
    "📤 Upload Dataset": "page_upload",
    "🤖 AI Risk Copilot": "page_copilot",
    "📈 Model Performance": "page_model_performance",
    "🔍 Data Quality": "page_data_quality",
    "⚙️ Model Information": "page_model_info",
}

with st.sidebar:
    st.markdown("## 💳 Credit Risk Copilot")
    selection = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")

    st.divider()
    metadata = get_active_metadata()
    st.caption("**Active dataset:**")
    st.caption(st.session_state.get("active_dataset_name", "n/a"))
    st.caption("**Active model:**")
    st.caption(f"{metadata.get('model_type', 'None')} ({metadata.get('data_source_type', 'n/a')})")

    st.divider()
    st.caption(
        "⚠️ Decision-support analytics only. Not a lending decision, not RBI-approved, "
        "and does not access CIBIL or other credit bureaus."
    )

module_name = PAGES[selection]
page_module = __import__(f"src.ui.{module_name}", fromlist=["render"])
page_module.render()
