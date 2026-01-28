"""🤖 AI Risk Copilot page.

The chat UI never sends raw data to the LLM. It routes the question to the
right grounded "tool" function (src/genai/tools.py), then sends only the
resulting structured facts to Groq (src/genai/copilot.py).
"""

from __future__ import annotations

import streamlit as st

from src.genai.copilot import CopilotError, ask_copilot, detect_intent
from src.genai.tools import (
    get_customer_summary,
    get_data_quality_report,
    get_model_metrics,
    get_portfolio_summary,
    get_risk_by_group,
    get_risk_distribution,
)
from src.ui.state import get_active_dataset, get_active_metadata, get_scored_active_dataset

EXAMPLE_QUESTIONS = [
    "Summarize the portfolio risk.",
    "How many customers are high risk?",
    "Which loan purposes have the highest predicted risk?",
    "What data quality issues do you see in this dataset?",
    "What are the current model's ROC-AUC and F1 score?",
]


def _gather_facts_for_intent(intent: str, question: str) -> dict:
    scored_df = get_scored_active_dataset()
    metadata = get_active_metadata()
    raw_df = get_active_dataset()
    target_col = st.session_state.get("active_dataset_target_col")

    facts: dict = {"active_model_type": metadata.get("model_type"),
                   "active_data_source_type": metadata.get("data_source_type")}

    if scored_df is None:
        facts["note"] = "No active scored dataset is available in this session."
        return facts

    if intent == "customer":
        # Try to find a customer id mentioned in the question, else use the first customer.
        id_col = "customer_id" if "customer_id" in scored_df.columns else None
        candidate_id = None
        if id_col:
            for token in question.replace(",", " ").split():
                if token.upper().startswith("CUST") or token in scored_df[id_col].astype(str).values:
                    candidate_id = token
                    break
            if candidate_id is None:
                candidate_id = str(scored_df[id_col].iloc[0])
        facts["customer"] = get_customer_summary(scored_df, candidate_id, id_col) if id_col else \
            {"error": "No customer_id column available in this dataset."}
        facts["portfolio_context"] = get_portfolio_summary(scored_df, metadata.get("threshold", 0.5),
                                                             target_col if st.session_state.get("active_dataset_has_target") else None)
    elif intent == "data_quality":
        facts["data_quality"] = get_data_quality_report(raw_df) if raw_df is not None else {}
    elif intent == "model_metrics":
        facts["model_metrics"] = get_model_metrics(metadata)
    elif intent == "risk_by_group":
        for col in ("loan_purpose", "employment_type"):
            if col in scored_df.columns:
                facts[f"risk_by_{col}"] = get_risk_by_group(scored_df, col)
    else:  # portfolio (default)
        facts["portfolio_summary"] = get_portfolio_summary(
            scored_df, metadata.get("threshold", 0.5),
            target_col if st.session_state.get("active_dataset_has_target") else None,
        )
        facts["risk_distribution"] = get_risk_distribution(scored_df)

    return facts


def render() -> None:
    st.title("🤖 AI Risk Copilot")
    st.caption(
        "Ask about the active dataset, portfolio, or a specific customer. The Copilot only "
        "narrates numbers that this application has already calculated — it does not invent "
        "data, metrics, or explanations."
    )

    import os

    if not os.environ.get("GROQ_API_KEY"):
        st.warning(
            "`GROQ_API_KEY` is not set. Add it to a `.env` file (see `.env.example`) to enable "
            "the AI Copilot. The rest of the application works without it."
        )

    with st.expander("💡 Example questions"):
        for q in EXAMPLE_QUESTIONS:
            st.markdown(f"- {q}")

    for msg in st.session_state.get("chat_history", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask the Risk Copilot a question...")
    if not question:
        return

    st.session_state.setdefault("chat_history", []).append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    intent = detect_intent(question)
    facts = _gather_facts_for_intent(intent, question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Analyzing..."):
                response = ask_copilot(question, facts, intent_override=intent)
            st.markdown(response.answer)
            with st.expander("🔎 Facts used to ground this answer"):
                st.json(response.facts_used)
            st.session_state["chat_history"].append({"role": "assistant", "content": response.answer})
        except CopilotError as exc:
            st.error(str(exc))
