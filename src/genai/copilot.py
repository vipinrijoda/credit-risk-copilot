"""GenAI Credit Risk Copilot orchestrator.

Architecture (per project spec, section 24):

    User Question
        -> Intent Detection / Tool Selection   (this module, heuristic)
        -> Python Analytics Function           (src/genai/tools.py)
        -> Structured Facts (JSON)
        -> LLM Explanation                     (Groq)
        -> Response

The LLM never calculates statistics; it only narrates facts it is given.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

from src.genai.prompts import SYSTEM_PROMPT, build_user_message

load_dotenv()

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")


class CopilotError(Exception):
    """Raised for any user-facing Copilot failure (missing key, API error)."""


@dataclass
class CopilotResponse:
    answer: str
    facts_used: dict
    intent: str


def detect_intent(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ("this customer", "why is", "compare this", "customer id", "risk factors for")):
        return "customer"
    if any(k in q for k in ("data quality", "missing values", "duplicate", "columns", "dataset issues", "pii")):
        return "data_quality"
    if any(k in q for k in ("metric", "roc", "auc", "accuracy", "precision", "recall", "f1", "model performance")):
        return "model_metrics"
    if any(k in q for k in ("loan purpose", "employment type", "by group", "which purpose", "which segment")):
        return "risk_by_group"
    return "portfolio"


def _get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise CopilotError(
            "GROQ_API_KEY is not set. Add it to a .env file (see .env.example) to enable "
            "the AI Risk Copilot. The rest of the application works without it."
        )
    try:
        from groq import Groq
    except ImportError as exc:
        raise CopilotError("The 'groq' package is not installed. Run: pip install groq") from exc

    try:
        return Groq(api_key=api_key)
    except Exception as exc:  # noqa: BLE001
        raise CopilotError(f"Could not initialize Groq client: {exc}") from exc


def ask_copilot(question: str, facts: dict, intent_override: Optional[str] = None) -> CopilotResponse:
    """Send a grounded question to the LLM and return its narration.

    `facts` must already be the OUTPUT of one or more functions in
    src/genai/tools.py — never raw dataset rows.
    """
    intent = intent_override or detect_intent(question)
    client = _get_client()

    user_message = build_user_message(question, facts)

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
            max_tokens=700,
        )
        answer = completion.choices[0].message.content
    except Exception as exc:  # noqa: BLE001
        raise CopilotError(f"The Groq API request failed: {exc}") from exc

    return CopilotResponse(answer=answer, facts_used=facts, intent=intent)
