import asyncio
import uuid
import re
from typing import Optional, List, Dict

import streamlit as st

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.financial_advisor.agent import agent
from app.financial_advisor.guardrails.input_validation import normalize_user_text, looks_like_prompt_injection
from app.financial_advisor.guardrails.output_validation import validate_financial_output
from app.financial_advisor.tools.sheets_context import fetch_sheet_context, needs_sheet_context


def extract_ticker_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"\bticker(?:\s+code)?\s*[:is-]+\s*([A-Z]{1,7})\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m2 = re.search(r"\$([A-Z]{1,7})\b", text)
    if m2:
        return m2.group(1).upper()
    return None


def user_mentions_ticker(user_text: str) -> bool:
    return bool(re.search(r"\$?[A-Z]{1,7}\b", user_text))


def rewrite_followup(user_text: str, last_ticker: Optional[str]) -> str:
    if not last_ticker:
        return user_text

    t = (user_text or "").strip().lower()

    if any(p in t for p in ["its price", "what is its price", "price?", "current price", "how much is it"]) and not user_mentions_ticker(user_text):
        return f"Get the latest quote for {last_ticker} and answer with price, change %, and as_of timestamp."

    if "yesterday" in t and not user_mentions_ticker(user_text):
        return (
            f"For {last_ticker}: use get_history(symbol='{last_ticker}', days=5) and answer:\n"
            f"- yesterday (most recent prior trading day) close\n"
            f"- change vs previous trading day\n"
            f"- brief note if markets were closed."
        )

    return user_text


async def run_agent_once(runner: Runner, session_id: str, user_text: str) -> str:
    user_text = normalize_user_text(user_text)
    if looks_like_prompt_injection(user_text):
        return (
            "I can’t follow requests that try to override system rules or reveal secrets. "
            "Tell me your investing question (symbol, goal, time horizon), and I’ll help."
        )

    # ✅ Deterministic spreadsheet grounding (when relevant)
    if needs_sheet_context(user_text):
        sheet_ctx = await fetch_sheet_context()
        user_text = f"{sheet_ctx}\nUser question:\n{user_text}"

    content = types.Content(role="user", parts=[types.Part(text=user_text)])

    final_text = ""
    async for event in runner.run_async(
        user_id=st.session_state.user_id,
        session_id=session_id,
        new_message=content,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_text = event.content.parts[0].text or final_text

    return validate_financial_output(final_text or "No response returned.")


def ensure_runtime():
    if "user_id" not in st.session_state:
        st.session_state.user_id = "streamlit-user"

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "last_ticker" not in st.session_state:
        st.session_state.last_ticker = None

    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, str]] = []

    if "session_service" not in st.session_state:
        st.session_state.session_service = InMemorySessionService()

    if "runner" not in st.session_state:
        asyncio.run(
            st.session_state.session_service.create_session(
                app_name="financial-advisor-agent",
                user_id=st.session_state.user_id,
                session_id=st.session_state.session_id,
            )
        )
        st.session_state.runner = Runner(
            agent=agent,
            app_name="financial-advisor-agent",
            session_service=st.session_state.session_service,
        )


st.set_page_config(page_title="Financial Advisor Agent", page_icon="📈", layout="wide")
ensure_runtime()

st.title("📈 Financial Advisor Agent (ADK + MCP + Redis)")
st.caption("Spreadsheet grounding via Google Sheets MCP + market data via Alpha Vantage MCP. Not financial advice.")

with st.sidebar:
    st.header("Session")
    st.write("Session ID:", st.session_state.session_id)
    st.write("Last ticker:", st.session_state.last_ticker or "—")
    st.divider()
    st.subheader("Tips")
    st.markdown("- Try: **Summarize my portfolio from the spreadsheet**")
    st.markdown("- Try: **From my watchlist in sheet, pick top 3 tickers and justify**")
    st.markdown("- Then: **What is its price?**")
    st.divider()
    if st.button("Reset chat"):
        st.session_state.messages = []
        st.session_state.last_ticker = None
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask about a stock, portfolio, watchlist, budget…")

if prompt:
    rewritten = rewrite_followup(prompt, st.session_state.last_ticker)

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer = asyncio.run(run_agent_once(st.session_state.runner, st.session_state.session_id, rewritten))
        st.markdown(answer)

    new_ticker = extract_ticker_from_text(answer)
    if new_ticker:
        st.session_state.last_ticker = new_ticker

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
