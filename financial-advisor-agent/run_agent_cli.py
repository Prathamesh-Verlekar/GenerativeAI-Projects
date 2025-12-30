import asyncio
import re
import uuid
from typing import Optional

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.financial_advisor.agent import agent
from app.financial_advisor.guardrails.input_validation import normalize_user_text, looks_like_prompt_injection
from app.financial_advisor.guardrails.output_validation import validate_financial_output
from app.financial_advisor.tools.sheets_context import fetch_sheet_context, needs_sheet_context

APP_NAME = "financial-advisor-agent"
USER_ID = "local-user"


def extract_ticker_from_text(text: str) -> Optional[str]:
    """Pull ticker from phrases like 'ticker: CRWV' or 'ticker code ... is CRWV'."""
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
    """Rewrite pronoun follow-ups using remembered ticker."""
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


async def run_once(runner: Runner, session_id: str, user_text: str) -> str:
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
        user_id=USER_ID,
        session_id=session_id,
        new_message=content,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_text = event.content.parts[0].text or final_text

    return validate_financial_output(final_text or "No response returned.")


async def main():
    session_service = InMemorySessionService()
    session_id = str(uuid.uuid4())
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)

    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)

    last_ticker: Optional[str] = None
    print("Financial Advisor Agent is running. Type 'exit' to quit.")

    while True:
        user = input("\nYou: ").strip()
        if user.lower() in {"exit", "quit"}:
            break

        user2 = rewrite_followup(user, last_ticker)
        answer = await run_once(runner, session_id, user2)

        extracted = extract_ticker_from_text(answer)
        if extracted:
            last_ticker = extracted

        if last_ticker:
            print(f"\n[MEMORY] last_ticker={last_ticker}")

        print("\nAgent:", answer)


if __name__ == "__main__":
    asyncio.run(main())
