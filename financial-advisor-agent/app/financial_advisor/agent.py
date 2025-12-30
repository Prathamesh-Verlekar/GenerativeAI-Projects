from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from app.financial_advisor.config import settings
from app.financial_advisor.tools.mcp_alpha_vantage import build_alpha_toolset
from app.financial_advisor.tools.mcp_google_sheets import build_google_sheets_toolset


def _load_system_prompt() -> str:
    # If you package system.md differently, adjust path accordingly.
    # Keeping it simple: read from app/financial_advisor/prompts/system.md
    import pathlib
    p = pathlib.Path(__file__).resolve().parent / "prompts" / "system.md"
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        # Fallback minimal instruction
        return (
            "You are a cautious financial advisor assistant. Use tools for prices/history; "
            "do not guarantee returns; ask for missing context."
        )


def build_model() -> LiteLlm:
    return LiteLlm(model=settings.OPENAI_MODEL, temperature=0.2)


agent = LlmAgent(
    name="financial_advisor",
    model=build_model(),
    instruction=_load_system_prompt(),
    tools=[build_alpha_toolset(), build_google_sheets_toolset()],
)

