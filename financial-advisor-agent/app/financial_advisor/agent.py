from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from app.financial_advisor.config import settings
from app.financial_advisor.tools.mcp_alpha_vantage import build_alpha_toolset

# If you already load system.md, keep that logic.
INSTRUCTION = """You are a cautious financial advisor assistant.
- Use tools for market data (quotes/history).
- Do not claim certainty or guarantee returns.
- Ask for missing context (time horizon, risk tolerance) before giving suggestions.
"""

def build_model() -> LiteLlm:
    # Example: settings.OPENAI_MODEL = "openai/gpt-4.1-mini"
    return LiteLlm(model=settings.OPENAI_MODEL, temperature=0.2)

agent = LlmAgent(
    name="financial_advisor",
    model=build_model(),
    instruction=INSTRUCTION,
    tools=[build_alpha_toolset()],
)
