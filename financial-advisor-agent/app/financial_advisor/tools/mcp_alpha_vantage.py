from __future__ import annotations

from app.financial_advisor.config import settings

from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams


def build_alpha_toolset() -> McpToolset:
    return McpToolset(
        connection_params=SseConnectionParams(url=settings.MCP_ALPHA_URL),
        # Optional but recommended: allow only these tools
        tool_filter=["search_ticker", "get_quote", "get_history"],
    )


