from __future__ import annotations

from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams

from app.financial_advisor.config import settings


def build_google_sheets_toolset() -> McpToolset:
    """
    Exposes Google Sheets MCP tools to the agent (LLM can call these).
    Note: We also do deterministic context injection via sheets_context.py.
    """
    return McpToolset(
        connection_params=SseConnectionParams(url=settings.MCP_SHEETS_URL),
        tool_filter=["fetch_values", "list_rows", "append_row", "update_range"],
    )
