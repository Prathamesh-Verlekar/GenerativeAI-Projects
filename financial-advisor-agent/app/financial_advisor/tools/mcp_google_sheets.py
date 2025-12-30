from __future__ import annotations

from app.financial_advisor.config import settings

from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams


def build_google_sheets_toolset() -> McpToolset:
    return McpToolset(
        connection_params=SseConnectionParams(url=settings.MCP_SHEETS_URL),
        tool_filter=["fetch_values", "list_rows", "append_row", "update_range"],
    )
