from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class ToolDecision:
    allow: bool
    reason: str = ""

ALLOWED_MCP_TOOLS = {"get_quote", "get_history", "search_ticker"}

def validate_tool_call(tool_name: str, args: Dict[str, Any]) -> ToolDecision:
    if tool_name not in ALLOWED_MCP_TOOLS:
        return ToolDecision(False, f"Tool not allowed: {tool_name}")

    if tool_name in {"get_quote", "get_history"}:
        symbol = (args.get("symbol") or "").upper().strip()
        if not symbol or len(symbol) > 12:
            return ToolDecision(False, "Invalid symbol")

    if tool_name == "get_history":
        days = int(args.get("days", 30))
        if days > 3650:
            return ToolDecision(False, "History window too large (max 10 years)")
        if days < 1:
            return ToolDecision(False, "days must be >= 1")

    if tool_name == "search_ticker":
        q = (args.get("query") or "").strip()
        if not q:
            return ToolDecision(False, "query cannot be empty")
        if len(q) > 60:
            return ToolDecision(False, "query too long")

    return ToolDecision(True)
