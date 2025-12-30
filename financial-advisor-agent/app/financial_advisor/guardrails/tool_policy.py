from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import re


@dataclass
class ToolDecision:
    allow: bool
    reason: str = ""


# Allowed tools (Alpha + Sheets)
ALLOWED_MCP_TOOLS = {
    # Alpha Vantage MCP
    "search_ticker",
    "get_quote",
    "get_history",
    # Google Sheets MCP
    "fetch_values",
    "list_rows",
    "append_row",
    "update_range",
}

_A1_RE = re.compile(r"^(?:'[^']+'|[A-Za-z0-9 _-]+)!\s*[A-Za-z]{1,3}\d{1,7}(?::[A-Za-z]{1,3}\d{1,7})?$"r"|^[A-Za-z]{1,3}\d{1,7}(?::[A-Za-z]{1,3}\d{1,7})?$")



def validate_tool_call(tool_name: str, args: Dict[str, Any]) -> ToolDecision:
    if tool_name not in ALLOWED_MCP_TOOLS:
        return ToolDecision(False, f"Tool not allowed: {tool_name}")

    # -----------------------
    # Alpha Vantage validation
    # -----------------------
    if tool_name in {"get_quote", "get_history"}:
        symbol = (args.get("symbol") or "").upper().strip()
        if not symbol or len(symbol) > 12:
            return ToolDecision(False, "Invalid symbol")

    if tool_name == "get_history":
        try:
            days = int(args.get("days", 30))
        except Exception:
            return ToolDecision(False, "days must be an integer")

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

    # -----------------------
    # Google Sheets validation
    # -----------------------
    if tool_name == "fetch_values":
        range_a1 = (args.get("range_a1") or "").strip()
        if not range_a1 or not _A1_RE.match(range_a1):
            return ToolDecision(False, "Invalid A1 range")

    if tool_name == "list_rows":
        try:
            limit = int(args.get("limit", 50))
        except Exception:
            return ToolDecision(False, "limit must be an integer")
        if limit < 1 or limit > 200:
            return ToolDecision(False, "limit must be between 1 and 200")

    if tool_name == "append_row":
        row = args.get("row")
        if not isinstance(row, list):
            return ToolDecision(False, "row must be a list")
        if len(row) > 50:
            return ToolDecision(False, "row too wide (max 50 columns)")
        # Keep cell size bounded
        for cell in row:
            if cell is not None and len(str(cell)) > 2000:
                return ToolDecision(False, "cell too large")

    if tool_name == "update_range":
        range_a1 = (args.get("range_a1") or "").strip()
        if not range_a1 or not _A1_RE.match(range_a1):
            return ToolDecision(False, "Invalid A1 range")
        values = args.get("values")
        if not isinstance(values, list):
            return ToolDecision(False, "values must be a 2D list")
        if len(values) > 200:
            return ToolDecision(False, "too many rows in update (max 200)")
        for r in values:
            if not isinstance(r, list) or len(r) > 50:
                return ToolDecision(False, "each row must be a list with <= 50 columns")

    return ToolDecision(True)
