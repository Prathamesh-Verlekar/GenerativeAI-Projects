from __future__ import annotations

import os
import re
from typing import Any, List, Optional, Tuple

from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

from app.financial_advisor.config import settings

# Accept:
# - A1:Z200
# - Sheet1!A1:Z200
# - 'My Sheet'!A1:Z200
_TAB_RANGE_RE = re.compile(
    r"""^(?:
        (?:'([^']+)'|([A-Za-z0-9 _-]+))!  # optional worksheet title
    )?
    ([A-Za-z]{1,3}\d{1,7}(?::[A-Za-z]{1,3}\d{1,7})?)$  # A1 or A1:B2
    """,
    re.VERBOSE,
)


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 200] + "\n…(truncated)…\n"


def _parse_tabbed_range(tabbed: str) -> Tuple[Optional[str], str]:
    """
    Converts:
      Sheet1!A1:Z200 -> ("Sheet1", "A1:Z200")
      A1:Z200        -> (None, "A1:Z200")
      'My Tab'!A1:C5 -> ("My Tab", "A1:C5")
    """
    s = (tabbed or "").strip()
    m = _TAB_RANGE_RE.match(s)
    if not m:
        # fallback: treat as plain range
        return None, s
    quoted_tab, plain_tab, a1 = m.group(1), m.group(2), m.group(3)
    return (quoted_tab or plain_tab), a1


def _extract_2d_values(tool_result: Any) -> List[List[str]]:
    """
    Works with the MCP python client response object which includes:
      tool_result.structuredContent = {'result': {'rows': [...]} }
      tool_result.structuredContent = {'result': {'values': [...]} }
    and also handles plain dict shapes defensively.
    """
    if tool_result is None:
        return []

    # 1) Newer MCP client: response object with structuredContent attr
    structured = getattr(tool_result, "structuredContent", None)
    if isinstance(structured, dict):
        res = structured.get("result", {})
        if isinstance(res, dict):
            if isinstance(res.get("rows"), list):
                return res["rows"]
            if isinstance(res.get("values"), list):
                return res["values"]

    # 2) Sometimes tool_result itself is dict-like
    if isinstance(tool_result, dict):
        sc = tool_result.get("structuredContent")
        if isinstance(sc, dict):
            res = sc.get("result", {})
            if isinstance(res, dict):
                if isinstance(res.get("rows"), list):
                    return res["rows"]
                if isinstance(res.get("values"), list):
                    return res["values"]

        res = tool_result.get("result")
        if isinstance(res, dict):
            if isinstance(res.get("rows"), list):
                return res["rows"]
            if isinstance(res.get("values"), list):
                return res["values"]

    return []


def _format_as_context(values: List[List[str]], title: str) -> str:
    if not values:
        return f"### {title}\n(No rows returned from Sheets MCP)\n"

    lines = [f"### {title}"]
    for r in values[: settings.SHEETS_CONTEXT_MAX_ROWS]:
        lines.append(" | ".join(str(x) for x in r))
    return "\n".join(lines) + "\n"


async def fetch_sheet_context(
    *,
    tabbed_range: Optional[str] = None,
    list_rows_limit: Optional[int] = None,
    worksheet_title: Optional[str] = None,
    spreadsheet_id: Optional[str] = None,
    mcp_url: Optional[str] = None,
) -> str:
    """
    Deterministically pulls sheet data via Sheets MCP and returns a compact context block.
    """
    url = mcp_url or os.getenv("MCP_SHEETS_URL", settings.MCP_SHEETS_URL)

    # Defaults: prefer list_rows (it clearly returns "rows")
    if tabbed_range is None and list_rows_limit is None:
        list_rows_limit = min(settings.SHEETS_CONTEXT_MAX_ROWS, 60)

    # Allow config-driven worksheet title
    ws = worksheet_title or os.getenv("GOOGLE_SHEETS_WORKSHEET_TITLE", None)

    # If user provided Sheet1!A1:Z200, split it into worksheet_title + range_a1
    range_a1 = None
    if tabbed_range:
        maybe_ws, a1 = _parse_tabbed_range(tabbed_range)
        range_a1 = a1
        ws = ws or maybe_ws

    async with sse_client(url) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            if range_a1 is not None:
                args = {"range_a1": range_a1}
                if ws:
                    args["worksheet_title"] = ws
                if spreadsheet_id:
                    args["spreadsheet_id"] = spreadsheet_id

                res = await session.call_tool("fetch_values", args)
                values = _extract_2d_values(res)
                ctx = _format_as_context(values, title=f"Google Sheets Context ({ws or 'default'}!{range_a1})")
                return _clip(ctx, settings.SHEETS_CONTEXT_MAX_CHARS)

            # list_rows path
            limit = max(1, min(int(list_rows_limit or 20), 200))
            args = {"limit": limit}
            if ws:
                args["worksheet_title"] = ws
            if spreadsheet_id:
                args["spreadsheet_id"] = spreadsheet_id

            res = await session.call_tool("list_rows", args)
            values = _extract_2d_values(res)
            ctx = _format_as_context(values, title=f"Google Sheets Context ({ws or 'default'} top {limit} rows)")
            return _clip(ctx, settings.SHEETS_CONTEXT_MAX_CHARS)


def needs_sheet_context(user_text: str) -> bool:
    t = (user_text or "").lower()
    keywords = [
        "sheet", "spreadsheet",
        "portfolio", "holdings", "positions", "allocation",
        "watchlist", "instrument", "qty", "shares",
        "transactions", "trades",
        "expenses", "budget", "income", "cashflow", "p&l",
    ]
    return any(k in t for k in keywords)
