from __future__ import annotations

import os
import time
from typing import Any, Dict, List

from mcp.server.fastmcp import FastMCP
from sheets_client import GoogleSheetsClient
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route, Mount
import uvicorn


SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SHEETS_SERVICE_ACCOUNT_JSON", "").strip()
if not SERVICE_ACCOUNT_JSON:
    raise RuntimeError("GOOGLE_SHEETS_SERVICE_ACCOUNT_JSON is not set")

DEFAULT_SPREADSHEET_ID = os.getenv("GOOGLE_SHEETS_SPREADSHEET_ID", "").strip() or None

client = GoogleSheetsClient(service_account_json=SERVICE_ACCOUNT_JSON)
mcp = FastMCP(name="google_sheets_manager")

def healthz(request):
    return JSONResponse({"ok": True, "service": "google-sheets-mcp"})

def root(request):
    return PlainTextResponse("Google Sheets MCP is running. Use /sse for MCP SSE endpoint.\n")

app = Starlette(
    routes=[
        Route("/", root),
        Route("/healthz", healthz),
        Mount("/", app=mcp.sse_app()),
    ]
)


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def _resolve_spreadsheet_id(spreadsheet_id: str | None) -> str:
    candidate = (spreadsheet_id or DEFAULT_SPREADSHEET_ID or "").strip()
    if not candidate:
        raise ValueError("A spreadsheet_id is required (pass one or set GOOGLE_SHEETS_SPREADSHEET_ID)")
    return candidate


@mcp.tool()
def fetch_values(range_a1: str, spreadsheet_id: str | None = None, worksheet_title: str | None = None) -> Dict[str, Any]:
    """Fetch values from a worksheet range (A1 notation)."""
    t0 = time.perf_counter()
    range_name = (range_a1 or "").strip()
    if not range_name:
        raise ValueError("range_a1 is required")

    sheet_id = _resolve_spreadsheet_id(spreadsheet_id)
    worksheet_title_resolved, values = client.fetch_range(sheet_id, range_name, worksheet_title)
    print(
        f"[MCP] fetch_values range={range_name} sheet={sheet_id} worksheet={worksheet_title_resolved} latency_ms={_ms(t0):.1f}",
        flush=True,
    )
    return {
        "spreadsheet_id": sheet_id,
        "worksheet": worksheet_title_resolved,
        "range": range_name,
        "values": values,
    }


@mcp.tool()
def list_rows(spreadsheet_id: str | None = None, worksheet_title: str | None = None, limit: int = 20) -> Dict[str, Any]:
    """Return the first N rows from a worksheet (defaults to 20)."""
    t0 = time.perf_counter()
    limit_int = int(limit)
    if limit_int < 1:
        raise ValueError("limit must be >= 1")

    sheet_id = _resolve_spreadsheet_id(spreadsheet_id)
    worksheet_title_resolved, values = client.head_rows(sheet_id, worksheet_title, limit_int)
    print(
        f"[MCP] list_rows limit={limit_int} sheet={sheet_id} worksheet={worksheet_title_resolved} latency_ms={_ms(t0):.1f}",
        flush=True,
    )
    return {
        "spreadsheet_id": sheet_id,
        "worksheet": worksheet_title_resolved,
        "limit": limit_int,
        "rows": values,
    }


@mcp.tool()
def append_row(row: List[str], spreadsheet_id: str | None = None, worksheet_title: str | None = None) -> Dict[str, Any]:
    """Append a single row of values (uses USER_ENTERED semantics)."""
    t0 = time.perf_counter()
    if not row:
        raise ValueError("row must contain at least one cell")

    sheet_id = _resolve_spreadsheet_id(spreadsheet_id)
    worksheet_title_resolved, updated_range = client.append_row(sheet_id, row, worksheet_title)
    print(
        f"[MCP] append_row cells={len(row)} sheet={sheet_id} worksheet={worksheet_title_resolved} latency_ms={_ms(t0):.1f}",
        flush=True,
    )
    return {
        "spreadsheet_id": sheet_id,
        "worksheet": worksheet_title_resolved,
        "updated_range": updated_range,
        "row_values": row,
    }


@mcp.tool()
def update_range(
    range_a1: str,
    values: List[List[str]],
    spreadsheet_id: str | None = None,
    worksheet_title: str | None = None,
) -> Dict[str, Any]:
    """Overwrite a range of cells with provided 2D values."""
    t0 = time.perf_counter()
    range_name = (range_a1 or "").strip()
    if not range_name:
        raise ValueError("range_a1 is required")
    if not values or not all(isinstance(row, list) and row for row in values):
        raise ValueError("values must be a non-empty 2D list of rows")

    sheet_id = _resolve_spreadsheet_id(spreadsheet_id)
    worksheet_title_resolved, updated_range = client.update_range(sheet_id, range_name, values, worksheet_title)
    print(
        f"[MCP] update_range rows={len(values)} sheet={sheet_id} worksheet={worksheet_title_resolved} latency_ms={_ms(t0):.1f}",
        flush=True,
    )
    return {
        "spreadsheet_id": sheet_id,
        "worksheet": worksheet_title_resolved,
        "range": updated_range or range_name,
        "values": values,
    }


app = Starlette(
    routes=[
        Mount("/", app=mcp.sse_app()),
    ]
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8790, log_level="info")
