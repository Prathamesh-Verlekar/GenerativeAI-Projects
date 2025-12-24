from __future__ import annotations

import os
import time
from typing import Dict, List

from alpha_client import AlphaVantageClient
from cache import Cache

from mcp.server.fastmcp import FastMCP

from starlette.applications import Starlette
from starlette.routing import Mount
import uvicorn


# ----------------------------
# Environment / configuration
# ----------------------------
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()
if not API_KEY:
    raise RuntimeError("ALPHAVANTAGE_API_KEY is not set")

REDIS_URL = os.getenv("REDIS_URL", "").strip() or None
QUOTE_TTL = int(os.getenv("QUOTE_CACHE_TTL_SEC", "20"))
MAX_HISTORY_DAYS = int(os.getenv("MAX_HISTORY_DAYS", "3650"))

client = AlphaVantageClient(api_key=API_KEY)
cache = Cache(redis_url=REDIS_URL)

mcp = FastMCP(name="alpha_vantage_market_data")


# ----------------------------
# Helpers
# ----------------------------
def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


# ----------------------------
# MCP Tools with observability
# ----------------------------
@mcp.tool()
def search_ticker(query: str) -> List[Dict[str, str]]:
    """Search tickers by keyword (Alpha Vantage SYMBOL_SEARCH)."""
    t0 = time.perf_counter()
    q = (query or "").strip()
    if not q:
        print(f"[MCP] search_ticker query='' cache=NA latency_ms={_ms(t0):.1f}", flush=True)
        return []

    # Optional cache (helpful if you do a lot of searches)
    cache_key = f"search:{q.lower()}"
    cached = cache.get_json(cache_key)
    if cached is not None:
        print(f"[MCP] search_ticker query='{q}' cache=HIT latency_ms={_ms(t0):.1f}", flush=True)
        return cached

    data = client.search(q)
    cache.set_json(cache_key, data, ttl_sec=max(60, QUOTE_TTL))
    print(f"[MCP] search_ticker query='{q}' cache=MISS latency_ms={_ms(t0):.1f}", flush=True)
    return data


@mcp.tool()
def get_quote(symbol: str) -> Dict:
    """Get latest quote snapshot for a symbol (Alpha Vantage GLOBAL_QUOTE)."""
    t0 = time.perf_counter()
    sym = (symbol or "").upper().strip()
    if not sym or len(sym) > 12:
        print(f"[MCP] get_quote symbol='{sym}' cache=NA latency_ms={_ms(t0):.1f} ERROR=invalid_symbol", flush=True)
        raise ValueError("Invalid symbol")

    cache_key = f"quote:{sym}"
    cached = cache.get_json(cache_key)
    if cached:
        cached["cache"] = {"hit": True, "ttl_sec": QUOTE_TTL}
        print(f"[MCP] get_quote symbol={sym} cache=HIT latency_ms={_ms(t0):.1f}", flush=True)
        return cached

    data = client.quote(sym)
    cache.set_json(cache_key, data, ttl_sec=QUOTE_TTL)
    data["cache"] = {"hit": False, "ttl_sec": QUOTE_TTL}
    print(f"[MCP] get_quote symbol={sym} cache=MISS latency_ms={_ms(t0):.1f}", flush=True)
    return data


@mcp.tool()
def get_history(symbol: str, days: int = 30) -> Dict:
    """Get daily adjusted OHLC history (Alpha Vantage TIME_SERIES_DAILY_ADJUSTED)."""
    t0 = time.perf_counter()
    sym = (symbol or "").upper().strip()
    if not sym or len(sym) > 12:
        print(
            f"[MCP] get_history symbol='{sym}' days={days} cache=NA latency_ms={_ms(t0):.1f} ERROR=invalid_symbol",
            flush=True,
        )
        raise ValueError("Invalid symbol")

    days_i = int(days)
    if days_i < 1:
        print(
            f"[MCP] get_history symbol={sym} days={days_i} cache=NA latency_ms={_ms(t0):.1f} ERROR=invalid_days",
            flush=True,
        )
        raise ValueError("days must be >= 1")

    if days_i > MAX_HISTORY_DAYS:
        print(
            f"[MCP] get_history symbol={sym} days={days_i} cache=NA latency_ms={_ms(t0):.1f} ERROR=days_too_large",
            flush=True,
        )
        raise ValueError(f"days too large; max is {MAX_HISTORY_DAYS}")

    # Daily bars don't change frequently; cache longer than quotes
    hist_ttl = max(60, QUOTE_TTL * 3)
    cache_key = f"hist:{sym}:{days_i}"
    cached = cache.get_json(cache_key)
    if cached:
        cached["cache"] = {"hit": True, "ttl_sec": hist_ttl}
        print(f"[MCP] get_history symbol={sym} days={days_i} cache=HIT latency_ms={_ms(t0):.1f}", flush=True)
        return cached

    data = client.history_daily_adjusted(sym, days=days_i)
    cache.set_json(cache_key, data, ttl_sec=hist_ttl)
    data["cache"] = {"hit": False, "ttl_sec": hist_ttl}
    print(f"[MCP] get_history symbol={sym} days={days_i} cache=MISS latency_ms={_ms(t0):.1f}", flush=True)
    return data


# ----------------------------
# SSE App
# ----------------------------
# IMPORTANT: mount at "/" so FastMCP can expose /sse and its internal endpoints correctly.
app = Starlette(
    routes=[
        Mount("/", app=mcp.sse_app()),
    ]
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8787, log_level="info")
