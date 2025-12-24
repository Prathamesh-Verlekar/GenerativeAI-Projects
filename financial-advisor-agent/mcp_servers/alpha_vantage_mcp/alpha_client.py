from __future__ import annotations

import requests
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class AlphaVantageClient:
    """Thin client for Alpha Vantage REST API."""

    def __init__(self, api_key: str, session: Optional[requests.Session] = None) -> None:
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.session = session or requests.Session()

    def _get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params = dict(params)
        params["apikey"] = self.api_key
        resp = self.session.get(self.base_url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if "Error Message" in data:
            raise RuntimeError(f"Alpha Vantage error: {data['Error Message']}")
        if "Note" in data:
            raise RuntimeError(f"Alpha Vantage throttled: {data['Note']}")

        return data

    def search(self, keywords: str) -> List[Dict[str, Any]]:
        data = self._get({"function": "SYMBOL_SEARCH", "keywords": keywords})
        matches = data.get("bestMatches", []) or []
        results: List[Dict[str, Any]] = []

        for m in matches[:10]:
            results.append(
                {
                    "symbol": m.get("1. symbol"),
                    "name": m.get("2. name"),
                    "type": m.get("3. type"),
                    "region": m.get("4. region"),
                    "marketOpen": m.get("5. marketOpen"),
                    "marketClose": m.get("6. marketClose"),
                    "timezone": m.get("7. timezone"),
                    "currency": m.get("8. currency"),
                    "matchScore": m.get("9. matchScore"),
                    "source": "alpha_vantage",
                }
            )
        return results

    def quote(self, symbol: str) -> Dict[str, Any]:
        data = self._get({"function": "GLOBAL_QUOTE", "symbol": symbol})
        q = data.get("Global Quote", {}) or {}
        if not q:
            raise RuntimeError(f"No quote returned for {symbol}")

        price = float(q.get("05. price", "nan"))
        return {
            "symbol": q.get("01. symbol", symbol).upper(),
            "price": price,
            "open": _to_float(q.get("02. open")),
            "high": _to_float(q.get("03. high")),
            "low": _to_float(q.get("04. low")),
            "volume": _to_float(q.get("06. volume")),
            "latest_trading_day": q.get("07. latest trading day"),
            "previous_close": _to_float(q.get("08. previous close")),
            "change": _to_float(q.get("09. change")),
            "change_percent": q.get("10. change percent"),
            "as_of": datetime.now(timezone.utc).isoformat(),
            "source": "alpha_vantage",
        }

    def history_daily_adjusted(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        outputsize = "compact" if days <= 100 else "full"
        data = self._get(
            {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": symbol,
                "outputsize": outputsize,
            }
        )

        meta = data.get("Meta Data", {}) or {}
        series = data.get("Time Series (Daily)", {}) or {}

        bars: List[Dict[str, Any]] = []
        for d in sorted(series.keys(), reverse=True)[:days]:
            row = series[d]
            bars.append(
                {
                    "date": d,
                    "open": _to_float(row.get("1. open")),
                    "high": _to_float(row.get("2. high")),
                    "low": _to_float(row.get("3. low")),
                    "close": _to_float(row.get("4. close")),
                    "adjusted_close": _to_float(row.get("5. adjusted close")),
                    "volume": _to_float(row.get("6. volume")),
                    "dividend_amount": _to_float(row.get("7. dividend amount")),
                    "split_coefficient": _to_float(row.get("8. split coefficient")),
                }
            )

        return {
            "symbol": symbol.upper(),
            "days": days,
            "last_refreshed": meta.get("3. Last Refreshed"),
            "timezone": meta.get("5. Time Zone"),
            "bars": bars,
            "as_of": datetime.now(timezone.utc).isoformat(),
            "source": "alpha_vantage",
        }


def _to_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        s = str(x).strip().replace("%", "")
        return float(s)
    except Exception:
        return float("nan")
