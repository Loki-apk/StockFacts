"""The fallback provider: Financial Modeling Prep, on their ``/stable`` API.

Only turns on if ``FMP_API_KEY`` is set. It kicks in when yfinance fails, and
it's always used for peer resolution (the `stock-peers` endpoint) once a key
is set. Anything that fails raises ``ProviderError`` so ``FallbackProvider``
knows to move on.

Only the endpoints below are on FMP's free tier — ``insider_transactions``
and ``short_interest`` need a paid plan, so they just come back empty here.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd

from stockfact.providers.base import ProviderError

_BASE = "https://financialmodelingprep.com/stable"


class FMPProvider:
    name = "fmp"

    def __init__(self) -> None:
        self.api_key = os.environ.get("FMP_API_KEY")
        if not self.api_key:
            raise ProviderError("config", "FMP_API_KEY not set")

    # --- transport ---
    def _get(self, path: str, datatype: str, **params: Any) -> Any:
        import httpx

        params["apikey"] = self.api_key
        try:
            r = httpx.get(f"{_BASE}/{path}", params=params, timeout=httpx.Timeout(20.0))
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(datatype, f"{path}: {exc}") from exc
        if r.status_code in (401, 402, 403):
            raise ProviderError(datatype, f"{path}: {r.status_code} (plan/endpoint)")
        r.raise_for_status()
        try:
            return r.json()
        except ValueError as exc:
            raise ProviderError(datatype, f"{path}: non-JSON response") from exc

    def _first(self, path: str, datatype: str, **params: Any) -> dict[str, Any]:
        data = self._get(path, datatype, **params)
        if not isinstance(data, list) or not data:
            raise ProviderError(datatype, f"{path}: empty response")
        return data[0]

    # --- MarketDataProvider surface ---
    def resolve(self, ticker: str) -> dict[str, Any]:
        p = self._first("profile", "resolve", symbol=ticker)
        if not p.get("companyName"):
            raise ProviderError("resolve", f"unknown ticker {ticker!r}")
        return {
            "company_name": p["companyName"],
            "sector": p.get("sector") or "unknown",
            "industry": p.get("industry") or "unknown",
            "currency": p.get("currency") or "USD",
            "exchange": p.get("exchangeFullName") or p.get("exchange") or "unknown",
        }

    def fundamentals_raw(self, ticker: str) -> dict[str, Any]:
        ratios = self._first("ratios-ttm", "fundamentals", symbol=ticker)
        km = self._first("key-metrics-ttm", "fundamentals", symbol=ticker)
        profile = self._first("profile", "fundamentals", symbol=ticker)
        rev_g, eps_g = self._growth(ticker)
        return {
            "trailingPE": ratios.get("priceToEarningsRatioTTM"),
            "forwardPE": None,  # not on the free tier
            "enterpriseToEbitda": km.get("evToEBITDATTM"),
            "revenueGrowth": rev_g,
            "earningsGrowth": eps_g,
            "grossMargins": ratios.get("grossProfitMarginTTM"),
            "profitMargins": ratios.get("netProfitMarginTTM"),
            # FMP gives a true ratio; the app expects yfinance's percent convention
            "debtToEquity": _to_pct(ratios.get("debtToEquityRatioTTM")),
            "currentRatio": ratios.get("currentRatioTTM"),
            "returnOnEquity": ratios.get("returnOnEquityTTM"),
            "freeCashflow": None,
            "totalRevenue": None,
            "marketCap": profile.get("marketCap"),
            "beta": profile.get("beta"),
        }

    def _growth(self, ticker: str) -> tuple[float | None, float | None]:
        rows = self._get("income-statement", "fundamentals", symbol=ticker, limit=2)
        if not isinstance(rows, list) or len(rows) < 2:
            return None, None
        cur, prev = rows[0], rows[1]

        def grow(a, b):
            return (a / b - 1) if a and b else None

        return grow(cur.get("revenue"), prev.get("revenue")), grow(
            cur.get("epsDiluted") or cur.get("eps"),
            prev.get("epsDiluted") or prev.get("eps"),
        )

    def price_history(self, ticker: str, period: str = "5y") -> pd.DataFrame:
        rows = self._get("historical-price-eod/light", "price", symbol=ticker)
        if not isinstance(rows, list) or not rows:
            raise ProviderError("price", f"no history for {ticker!r}")
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df.rename(columns={"price": "Close", "volume": "Volume"})
        return df[["Close", "Volume"]]

    def calendar(self, ticker: str) -> dict[str, Any]:
        rows = self._get("earnings", "calendar", symbol=ticker, limit=8)
        from datetime import date

        today = date.today().isoformat()
        upcoming = sorted(
            r["date"] for r in (rows or []) if r.get("date") and r["date"] >= today
        )
        return {"Earnings Date": [upcoming[0]]} if upcoming else {}

    def insider_transactions(self, ticker: str) -> list[dict[str, Any]]:  # noqa: ARG002
        return []  # not on the free tier

    def recommendations(self, ticker: str) -> list[dict[str, Any]]:
        return self._get("grades-historical", "recommendations", symbol=ticker, limit=12) or []

    def upgrades_downgrades(self, ticker: str) -> list[dict[str, Any]]:
        rows = self._get("grades", "upgrades_downgrades", symbol=ticker, limit=40) or []
        out = []
        for r in rows:
            action = (r.get("action") or "").lower()
            out.append(
                {
                    "GradeDate": r.get("date"),
                    "Firm": r.get("gradingCompany"),
                    "Action": {"upgrade": "up", "downgrade": "down"}.get(action, action),
                    "ToGrade": r.get("newGrade"),
                    "FromGrade": r.get("previousGrade"),
                }
            )
        return out

    def income_stmt(self, ticker: str, *, quarterly: bool = False) -> dict[str, Any]:
        params = {"symbol": ticker, "limit": 6}
        if quarterly:
            params["period"] = "quarter"
        rows = self._get("income-statement", "income_stmt", **params)
        if not isinstance(rows, list) or not rows:
            raise ProviderError("income_stmt", f"no income statement for {ticker!r}")
        out: dict[str, Any] = {}
        for r in rows:
            out[str(r["date"])[:10]] = {
                "EBIT": r.get("ebit") or r.get("operatingIncome"),
                "Operating Income": r.get("operatingIncome"),
                "Interest Expense": r.get("interestExpense"),
                "Net Income": r.get("netIncome"),
                "Diluted EPS": r.get("epsDiluted") or r.get("eps"),
            }
        return out

    def short_interest(self, ticker: str) -> dict[str, Any]:  # noqa: ARG002
        return {}

    # --- peer resolution (used by fetch/peers.py) ---
    def peers(self, ticker: str, industry: str | None = None) -> list[str]:  # noqa: ARG002
        rows = self._get("stock-peers", "peers", symbol=ticker)
        if isinstance(rows, list):
            return [r["symbol"] for r in rows if r.get("symbol")]
        return []


def _to_pct(ratio: float | None) -> float | None:
    return None if ratio is None else ratio * 100.0
