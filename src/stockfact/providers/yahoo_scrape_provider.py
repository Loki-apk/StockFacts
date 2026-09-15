"""The last-resort fallback: scrapes finance.yahoo.com's quote page directly.

Unlike yfinance (which calls Yahoo's own undocumented JSON API), this just
fetches the actual quote-page HTML and pulls out the ``quoteSummary`` blob
sitting inside it — Yahoo's frontend hydrates itself from a
``<script data-sveltekit-fetched data-url="...quoteSummary...">`` tag that
happens to carry the exact same data the API returns. Every other page on
the site (key-statistics, financials, history) blocks plain HTTP requests
and redirects to an error page, so this can only get what's on that one
page — no price history, no income statement, no insider transactions.
Those methods just raise ``ProviderError`` so the fallback chain skips
past them.

It's off unless ``STOCKFACT_ENABLE_YAHOO_SCRAPE=1`` is set (see
``base.get_provider``). It's also the shakiest of the three providers — it
depends on how Yahoo happens to lay out their page instead of a real API
contract, and it's the one most likely to raise ToS concerns since it's
reading the human-facing site instead of hitting a data endpoint.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from stockfact.providers.base import ProviderError

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

_FUNDAMENTALS_KEYS = (
    "trailingPE forwardPE enterpriseToEbitda revenueGrowth earningsGrowth "
    "grossMargins profitMargins freeCashflow totalRevenue debtToEquity "
    "currentRatio returnOnEquity marketCap beta"
).split()


def _unwrap(value: Any) -> Any:
    """Yahoo wraps most numbers as {"raw": ..., "fmt": "..."}; recurse to unwrap."""
    if isinstance(value, dict):
        if "raw" in value and "fmt" in value:
            return value.get("raw")
        return {k: _unwrap(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unwrap(v) for v in value]
    return value


class YahooScrapeProvider:
    name = "yahoo_scrape"

    def __init__(self) -> None:
        self._cache: dict[str, dict[str, Any]] = {}

    def _result(self, ticker: str) -> dict[str, Any]:
        """Fetch+parse the quote page once per ticker per process."""
        if ticker in self._cache:
            return self._cache[ticker]

        import httpx
        from bs4 import BeautifulSoup

        url = f"https://finance.yahoo.com/quote/{ticker}/"
        try:
            resp = httpx.get(
                url, headers=_HEADERS, timeout=httpx.Timeout(15.0), follow_redirects=True
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError("fetch", f"{ticker}: {exc}") from exc

        soup = BeautifulSoup(resp.text, "html.parser")
        for script in soup.find_all("script", attrs={"data-sveltekit-fetched": True}):
            if "quoteSummary" not in (script.get("data-url") or ""):
                continue
            try:
                payload = json.loads(script.string)
                body = payload["body"]
                body = json.loads(body) if isinstance(body, str) else body
                result = body["quoteSummary"]["result"][0]
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                raise ProviderError(
                    "parse", f"malformed quoteSummary for {ticker!r}: {exc}"
                ) from exc
            result = _unwrap(result)
            self._cache[ticker] = result
            return result
        raise ProviderError(
            "parse", f"no quoteSummary block found for {ticker!r} (Yahoo page layout changed?)"
        )

    def resolve(self, ticker: str) -> dict[str, Any]:
        result = self._result(ticker)
        price = result.get("price", {})
        profile = result.get("summaryProfile", {})
        name = price.get("longName") or price.get("shortName")
        if not name:
            raise ProviderError("resolve", f"unknown ticker {ticker!r}")
        return {
            "company_name": name,
            "sector": profile.get("sector") or "unknown",
            "industry": profile.get("industry") or "unknown",
            "currency": price.get("currency") or "USD",
            "exchange": price.get("exchangeName") or price.get("exchange") or "unknown",
        }

    def fundamentals_raw(self, ticker: str) -> dict[str, Any]:
        result = self._result(ticker)
        # financialData wins on overlap (e.g. profitMargins appears in both
        # defaultKeyStatistics and financialData; financialData's is fresher).
        merged = {
            **result.get("summaryDetail", {}),
            **result.get("defaultKeyStatistics", {}),
            **result.get("financialData", {}),
        }
        raw = {k: merged.get(k) for k in _FUNDAMENTALS_KEYS}
        if all(v is None for v in raw.values()):
            raise ProviderError("fundamentals", f"no fundamentals for {ticker!r}")
        return raw

    def price_history(self, ticker: str, period: str = "5y") -> pd.DataFrame:
        raise ProviderError(
            "price",
            "price history isn't on the quote page; Yahoo bot-walls the chart/history pages",
        )

    def calendar(self, ticker: str) -> dict[str, Any]:
        events = self._result(ticker).get("calendarEvents")
        if not events:
            raise ProviderError("calendar", f"no calendar events for {ticker!r}")
        return dict(events)

    def insider_transactions(self, ticker: str) -> list[dict[str, Any]]:
        raise ProviderError("insider", "insider transactions aren't embedded on the quote page")

    def recommendations(self, ticker: str) -> list[dict[str, Any]]:
        return list(self._result(ticker).get("recommendationTrend", {}).get("trend", []))

    def upgrades_downgrades(self, ticker: str) -> list[dict[str, Any]]:
        return list(self._result(ticker).get("upgradeDowngradeHistory", {}).get("history", []))

    def income_stmt(self, ticker: str, *, quarterly: bool = False) -> dict[str, Any]:
        raise ProviderError("income_stmt", "income statement isn't embedded on the quote page")

    def short_interest(self, ticker: str) -> dict[str, Any]:
        stats = self._result(ticker).get("defaultKeyStatistics", {})
        return {
            "short_ratio": stats.get("shortRatio"),
            "short_percent_of_float": stats.get("shortPercentOfFloat"),
            "shares_short": stats.get("sharesShort"),
        }
