"""Unit tests for the Yahoo-quote-page scraper. No live network access —
httpx.get is monkeypatched to return a minimal fixture page."""

from __future__ import annotations

import json

import httpx
import pytest

from stockfact.providers.base import ProviderError
from stockfact.providers.yahoo_scrape_provider import YahooScrapeProvider

_QUOTE_SUMMARY = {
    "price": {
        "longName": "Test Corp",
        "shortName": "Test",
        "currency": "USD",
        "exchangeName": "NasdaqGS",
        "regularMarketPrice": {"raw": 123.45, "fmt": "123.45"},
    },
    "summaryProfile": {
        "sector": "Technology",
        "industry": "Software",
    },
    "summaryDetail": {
        "trailingPE": {"raw": 20.1, "fmt": "20.10"},
        "marketCap": {"raw": 1000000, "fmt": "1M"},
        "beta": {"raw": 1.2, "fmt": "1.20"},
    },
    "defaultKeyStatistics": {
        "forwardPE": {"raw": 18.0, "fmt": "18.00"},
        "shortRatio": {"raw": 2.5, "fmt": "2.50"},
        "shortPercentOfFloat": {"raw": 0.01, "fmt": "1.00%"},
        "sharesShort": {"raw": 500000, "fmt": "500,000"},
    },
    "financialData": {
        "revenueGrowth": {"raw": 0.15, "fmt": "15.00%"},
        "profitMargins": {"raw": 0.2, "fmt": "20.00%"},
    },
    "calendarEvents": {
        "earnings": {"earningsDate": [{"raw": 1793304000, "fmt": "2026-10-29"}]},
    },
    "recommendationTrend": {"trend": [{"period": "0m", "strongBuy": 1}]},
    "upgradeDowngradeHistory": {"history": [{"firm": "Test Bank", "toGrade": "Buy"}]},
}


def _fixture_html(ticker: str, result: dict | None = None) -> str:
    summary = result if result is not None else _QUOTE_SUMMARY
    body = json.dumps({"quoteSummary": {"result": [summary]}})
    payload = json.dumps({"body": body})
    url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=x"
    return (
        "<html><body>"
        f'<script type="application/json" data-sveltekit-fetched data-url="{url}">'
        f"{payload}</script>"
        "</body></html>"
    )


@pytest.fixture
def provider(monkeypatch):
    def fake_get(url, **kwargs):
        ticker = url.rstrip("/").rsplit("/", 1)[-1]
        return httpx.Response(200, text=_fixture_html(ticker), request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    return YahooScrapeProvider()


def test_resolve(provider):
    out = provider.resolve("TEST")
    assert out == {
        "company_name": "Test Corp",
        "sector": "Technology",
        "industry": "Software",
        "currency": "USD",
        "exchange": "NasdaqGS",
    }


def test_fundamentals_raw(provider):
    out = provider.fundamentals_raw("TEST")
    assert out["trailingPE"] == 20.1
    assert out["forwardPE"] == 18.0
    assert out["marketCap"] == 1000000
    assert out["beta"] == 1.2
    assert out["revenueGrowth"] == 0.15
    assert out["profitMargins"] == 0.2


def test_short_interest(provider):
    out = provider.short_interest("TEST")
    assert out == {"short_ratio": 2.5, "short_percent_of_float": 0.01, "shares_short": 500000}


def test_calendar(provider):
    out = provider.calendar("TEST")
    assert out["earnings"]["earningsDate"][0] == 1793304000


def test_recommendations_and_upgrades(provider):
    assert provider.recommendations("TEST") == [{"period": "0m", "strongBuy": 1}]
    assert provider.upgrades_downgrades("TEST") == [{"firm": "Test Bank", "toGrade": "Buy"}]


def test_unsupported_methods_raise(provider):
    with pytest.raises(ProviderError):
        provider.price_history("TEST")
    with pytest.raises(ProviderError):
        provider.insider_transactions("TEST")
    with pytest.raises(ProviderError):
        provider.income_stmt("TEST")


def test_missing_quote_summary_block_raises(monkeypatch):
    def fake_get(url, **kwargs):
        return httpx.Response(
            200,
            text="<html><body>no data here</body></html>",
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    p = YahooScrapeProvider()
    with pytest.raises(ProviderError):
        p.resolve("TEST")


def test_result_is_cached_per_ticker(monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        ticker = url.rstrip("/").rsplit("/", 1)[-1]
        return httpx.Response(200, text=_fixture_html(ticker), request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    p = YahooScrapeProvider()
    p.resolve("TEST")
    p.fundamentals_raw("TEST")
    p.short_interest("TEST")
    assert len(calls) == 1
