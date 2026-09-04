"""Related Exposure fetch layer — fully offline. EDGAR and price history are
monkeypatched: these tests exercise the deterministic wiring (candidate table,
verbatim-quote-or-skip, error containment, correlation gating), not live data.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from stocks.fetch import related_exposure as re_mod
from stocks.run_context import RunContext


def _ctx(ticker: str = "NVDA", company_name: str = "NVIDIA Corporation") -> RunContext:
    return RunContext(
        ticker=ticker,
        company_name=company_name,
        sector="Technology",
        industry="Semiconductors",
        currency="USD",
        exchange="NASDAQ",
        generated_at=date(2026, 9, 3),
    )


def test_search_alias_strips_legal_suffix():
    assert re_mod._search_alias("NVIDIA Corporation") == "NVIDIA"
    assert re_mod._search_alias("Apple Inc.") == "Apple"
    assert re_mod._search_alias("Some Weird Name") == "Some Weird Name"


def test_unlisted_ticker_returns_no_exposures():
    report = re_mod.fetch_related_exposure(_ctx(ticker="ZZZZ", company_name="Nowhere Corp"))
    assert report.exposures == []
    assert report.candidates_checked == 0


def test_confirmed_mention_becomes_an_exposure(monkeypatch):
    ctx = _ctx()

    def fake_mention(candidate_ticker: str, needle: str):
        assert needle == "NVIDIA"
        if candidate_ticker == "SMCI":
            return {
                "quote": "...a significant customer, including NVIDIA...",
                "url": "https://www.sec.gov/example",
                "filed_at": "2026-02-01",
            }
        return None  # no confirmed mention for the other candidates

    monkeypatch.setattr(re_mod.edgar, "find_relationship_mention", fake_mention)
    monkeypatch.setattr(re_mod, "_resolve_cached", lambda t: {"company_name": f"{t} Inc."})
    monkeypatch.setattr(re_mod, "_correlation", lambda a, b: 0.42)

    report = re_mod.fetch_related_exposure(ctx)

    assert report.candidates_checked == len(re_mod._STATIC_EXPOSURES["NVDA"])
    assert {e.related_ticker for e in report.exposures} == {"SMCI"}
    exp = report.exposures[0]
    assert exp.relationship_type == "customer"
    assert exp.related_company_name == "SMCI Inc."
    assert "NVIDIA" in exp.evidence_quote
    assert exp.evidence_source == "https://www.sec.gov/example"
    assert exp.price_correlation_2y == 0.42
    # never invented — an untouched candidate never becomes an exposure
    assert "DELL" not in {e.related_ticker for e in report.exposures}


def test_edgar_failure_is_recorded_not_raised(monkeypatch):
    ctx = _ctx()
    # the cache layer converts a raised exception into {"error": ...} before
    # it reaches fetch_related_exposure — simulate that directly here.
    monkeypatch.setattr(re_mod, "_mention_raw", lambda *a, **k: {"error": "edgar unavailable"})

    report = re_mod.fetch_related_exposure(ctx)

    assert report.exposures == []
    assert any("edgar unavailable" in v for v in ctx.fetch_errors.values())


def test_correlation_none_without_enough_overlapping_history(monkeypatch):
    dates = pd.date_range("2026-01-01", periods=40, freq="D")
    short_series = {
        "index": [d.date().isoformat() for d in dates],
        "close": [100.0 + i for i in range(40)],
    }
    monkeypatch.setattr(re_mod, "_price_series", lambda t: short_series)
    assert re_mod._correlation("NVDA", "SMCI") is None


def test_correlation_none_on_provider_error(monkeypatch):
    monkeypatch.setattr(
        re_mod, "_price_series", lambda t: {"error": "rate limited"}
    )
    assert re_mod._correlation("NVDA", "SMCI") is None
