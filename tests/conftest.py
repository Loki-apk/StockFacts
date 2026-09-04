"""Shared fixtures. Real runs are offline: providers are replaced with fixture data.

Golden tickers to add under tests/fixtures/ (build-order step 1):
  - a megacap profitable name
  - a loss-maker (negative margins / no PE)
  - a high-debt name
  - a recent IPO (short price history)
  - a non-USD listing
"""

from __future__ import annotations

import os
from datetime import date

import pytest

from stocks.models.reports import (
    FundamentalsReport,
    MetricWithContext,
    NarrativeSections,
    NewsSentimentReport,
    PeerComparisonReport,
    RelatedExposureReport,
    RiskFlagReport,
    TechnicalReport,
    ValuationContext,
)
from stocks.run_context import RunContext

os.environ.setdefault("STOCKS_SKIP_LLM", "1")


@pytest.fixture(autouse=True)
def _isolated_dbs(tmp_path, monkeypatch):
    monkeypatch.setenv("STOCKS_CACHE_DB", str(tmp_path / "cache.db"))
    monkeypatch.setenv("STOCKS_RUNS_DB", str(tmp_path / "runs.db"))
    from stocks.providers import base

    base.reset_provider()


def _metric(value: float, unit: str = "x") -> MetricWithContext:
    return MetricWithContext(value=value, unit=unit, as_of=date(2026, 9, 3), source="fixture")


@pytest.fixture
def sample_ctx() -> RunContext:
    ctx = RunContext(
        ticker="TEST",
        company_name="Test Corp",
        sector="Technology",
        industry="Software",
        currency="USD",
        exchange="NASDAQ",
        generated_at=date(2026, 9, 3),
    )
    ctx.fundamentals = FundamentalsReport(
        ticker="TEST",
        pe_ratio=_metric(24.0),
        forward_pe=_metric(20.0),
        revenue_growth_yoy=_metric(0.18, "pct"),
        profit_margin=_metric(0.22, "pct"),
        debt_to_equity=_metric(0.75, "ratio"),
        current_ratio=_metric(1.8, "ratio"),
        interest_coverage=_metric(9.0, "x"),
        market_cap=1_500_000_000_000,
        data_as_of=date(2026, 9, 3),
        source="fixture",
    )
    ctx.technicals = TechnicalReport(
        ticker="TEST",
        current_price=190.0,
        fifty_two_week_high=210.0,
        fifty_two_week_low=150.0,
        pct_off_high=-0.095,
        pct_above_low=0.267,
        price_vs_sma50=0.02,
        price_vs_sma200=0.08,
        sma_50=186.0,
        sma_200=176.0,
        rsi_14=57.0,
        realized_vol_30d=0.24,
        beta=1.1,
        avg_volume_30d=52_000_000,
        data_as_of=date(2026, 9, 3),
    )
    ctx.valuation = ValuationContext(
        ticker="TEST",
        pe_now=24.0,
        pe_5y_median=21.0,
        pe_5y_low=15.0,
        pe_5y_high=34.0,
        pe_percentile_5y=62.0,
        data_as_of=date(2026, 9, 3),
    )
    ctx.peers = PeerComparisonReport(
        ticker="TEST",
        peers=["PEERA", "PEERB"],
        peer_selection_method="static table",
        metric_comparison={"pe_ratio": {"TEST": 24.0, "PEERA": 21.0, "PEERB": 26.0}},
        sector_median={"pe_ratio": 23.0},
    )
    ctx.news_report = NewsSentimentReport(
        ticker="TEST", window_days=7, items=[], dominant_themes=[],
        item_count_by_sentiment={},
    )
    ctx.risk_report = RiskFlagReport(ticker="TEST", flags=[])
    ctx.related_exposure = RelatedExposureReport(
        ticker="TEST", candidates_checked=0, exposures=[]
    )
    ctx.narrative = NarrativeSections(
        fundamentals_summary="PE is 24.0 versus a 5y median of 21.0.",
        technical_position="Price 190.00 is 9.5% off the 52-week high.",
        news_and_themes="No notable news in the window.",
        peer_context="PE 24.0 sits between peers at 21.0 and 26.0.",
        risk_summary="No risk flags raised.",
        related_exposure_summary="No documented related exposure on file.",
    )
    return ctx
