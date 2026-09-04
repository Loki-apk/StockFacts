"""-> FundamentalsReport. Raw provider fields wrapped in MetricWithContext.

Peer/sector medians are filled later by valuation_context + peers; this module
sets `value`, `unit`, `as_of`, `source` and leaves context fields None.
"""

from __future__ import annotations

from datetime import date

from stocks.cache import cached
from stocks.models.reports import FundamentalsReport, MetricWithContext
from stocks.providers import get_provider
from stocks.run_context import RunContext

_FIELD_UNITS = {
    "pe_ratio": ("trailingPE", "x"),
    "forward_pe": ("forwardPE", "x"),
    "ev_to_ebitda": ("enterpriseToEbitda", "x"),
    "revenue_growth_yoy": ("revenueGrowth", "pct"),
    "eps_growth_yoy": ("earningsGrowth", "pct"),
    "gross_margin": ("grossMargins", "pct"),
    "profit_margin": ("profitMargins", "pct"),
    "debt_to_equity": ("debtToEquity", "ratio"),
    "current_ratio": ("currentRatio", "ratio"),
    "return_on_equity": ("returnOnEquity", "pct"),
}


@cached("fundamentals")
def _fundamentals_raw(ticker: str) -> dict:
    return get_provider().fundamentals_raw(ticker)


@cached("income_stmt")
def _income_stmt(ticker: str) -> dict:
    try:
        return get_provider().income_stmt(ticker)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _interest_coverage(ticker: str) -> float | None:
    """EBIT / interest expense from the most recent annual income statement."""
    stmt = _income_stmt(ticker)
    if not isinstance(stmt, dict) or "error" in stmt or not stmt:
        return None
    latest = stmt[sorted(stmt, reverse=True)[0]]
    ebit = latest.get("EBIT") or latest.get("Operating Income")
    interest = latest.get("Interest Expense") or latest.get("Interest Expense Non Operating")
    if not ebit or not interest:
        return None
    return round(ebit / abs(interest), 2)


def _metric(value: float | None, unit: str, source: str) -> MetricWithContext | None:
    if value is None:
        return None
    return MetricWithContext(value=float(value), unit=unit, as_of=date.today(), source=source)


def fetch_fundamentals(ctx: RunContext) -> FundamentalsReport:
    raw = _fundamentals_raw(ctx.ticker)
    if isinstance(raw, dict) and "error" in raw:
        ctx.fetch_errors["fundamentals"] = raw["error"]
        raw = {}

    source = get_provider().name
    # yfinance reports debtToEquity as a percent (154.5 -> 1.545x); normalise.
    if raw.get("debtToEquity") is not None:
        raw = {**raw, "debtToEquity": raw["debtToEquity"] / 100.0}

    fields: dict[str, MetricWithContext | None] = {}
    missing: list[str] = []
    for name, (raw_key, unit) in _FIELD_UNITS.items():
        m = _metric(raw.get(raw_key), unit, source)
        fields[name] = m
        if m is None:
            missing.append(name)

    # fcf_margin is derived, not raw
    fcf, rev = raw.get("freeCashflow"), raw.get("totalRevenue")
    fields["fcf_margin"] = (
        _metric(fcf / rev, "pct", f"{source} (derived)")
        if fcf is not None and rev
        else None
    )
    fields["interest_coverage"] = _metric(
        _interest_coverage(ctx.ticker), "x", f"{source} (income stmt)"
    )
    if fields["interest_coverage"] is None and "interest_coverage" not in missing:
        missing.append("interest_coverage")

    report = FundamentalsReport(
        ticker=ctx.ticker,
        market_cap=raw.get("marketCap"),
        data_as_of=date.today(),
        source=source,
        provider_fallback_used=bool(getattr(get_provider(), "fallbacks", [])),
        missing_fields=missing,
        **fields,
    )
    ctx.fundamentals = report
    return report
