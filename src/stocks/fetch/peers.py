"""-> PeerComparisonReport. Same metrics across 2-4 competitors + peer medians.

Peer resolution: a static sector map (predictable, cheap) with an optional FMP
stock-screener fallback when ``FMP_API_KEY`` is set. Unknown -> empty peer list
with a note, never a guess.
"""

from __future__ import annotations

import os

import numpy as np

from stocks.cache import cached
from stocks.models.reports import PeerComparisonReport
from stocks.providers import get_provider
from stocks.run_context import RunContext

# Small curated seed. Extend freely; this is not meant to be exhaustive.
_STATIC_PEERS: dict[str, list[str]] = {
    "AAPL": ["MSFT", "GOOGL", "DELL"],
    "MSFT": ["AAPL", "GOOGL", "AMZN"],
    "GOOGL": ["MSFT", "META", "AMZN"],
    "META": ["GOOGL", "SNAP", "PINS"],
    "AMZN": ["MSFT", "GOOGL", "WMT"],
    "NVDA": ["AMD", "INTC", "AVGO"],
    "AMD": ["NVDA", "INTC", "AVGO"],
    "TSLA": ["GM", "F", "RIVN"],
    "JPM": ["BAC", "WFC", "C"],
    "KO": ["PEP", "KDP", "MNST"],
    "PEP": ["KO", "MDLZ", "KDP"],
    "DIS": ["NFLX", "WBD", "PARA"],
    "NFLX": ["DIS", "WBD", "SPOT"],
}

# metric name -> raw yfinance key
_COMPARE_METRICS = {
    "pe_ratio": "trailingPE",
    "forward_pe": "forwardPE",
    "revenue_growth_yoy": "revenueGrowth",
    "profit_margin": "profitMargins",
}


@cached("peers")
def _peer_metrics(ticker: str) -> dict:
    try:
        raw = get_provider().fundamentals_raw(ticker)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    return {name: raw.get(key) for name, key in _COMPARE_METRICS.items()}


def resolve_peers(ctx: RunContext) -> tuple[list[str], str]:
    if os.environ.get("FMP_API_KEY"):
        try:
            from stocks.providers.fmp_provider import FMPProvider

            peers = [p for p in FMPProvider().peers(ctx.ticker) if p != ctx.ticker][:4]
            if peers:
                return peers, "FMP stock-peers"
        except Exception:  # noqa: BLE001
            pass
    if ctx.ticker in _STATIC_PEERS:
        return _STATIC_PEERS[ctx.ticker], "static table"
    return [], "unresolved"


def fetch_peer_comparison(ctx: RunContext) -> PeerComparisonReport:
    peers, method = resolve_peers(ctx)

    own = {
        name: (getattr(ctx.fundamentals, name).value if ctx.fundamentals
               and getattr(ctx.fundamentals, name) else None)
        for name in _COMPARE_METRICS
    }
    grid: dict[str, dict[str, float | None]] = {
        m: {ctx.ticker: own[m]} for m in _COMPARE_METRICS
    }
    for peer in peers:
        pm = _peer_metrics(peer)
        if isinstance(pm, dict) and "error" in pm:
            ctx.fetch_errors[f"peer:{peer}"] = pm["error"]
            continue
        for m in _COMPARE_METRICS:
            grid[m][peer] = pm.get(m)

    sector_median: dict[str, float | None] = {}
    for m in _COMPARE_METRICS:
        vals = [v for v in grid[m].values() if v is not None]
        med = round(float(np.median(vals)), 4) if vals else None
        sector_median[m] = med
        # back-fill peer_median onto the fundamentals metric
        peer_vals = [v for k, v in grid[m].items() if k != ctx.ticker and v is not None]
        if peer_vals and ctx.fundamentals and getattr(ctx.fundamentals, m, None):
            getattr(ctx.fundamentals, m).peer_median = round(
                float(np.median(peer_vals)), 4
            )

    report = PeerComparisonReport(
        ticker=ctx.ticker,
        peers=peers,
        peer_selection_method=method,
        metric_comparison=grid,
        sector_median=sector_median,
    )
    ctx.peers = report
    return report
