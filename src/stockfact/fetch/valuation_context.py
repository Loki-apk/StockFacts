"""-> ValuationContext, i.e. where today's P/E actually sits against this
stock's own history.

"P/E 24" on its own doesn't tell you much. "P/E 24, the 78th percentile of
its own 5-year range, vs. a peer median of 21" is something you can actually
reason about.

One catch: yfinance's free tier only gives ~5 annual and ~5 quarterly income
statement periods, not a full daily history of fundamentals. So this module
builds a rough trailing-twelve-month EPS curve by interpolating between the
annual Diluted EPS points, applies that to the daily close price, and looks
at the resulting distribution of P/E values. It's coarser than a paid
fundamentals feed would give you, and `notes` says so.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from stockfact.cache import cached, get_cached_or_fetch
from stockfact.fetch.technicals import _history_payload
from stockfact.models.reports import ValuationContext
from stockfact.providers import get_provider


@cached("valuation_context")
def _eps_history(ticker: str) -> dict:
    """{'YYYY-MM-DD': diluted_eps_ttm} from annual income statements, newest first."""
    try:
        stmt = get_provider().income_stmt(ticker)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    out: dict[str, float] = {}
    for period, rows in stmt.items():
        eps = rows.get("Diluted EPS") or rows.get("Basic EPS")
        if eps:
            out[period] = float(eps)
    return out or {"error": "no EPS rows"}


def _pe_series(ticker: str) -> pd.Series | None:
    eps_hist = _eps_history(ticker)
    if "error" in eps_hist:
        return None
    prices, _ = get_cached_or_fetch(
        f"price:{ticker.upper()}", lambda: _history_payload(ticker), ttl_hours=1
    )
    if isinstance(prices, dict) and "error" in prices:
        return None

    close = pd.Series(
        prices["close"], index=pd.to_datetime(prices["index"]), dtype="float64"
    )
    eps_points = pd.Series(
        {pd.Timestamp(k): v for k, v in eps_hist.items()}
    ).sort_index()
    if len(eps_points) < 2:
        return None

    # interpolate an annual-EPS curve onto the daily price index
    eps_curve = (
        eps_points.reindex(eps_points.index.union(close.index))
        .interpolate(method="time")
        .ffill()
        .bfill()
        .reindex(close.index)
    )
    pe = (close / eps_curve).replace([np.inf, -np.inf], np.nan).dropna()
    pe = pe[(pe > 0) & (pe < 500)]  # drop nonsense from near-zero EPS
    return pe.tail(252 * 5) if len(pe) else None


def fetch_valuation_context(ctx) -> ValuationContext:
    pe_now = None
    if ctx.fundamentals and ctx.fundamentals.pe_ratio:
        pe_now = ctx.fundamentals.pe_ratio.value

    notes: list[str] = []
    median = low = high = pctile = None
    series = _pe_series(ctx.ticker)
    if series is not None and len(series) >= 60:
        median = round(float(series.median()), 2)
        low = round(float(series.quantile(0.05)), 2)
        high = round(float(series.quantile(0.95)), 2)
        if pe_now:
            pctile = round(float((series < pe_now).mean() * 100), 1)
        notes.append(
            f"P/E band from interpolated annual EPS over {len(series)} trading days "
            "(yfinance free tier — coarse)."
        )
        # back-fill context onto the fundamentals metric
        if ctx.fundamentals and ctx.fundamentals.pe_ratio:
            ctx.fundamentals.pe_ratio.self_median_3y = round(
                float(series.tail(252 * 3).median()), 2
            )
            ctx.fundamentals.pe_ratio.self_percentile_5y = pctile
    else:
        notes.append("Insufficient EPS/price history for a self-relative P/E band.")

    fwd_vs_peers = None
    if ctx.peers and ctx.fundamentals and ctx.fundamentals.forward_pe:
        peer_fpe = [
            v
            for k, v in ctx.peers.metric_comparison.get("forward_pe", {}).items()
            if k != ctx.ticker and v
        ]
        if peer_fpe:
            fwd_vs_peers = round(
                ctx.fundamentals.forward_pe.value / float(np.median(peer_fpe)) - 1, 3
            )

    report = ValuationContext(
        ticker=ctx.ticker,
        pe_now=round(pe_now, 2) if pe_now else None,
        pe_5y_median=median,
        pe_5y_low=low,
        pe_5y_high=high,
        pe_percentile_5y=pctile,
        fwd_pe_vs_peer_median=fwd_vs_peers,
        notes=notes,
        data_as_of=date.today(),
    )
    ctx.valuation = report
    return report
