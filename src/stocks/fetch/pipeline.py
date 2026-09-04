"""Runs the deterministic fetch modules and records failures without crashing.

Three phases by data dependency:
  1. fundamentals, technicals, news, filings, related_exposure   (independent)
  2. risk_signals (needs filings), peers (needs fundamentals)
  3. valuation_context (needs fundamentals + peers)
Within a phase the fetches run on a thread pool (they're I/O bound on provider
HTTP calls).
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor

from stocks.fetch.filings import fetch_filings
from stocks.fetch.fundamentals import fetch_fundamentals
from stocks.fetch.news import fetch_news
from stocks.fetch.peers import fetch_peer_comparison
from stocks.fetch.related_exposure import fetch_related_exposure
from stocks.fetch.risk_signals import fetch_risk_signals
from stocks.fetch.technicals import fetch_technicals
from stocks.fetch.valuation_context import fetch_valuation_context
from stocks.run_context import RunContext

log = logging.getLogger(__name__)

_PARALLEL = os.environ.get("STOCKS_FETCH_SERIAL") != "1"


def _safe(ctx: RunContext, name: str, fn) -> None:
    try:
        fn(ctx)
    except Exception as exc:  # noqa: BLE001 - one broken fetch shouldn't kill the run
        ctx.fetch_errors[name] = f"{type(exc).__name__}: {exc}"
        log.warning("fetch %s failed for %s: %s", name, ctx.ticker, exc)


def _run_phase(ctx: RunContext, jobs: list[tuple[str, object]]) -> None:
    if not _PARALLEL or len(jobs) == 1:
        for name, fn in jobs:
            _safe(ctx, name, fn)
        return
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        list(pool.map(lambda job: _safe(ctx, job[0], job[1]), jobs))


def run_fetch_layer(ctx: RunContext) -> RunContext:
    _run_phase(
        ctx,
        [
            ("fundamentals", fetch_fundamentals),
            ("technicals", fetch_technicals),
            ("news", fetch_news),
            ("filings", fetch_filings),
            ("related_exposure", fetch_related_exposure),
        ],
    )
    _run_phase(
        ctx,
        [
            ("risk_signals", fetch_risk_signals),
            ("peers", fetch_peer_comparison),
        ],
    )
    _run_phase(ctx, [("valuation_context", fetch_valuation_context)])

    from stocks.providers import get_provider

    ctx.provider_fallbacks = list(getattr(get_provider(), "fallbacks", []))
    return ctx
