"""Runs all the fetch modules above and keeps going even if one of them fails.

They can't all run at once, since some depend on others' output, so this
splits them into three phases:
  1. fundamentals, technicals, news, filings, related_exposure — nothing
     depends on anything else yet, so these all run together
  2. risk_signals (needs filings), peers (needs fundamentals)
  3. valuation_context (needs both fundamentals and peers)
Each phase uses a thread pool since these are all just waiting on network
calls to providers, not doing real CPU work.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor

from stockfact.fetch.filings import fetch_filings
from stockfact.fetch.fundamentals import fetch_fundamentals
from stockfact.fetch.news import fetch_news
from stockfact.fetch.peers import fetch_peer_comparison
from stockfact.fetch.related_exposure import fetch_related_exposure
from stockfact.fetch.risk_signals import fetch_risk_signals
from stockfact.fetch.technicals import fetch_technicals
from stockfact.fetch.valuation_context import fetch_valuation_context
from stockfact.run_context import RunContext

log = logging.getLogger(__name__)

_PARALLEL = os.environ.get("STOCKFACT_FETCH_SERIAL") != "1"


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

    from stockfact.providers import get_provider

    ctx.provider_fallbacks = list(getattr(get_provider(), "fallbacks", []))
    return ctx
