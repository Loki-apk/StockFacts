"""Pulls the risk-factors (Item 1A) and MD&A sections out of SEC filings -> list[FilingExcerpt]."""

from __future__ import annotations

from stockfact.cache import cached
from stockfact.models.reports import FilingExcerpt
from stockfact.providers import edgar
from stockfact.run_context import RunContext


@cached("filings")
def _filings_raw(ticker: str) -> list[dict]:
    return edgar.risk_factor_excerpts(ticker)


def fetch_filings(ctx: RunContext) -> list[FilingExcerpt]:
    try:
        raw = _filings_raw(ctx.ticker)
    except Exception as exc:  # noqa: BLE001 - filings are optional context
        ctx.fetch_errors["filings"] = f"{type(exc).__name__}: {exc}"
        return []
    if isinstance(raw, dict) and "error" in raw:
        ctx.fetch_errors["filings"] = raw["error"]
        return []

    excerpts: list[FilingExcerpt] = []
    for row in raw:
        try:
            excerpts.append(FilingExcerpt.model_validate(row))
        except Exception:  # noqa: BLE001
            continue
    ctx.filings = excerpts
    return excerpts
