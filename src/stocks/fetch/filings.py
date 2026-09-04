"""-> list[FilingExcerpt] from SEC EDGAR (Item 1A risk factors, MD&A)."""

from __future__ import annotations

from stocks.cache import cached
from stocks.models.reports import FilingExcerpt
from stocks.providers import edgar
from stocks.run_context import RunContext


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
