"""Step 1: ticker -> RunContext. Deterministic, no LLM."""

from __future__ import annotations

from datetime import date

from stocks.cache import cached
from stocks.providers import get_provider
from stocks.run_context import RunContext


@cached("resolve")
def _resolve_raw(ticker: str) -> dict:
    return get_provider().resolve(ticker)


def resolve(ticker: str) -> RunContext:
    """Confirm the ticker exists and seed the run context.

    Raises ValueError if the ticker cannot be resolved by any provider.
    """
    ticker = ticker.strip().upper()
    data = _resolve_raw(ticker)
    if isinstance(data, dict) and "error" in data:
        raise ValueError(f"could not resolve ticker {ticker!r}: {data['error']}")
    return RunContext(
        ticker=ticker,
        company_name=data["company_name"],
        sector=data["sector"],
        industry=data["industry"],
        currency=data["currency"],
        exchange=data["exchange"],
        generated_at=date.today(),
    )
