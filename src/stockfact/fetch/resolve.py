"""Step 1: turn a ticker into the first RunContext fields — company name, sector, etc. No LLM."""

from __future__ import annotations

from datetime import date

from stockfact.cache import cached
from stockfact.providers import get_provider
from stockfact.run_context import RunContext


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
