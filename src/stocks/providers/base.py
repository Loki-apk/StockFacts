"""Provider protocol + a primary/fallback wrapper.

Every method returns plain dicts/lists (JSON-serialisable) or raises
``ProviderError``. Nothing here raises a bare exception into the pipeline — the
cache layer and ``FallbackProvider`` convert failures into structured results.
"""

from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable

import pandas as pd


class ProviderError(RuntimeError):
    """A provider could not answer. Carries the datatype that failed."""

    def __init__(self, datatype: str, message: str) -> None:
        super().__init__(f"[{datatype}] {message}")
        self.datatype = datatype


@runtime_checkable
class MarketDataProvider(Protocol):
    name: str

    def resolve(self, ticker: str) -> dict[str, Any]:
        """-> {company_name, sector, industry, currency, exchange}. Raises if unknown."""

    def fundamentals_raw(self, ticker: str) -> dict[str, Any]:
        """Flat dict of raw fundamental fields (trailingPE, revenueGrowth, ...)."""

    def price_history(self, ticker: str, period: str = "5y") -> pd.DataFrame:
        """OHLCV DataFrame indexed by date. Used to compute indicators locally."""

    def calendar(self, ticker: str) -> dict[str, Any]:
        """Earnings date(s), ex-dividend date, etc."""

    def insider_transactions(self, ticker: str) -> list[dict[str, Any]]:
        ...

    def recommendations(self, ticker: str) -> list[dict[str, Any]]:
        """Analyst rating buckets over recent months."""

    def upgrades_downgrades(self, ticker: str) -> list[dict[str, Any]]:
        """Firm-level rating actions with dates (for detecting recent downgrades)."""

    def income_stmt(self, ticker: str, *, quarterly: bool = False) -> dict[str, Any]:
        """Income statement as {period: {row_label: value}}, newest first."""

    def short_interest(self, ticker: str) -> dict[str, Any]:
        ...


class FallbackProvider:
    """Try ``primary``; on ``ProviderError`` or empty core fields, try ``secondary``.

    Records which datatypes fell back so ``Coverage.provider_fallbacks`` can
    report it.
    """

    def __init__(
        self, primary: MarketDataProvider, secondary: MarketDataProvider | None
    ) -> None:
        self.primary = primary
        self.secondary = secondary
        self.fallbacks: list[str] = []

    @property
    def name(self) -> str:
        return getattr(self.primary, "name", "primary")

    def _try(self, method: str, *args: Any, **kwargs: Any) -> Any:
        try:
            return getattr(self.primary, method)(*args, **kwargs)
        except ProviderError:
            if self.secondary is None:
                raise
            self.fallbacks.append(method)
            return getattr(self.secondary, method)(*args, **kwargs)

    def __getattr__(self, item: str):  # delegate protocol methods through _try
        if item in {
            "resolve",
            "fundamentals_raw",
            "price_history",
            "calendar",
            "insider_transactions",
            "recommendations",
            "upgrades_downgrades",
            "income_stmt",
            "short_interest",
        }:
            return lambda *a, **k: self._try(item, *a, **k)
        raise AttributeError(item)


_PROVIDER: FallbackProvider | None = None


def get_provider() -> FallbackProvider:
    """Assemble (once per process) the configured primary + fallback pair.

    Memoised so ``.fallbacks`` accumulates across the whole run.
    TODO: read provider choice from env (STOCKS_PRIMARY / STOCKS_FALLBACK).
    """
    global _PROVIDER
    if _PROVIDER is not None:
        return _PROVIDER

    from stocks.providers.yfinance_provider import YFinanceProvider

    yf_provider = YFinanceProvider()
    fmp_provider: MarketDataProvider | None = None
    try:
        from stocks.providers.fmp_provider import FMPProvider

        fmp_provider = FMPProvider()
    except Exception:  # noqa: BLE001 - fallback is optional
        fmp_provider = None

    if os.environ.get("STOCKS_PRIMARY", "").lower() == "fmp" and fmp_provider is not None:
        _PROVIDER = FallbackProvider(fmp_provider, yf_provider)
    else:
        _PROVIDER = FallbackProvider(yf_provider, fmp_provider)
    return _PROVIDER


def reset_provider() -> None:
    """Test hook — drop the memoised provider."""
    global _PROVIDER
    _PROVIDER = None
