"""Raw data access. Provider-agnostic — the fetch layer depends on the
``MarketDataProvider`` protocol, not on yfinance directly.
"""

from stocks.providers.base import MarketDataProvider, ProviderError, get_provider

__all__ = ["MarketDataProvider", "ProviderError", "get_provider"]
