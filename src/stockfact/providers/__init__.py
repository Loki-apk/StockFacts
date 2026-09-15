"""Where raw market data comes from. The fetch layer only ever talks to the
``MarketDataProvider`` protocol below, never to yfinance (or anything else)
directly — so swapping providers doesn't mean touching the fetch code.
"""

from stockfact.providers.base import MarketDataProvider, ProviderError, get_provider

__all__ = ["MarketDataProvider", "ProviderError", "get_provider"]
