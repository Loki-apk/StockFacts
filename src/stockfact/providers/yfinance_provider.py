"""The primary provider — yfinance.

yfinance itself is scraping Yahoo under the hood, so it breaks whenever
Yahoo changes something and rate-limits pretty aggressively. Every method
below just wraps yfinance's quirks (empty frames, missing keys, random
exceptions) so they come out as either a clean result or a ``ProviderError``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from stockfact.providers.base import ProviderError

try:  # keep import failure from killing `import stockfact`
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None


def _isnan(val: Any) -> bool:
    try:
        return val != val  # NaN is the only value not equal to itself
    except Exception:  # noqa: BLE001
        return False


class YFinanceProvider:
    name = "yfinance"

    def _ticker(self, ticker: str):
        if yf is None:
            raise ProviderError("import", "yfinance is not installed")
        return yf.Ticker(ticker)

    def resolve(self, ticker: str) -> dict[str, Any]:
        info = self._info(ticker)
        name = info.get("longName") or info.get("shortName")
        if not name:
            raise ProviderError("resolve", f"unknown ticker {ticker!r}")
        return {
            "company_name": name,
            "sector": info.get("sector") or "unknown",
            "industry": info.get("industry") or "unknown",
            "currency": info.get("currency") or "USD",
            "exchange": info.get("fullExchangeName") or info.get("exchange") or "unknown",
        }

    def fundamentals_raw(self, ticker: str) -> dict[str, Any]:
        info = self._info(ticker)
        keys = (
            "trailingPE forwardPE enterpriseToEbitda revenueGrowth earningsGrowth "
            "grossMargins profitMargins freeCashflow totalRevenue debtToEquity "
            "currentRatio returnOnEquity marketCap beta"
        ).split()
        raw = {k: info.get(k) for k in keys}
        if all(v is None for v in raw.values()):
            raise ProviderError("fundamentals", f"no fundamentals for {ticker!r}")
        return raw

    def price_history(self, ticker: str, period: str = "5y") -> pd.DataFrame:
        try:
            hist = self._ticker(ticker).history(period=period, auto_adjust=True)
        except Exception as exc:  # noqa: BLE001
            raise ProviderError("price", str(exc)) from exc
        if hist is None or hist.empty:
            raise ProviderError("price", f"empty history for {ticker!r}")
        return hist

    def calendar(self, ticker: str) -> dict[str, Any]:
        try:
            cal = self._ticker(ticker).calendar
        except Exception as exc:  # noqa: BLE001
            raise ProviderError("calendar", str(exc)) from exc
        if isinstance(cal, pd.DataFrame):
            return cal.to_dict()
        return dict(cal or {})

    def insider_transactions(self, ticker: str) -> list[dict[str, Any]]:
        return self._frame_to_records(
            lambda: self._ticker(ticker).insider_transactions, "insider"
        )

    def recommendations(self, ticker: str) -> list[dict[str, Any]]:
        return self._frame_to_records(
            lambda: self._ticker(ticker).recommendations, "recommendations"
        )

    def upgrades_downgrades(self, ticker: str) -> list[dict[str, Any]]:
        return self._frame_to_records(
            lambda: self._ticker(ticker).upgrades_downgrades, "upgrades_downgrades"
        )

    def income_stmt(self, ticker: str, *, quarterly: bool = False) -> dict[str, Any]:
        """{'YYYY-MM-DD': {row_label: value, ...}, ...}, newest column first."""
        try:
            t = self._ticker(ticker)
            frame = t.quarterly_income_stmt if quarterly else t.income_stmt
        except Exception as exc:  # noqa: BLE001
            raise ProviderError("income_stmt", str(exc)) from exc
        if frame is None or frame.empty:
            raise ProviderError("income_stmt", f"no income statement for {ticker!r}")
        out: dict[str, Any] = {}
        for col in frame.columns:
            key = col.date().isoformat() if hasattr(col, "date") else str(col)
            out[key] = {
                str(idx): (None if _isnan(val) else float(val))
                for idx, val in frame[col].items()
            }
        return out

    def short_interest(self, ticker: str) -> dict[str, Any]:
        info = self._info(ticker)
        return {
            "short_ratio": info.get("shortRatio"),
            "short_percent_of_float": info.get("shortPercentOfFloat"),
            "shares_short": info.get("sharesShort"),
        }

    # --- helpers ---
    def _info(self, ticker: str) -> dict[str, Any]:
        try:
            info = self._ticker(ticker).info
        except Exception as exc:  # noqa: BLE001
            raise ProviderError("info", str(exc)) from exc
        if not info or not isinstance(info, dict):
            raise ProviderError("info", f"no info for {ticker!r}")
        return info

    @staticmethod
    def _frame_to_records(getter, datatype: str) -> list[dict[str, Any]]:
        try:
            frame = getter()
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(datatype, str(exc)) from exc
        if frame is None or (isinstance(frame, pd.DataFrame) and frame.empty):
            return []
        return frame.reset_index().to_dict("records")
