"""-> TechnicalReport. SMA, RSI, beta, volatility — all computed here with numpy/pandas.

None of this ever goes near the LLM. It's just arithmetic, so it belongs in
plain Python where the answer is actually trustworthy.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from stockfact.cache import cached, get_cached_or_fetch
from stockfact.models.reports import TechnicalReport
from stockfact.providers import get_provider
from stockfact.run_context import RunContext

_BENCHMARK = "^GSPC"  # S&P 500, for beta


def _history_payload(ticker: str) -> dict:
    df = get_provider().price_history(ticker, period="5y")
    return {
        "index": [d.isoformat() for d in df.index.date],
        "close": [float(x) for x in df["Close"].tolist()],
        "volume": [float(x) for x in df["Volume"].tolist()],
    }


@cached("price")
def _history_json(ticker: str) -> dict:
    return _history_payload(ticker)


def _benchmark_returns() -> pd.Series | None:
    payload, _ = get_cached_or_fetch(
        f"price:{_BENCHMARK}", lambda: _history_payload(_BENCHMARK), ttl_hours=6
    )
    if isinstance(payload, dict) and "error" in payload:
        return None
    s = pd.Series(payload["close"], index=pd.to_datetime(payload["index"]), dtype="float64")
    return s.pct_change().dropna()


def _rsi(close: pd.Series, window: int = 14) -> float | None:
    if len(close) <= window:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    last = rsi.iloc[-1]
    return float(last) if pd.notna(last) else None


def _beta(stock_close: pd.Series, stock_index: pd.DatetimeIndex) -> float | None:
    bench = _benchmark_returns()
    if bench is None or len(bench) < 60:
        return None
    stock_ret = pd.Series(stock_close.values, index=stock_index).pct_change().dropna()
    joined = pd.concat([stock_ret, bench], axis=1, join="inner").dropna()
    joined = joined.tail(252 * 2)
    if len(joined) < 60:
        return None
    s, m = joined.iloc[:, 0].to_numpy(), joined.iloc[:, 1].to_numpy()
    var = float(np.var(m))
    if var == 0:
        return None
    return float(np.cov(s, m)[0, 1] / var)


def fetch_technicals(ctx: RunContext) -> TechnicalReport:
    raw = _history_json(ctx.ticker)
    if isinstance(raw, dict) and "error" in raw:
        ctx.fetch_errors["price"] = raw["error"]
        raise ValueError(f"no price history for {ctx.ticker}: {raw['error']}")

    idx = pd.to_datetime(raw["index"])
    close = pd.Series(raw["close"], dtype="float64")
    volume = pd.Series(raw["volume"], dtype="float64")
    price = float(close.iloc[-1])

    last_year = close.tail(252)
    hi, lo = float(last_year.max()), float(last_year.min())
    sma50 = float(close.tail(50).mean()) if len(close) >= 50 else None
    sma200 = float(close.tail(200).mean()) if len(close) >= 200 else None
    daily_ret = close.pct_change().dropna()
    vol30 = float(daily_ret.tail(30).std() * np.sqrt(252)) if len(daily_ret) >= 30 else None
    avg_vol30 = float(volume.tail(30).mean()) if len(volume) >= 30 else None
    avg_vol90 = float(volume.tail(90).mean()) if len(volume) >= 90 else None
    rsi = _rsi(close)
    beta = _beta(close, idx)

    report = TechnicalReport(
        ticker=ctx.ticker,
        current_price=round(price, 4),
        fifty_two_week_high=round(hi, 4),
        fifty_two_week_low=round(lo, 4),
        pct_off_high=round(price / hi - 1, 4) if hi else 0.0,
        pct_above_low=round(price / lo - 1, 4) if lo else 0.0,
        price_vs_sma50=round(price / sma50 - 1, 4) if sma50 else None,
        price_vs_sma200=round(price / sma200 - 1, 4) if sma200 else None,
        sma_50=round(sma50, 4) if sma50 else None,
        sma_200=round(sma200, 4) if sma200 else None,
        rsi_14=round(rsi, 2) if rsi is not None else None,
        realized_vol_30d=round(vol30, 4) if vol30 else None,
        beta=round(beta, 3) if beta is not None else None,
        avg_volume_30d=round(avg_vol30) if avg_vol30 else None,
        volume_vs_90d_avg=round(avg_vol30 / avg_vol90 - 1, 4)
        if avg_vol30 and avg_vol90
        else None,
        data_as_of=date.today(),
    )
    ctx.technicals = report
    return report
