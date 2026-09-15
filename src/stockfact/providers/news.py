"""Grabs company news. yfinance's own feed works out of the box with no
setup; Finnhub takes over instead if ``FINNHUB_API_KEY`` is set, since it
gives more history and cleaner data.

Everything here is just raw dicts shaped to match ``RawNewsItem`` — no
summarizing happens in this file, that's the News & Theme agent's job later.
"""

from __future__ import annotations

import os
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx

_FINNHUB = "https://finnhub.io/api/v1/company-news"
_TIMEOUT = httpx.Timeout(20.0)


def fetch_company_news(
    ticker: str, *, window_days: int = 7, limit: int = 25
) -> list[dict[str, Any]]:
    """-> [{headline, source, url, published_at, snippet}], newest first, deduped."""
    cutoff = date.today() - timedelta(days=window_days)
    if os.environ.get("FINNHUB_API_KEY"):
        try:
            items = _from_finnhub(ticker, cutoff)
        except Exception:  # noqa: BLE001 - fall through to yfinance
            items = _from_yfinance(ticker)
    else:
        items = _from_yfinance(ticker)

    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for it in items:
        if it["published_at"] < cutoff:
            continue
        key = (it["headline"].strip().lower(), it["source"].lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    out.sort(key=lambda i: i["published_at"], reverse=True)
    return out[:limit]


def _from_finnhub(ticker: str, cutoff: date) -> list[dict[str, Any]]:
    params = {
        "symbol": ticker.upper(),
        "from": cutoff.isoformat(),
        "to": date.today().isoformat(),
        "token": os.environ["FINNHUB_API_KEY"],
    }
    with httpx.Client(timeout=_TIMEOUT) as c:
        rows = c.get(_FINNHUB, params=params).raise_for_status().json()
    out = []
    for r in rows:
        if not r.get("headline") or not r.get("url"):
            continue
        out.append(
            {
                "headline": r["headline"],
                "source": r.get("source") or "Finnhub",
                "url": r["url"],
                "published_at": datetime.fromtimestamp(
                    r["datetime"], tz=UTC
                ).date(),
                "snippet": (r.get("summary") or "").strip() or None,
            }
        )
    return out


def _from_yfinance(ticker: str) -> list[dict[str, Any]]:
    import yfinance as yf

    raw = yf.Ticker(ticker).news or []
    out = []
    for item in raw:
        c = item.get("content", item)
        url = (
            (c.get("canonicalUrl") or {}).get("url")
            or (c.get("clickThroughUrl") or {}).get("url")
            or c.get("previewUrl")
        )
        title = c.get("title")
        if not url or not title:
            continue
        out.append(
            {
                "headline": title,
                "source": (c.get("provider") or {}).get("displayName") or "Yahoo Finance",
                "url": url,
                "published_at": _parse_dt(c.get("pubDate") or c.get("displayTime")),
                "snippet": (c.get("summary") or c.get("description") or "").strip() or None,
            }
        )
    return out


def _parse_dt(value: Any) -> date:
    if not value:
        return date.today()
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except ValueError:
        return date.today()
