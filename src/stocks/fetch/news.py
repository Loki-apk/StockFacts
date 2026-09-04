"""-> list[RawNewsItem]. Raw items only; the LLM labels themes/sentiment later."""

from __future__ import annotations

from stocks.cache import cached
from stocks.models.reports import RawNewsItem
from stocks.providers import news as news_provider
from stocks.run_context import RunContext

NEWS_WINDOW_DAYS = 7


@cached("news")
def _news_raw(ticker: str) -> list[dict]:
    return news_provider.fetch_company_news(ticker, window_days=NEWS_WINDOW_DAYS)


def fetch_news(ctx: RunContext) -> list[RawNewsItem]:
    raw = _news_raw(ctx.ticker)
    if isinstance(raw, dict) and "error" in raw:
        ctx.fetch_errors["news"] = raw["error"]
        return []

    items: list[RawNewsItem] = []
    for row in raw:
        try:
            items.append(RawNewsItem.model_validate(row))
        except Exception:  # noqa: BLE001 - skip malformed rows, don't fail the run
            continue
    ctx.raw_news = items
    return items
