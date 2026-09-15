"""-> RelatedExposureReport. Finds other companies whose *business* is actually
tied to the ticker you searched — a customer, supplier, or partner named in
their own SEC filing — plus how correlated their prices have been.

The whole thing is deterministic and costs nothing in LLM calls:
  - Candidates start from a small hand-curated table (same idea as
    ``_STATIC_PEERS`` in fetch/peers.py — not complete, add to it as you
    verify more companies).
  - A candidate only makes it into the report if the company being evaluated
    is actually named, verbatim, somewhere in that candidate's latest 10-K
    (checked via SEC EDGAR — ``providers.edgar.find_relationship_mention``).
    The quote you see in the report is copied straight out of the filing,
    never written or paraphrased by anything.
  - The correlation number is just numpy/pandas on price history already on
    hand, the same way fetch/technicals.py works out beta.

If a ticker isn't in the candidate table, or nothing checks out, you get an
empty list plus how many candidates were checked — never a guessed
relationship.
"""

from __future__ import annotations

import pandas as pd

from stockfact.cache import cached
from stockfact.models.reports import RelatedExposure, RelatedExposureReport
from stockfact.providers import edgar, get_provider
from stockfact.run_context import RunContext

# ticker -> [(related_ticker, relationship_type), ...]. Relationship is from
# the *related* company's point of view, e.g. NVDA -> ("SMCI", "customer")
# means Super Micro is a documented NVIDIA customer.
_STATIC_EXPOSURES: dict[str, list[tuple[str, str]]] = {
    "NVDA": [("SMCI", "customer"), ("DELL", "customer"), ("MSFT", "customer"), ("TSM", "supplier")],
    "TSM": [("AAPL", "customer"), ("NVDA", "customer"), ("AMD", "customer")],
    "AAPL": [("TSM", "supplier"), ("QCOM", "supplier")],
    "MSFT": [("NVDA", "supplier")],
    "AMD": [("TSM", "supplier"), ("MSFT", "customer")],
}

_MAX_CANDIDATES = 6
_NAME_SUFFIXES = (
    " Corporation", " Incorporated", " Inc.", " Inc", " Limited", " Ltd.",
    " Co.", " Company", " Group", " Holdings", " plc", " PLC",
)


def _search_alias(company_name: str) -> str:
    """A short, distinctive token to grep for — filings rarely spell out the
    full legal suffix ('Corporation', 'Inc.', 'Limited')."""
    for suffix in _NAME_SUFFIXES:
        if company_name.endswith(suffix):
            return company_name[: -len(suffix)].strip()
    return company_name


@cached("resolve")
def _resolve_cached(ticker: str) -> dict:
    return get_provider().resolve(ticker)


@cached("related_exposure")
def _mention_raw(_cache_key: str, candidate_ticker: str, needle: str) -> dict:
    hit = edgar.find_relationship_mention(candidate_ticker, needle)
    return hit or {}


@cached("price")
def _price_series(ticker: str) -> dict:
    df = get_provider().price_history(ticker, period="2y")
    return {
        "index": [d.isoformat() for d in df.index.date],
        "close": [float(x) for x in df["Close"].tolist()],
    }


def _correlation(ticker_a: str, ticker_b: str) -> float | None:
    a_raw, b_raw = _price_series(ticker_a), _price_series(ticker_b)
    if not isinstance(a_raw, dict) or "error" in a_raw:
        return None
    if not isinstance(b_raw, dict) or "error" in b_raw:
        return None
    a_series = pd.Series(a_raw["close"], index=pd.to_datetime(a_raw["index"]), dtype="float64")
    b_series = pd.Series(b_raw["close"], index=pd.to_datetime(b_raw["index"]), dtype="float64")
    a, b = a_series.pct_change(), b_series.pct_change()
    joined = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(joined) < 60:
        return None
    corr = joined.iloc[:, 0].corr(joined.iloc[:, 1])
    return round(float(corr), 3) if corr == corr else None  # noqa: PLR0124 - NaN check


def fetch_related_exposure(ctx: RunContext) -> RelatedExposureReport:
    candidates = _STATIC_EXPOSURES.get(ctx.ticker, [])[:_MAX_CANDIDATES]
    needle = _search_alias(ctx.company_name)

    exposures: list[RelatedExposure] = []
    for related_ticker, rel_type in candidates:
        raw = _mention_raw(f"{ctx.ticker}:{related_ticker}", related_ticker, needle)
        if not isinstance(raw, dict) or "quote" not in raw:
            if isinstance(raw, dict) and "error" in raw:
                ctx.fetch_errors[f"related_exposure:{related_ticker}"] = raw["error"]
            continue  # no confirmed mention -> not surfaced, never a guess

        name_raw = _resolve_cached(related_ticker)
        related_name = (
            name_raw["company_name"]
            if isinstance(name_raw, dict) and "error" not in name_raw
            else related_ticker
        )
        exposures.append(
            RelatedExposure(
                related_ticker=related_ticker,
                related_company_name=related_name,
                relationship_type=rel_type,
                evidence_quote=raw["quote"],
                evidence_source=raw["url"],
                evidence_filed_at=raw["filed_at"],
                price_correlation_2y=_correlation(ctx.ticker, related_ticker),
            )
        )

    report = RelatedExposureReport(
        ticker=ctx.ticker,
        candidates_checked=len(candidates),
        exposures=exposures,
    )
    ctx.related_exposure = report
    return report
