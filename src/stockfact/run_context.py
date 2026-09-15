"""The shared object that carries data through a whole run.

The deterministic layer fills this in step by step, and the LLM agents later
just read the specific fields they need off it — as actual structured data,
not as a wall of text passed down from whatever the last agent said.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date

from stockfact.models.reports import (
    Coverage,
    FilingExcerpt,
    FundamentalsReport,
    NarrativeSections,
    NewsSentimentReport,
    PeerComparisonReport,
    RawNewsItem,
    RelatedExposureReport,
    RiskFlag,
    RiskFlagReport,
    SubScores,
    TechnicalReport,
    ValuationContext,
)


@dataclass
class RunContext:
    # --- set by fetch/resolve.py ---
    ticker: str
    company_name: str
    sector: str
    industry: str
    currency: str
    exchange: str
    generated_at: date
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    # --- deterministic fetch layer (step 2) ---
    fundamentals: FundamentalsReport | None = None
    technicals: TechnicalReport | None = None
    valuation: ValuationContext | None = None
    peers: PeerComparisonReport | None = None
    related_exposure: RelatedExposureReport | None = None
    raw_news: list[RawNewsItem] = field(default_factory=list)
    filings: list[FilingExcerpt] = field(default_factory=list)
    risk_flags_raw: list[RiskFlag] = field(default_factory=list)

    # --- LLM layer (step 3) ---
    news_report: NewsSentimentReport | None = None
    risk_report: RiskFlagReport | None = None
    narrative: NarrativeSections | None = None

    # --- scoring + QA (steps 4-5) ---
    scores: SubScores | None = None
    coverage: Coverage | None = None
    qa_violations: list[str] = field(default_factory=list)

    # accumulates as providers fall back / errors are swallowed
    provider_fallbacks: list[str] = field(default_factory=list)
    fetch_errors: dict[str, str] = field(default_factory=dict)

    def deterministic_layer_usable(self) -> bool:
        """True if enough core data came back to bother running the LLM step."""
        return self.fundamentals is not None and self.technicals is not None
