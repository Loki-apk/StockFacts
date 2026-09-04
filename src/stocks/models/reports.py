"""Pydantic report models — the anti-hallucination backbone.

Every agent and fetch module returns one of these. Numbers live in typed fields
sourced from tool calls; prose lives only in the narrative model, which the QA
pass scrubs against these numbers.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

from stocks.models.enums import FlagType, Sentiment, Severity, Theme


class MetricWithContext(BaseModel):
    """A single metric plus the context that makes it mean something.

    An absolute PE of 24 is close to useless; 24 against a 5y median of 19 and a
    peer median of 21 is a fact you can reason about.
    """

    value: float | None
    unit: str  # "ratio" | "pct" | "USD" | "x"
    self_median_3y: float | None = None
    self_percentile_5y: float | None = None  # 0-100, where `value` sits in its own range
    peer_median: float | None = None
    sector_median: float | None = None
    as_of: date
    source: str


class FundamentalsReport(BaseModel):
    ticker: str
    pe_ratio: MetricWithContext | None = None
    forward_pe: MetricWithContext | None = None
    ev_to_ebitda: MetricWithContext | None = None
    revenue_growth_yoy: MetricWithContext | None = None
    eps_growth_yoy: MetricWithContext | None = None
    gross_margin: MetricWithContext | None = None
    profit_margin: MetricWithContext | None = None
    fcf_margin: MetricWithContext | None = None
    debt_to_equity: MetricWithContext | None = None
    interest_coverage: MetricWithContext | None = None
    current_ratio: MetricWithContext | None = None
    return_on_equity: MetricWithContext | None = None
    market_cap: float | None = None
    data_as_of: date
    source: str = "yfinance"
    provider_fallback_used: bool = False
    missing_fields: list[str] = Field(default_factory=list)


class TechnicalReport(BaseModel):
    ticker: str
    current_price: float
    fifty_two_week_high: float
    fifty_two_week_low: float
    pct_off_high: float
    pct_above_low: float
    price_vs_sma50: float | None = None  # (price / sma50) - 1
    price_vs_sma200: float | None = None
    sma_50: float | None = None
    sma_200: float | None = None
    rsi_14: float | None = None
    realized_vol_30d: float | None = None
    beta: float | None = None
    avg_volume_30d: float | None = None
    volume_vs_90d_avg: float | None = None
    data_as_of: date


class ValuationContext(BaseModel):
    ticker: str
    pe_now: float | None = None
    pe_5y_median: float | None = None
    pe_5y_low: float | None = None
    pe_5y_high: float | None = None
    pe_percentile_5y: float | None = None  # 0-100
    fwd_pe_vs_peer_median: float | None = None
    notes: list[str] = Field(default_factory=list)  # deterministic strings only
    data_as_of: date


class RawNewsItem(BaseModel):
    """Pre-LLM. What the news provider gave us, unedited."""

    headline: str
    source: str
    url: str
    published_at: date
    snippet: str | None = None


class NewsSentimentItem(BaseModel):
    """Post-LLM. `url` is required — no item survives without a link back."""

    headline: str
    source: str
    url: str
    published_at: date
    theme: Theme
    sentiment: Sentiment
    rationale: str  # one clause, grounded in the headline text


class NewsSentimentReport(BaseModel):
    ticker: str
    window_days: int
    items: list[NewsSentimentItem]
    dominant_themes: list[Theme]
    item_count_by_sentiment: dict[str, int] = Field(default_factory=dict)


class FilingExcerpt(BaseModel):
    filing_type: str  # "10-K" | "10-Q"
    section: str  # "Item 1A" | "MD&A"
    filed_at: date
    url: str
    text: str  # verbatim excerpt


class RiskFlag(BaseModel):
    flag_type: FlagType
    description: str
    severity: Severity
    source: str
    as_of: date | None = None


class RiskFlagReport(BaseModel):
    ticker: str
    flags: list[RiskFlag]


class PeerComparisonReport(BaseModel):
    ticker: str
    peers: list[str]
    peer_selection_method: str  # "sector+size" | "static table"
    metric_comparison: dict[str, dict[str, float | None]]
    sector_median: dict[str, float | None] = Field(default_factory=dict)


class RelatedExposure(BaseModel):
    """A company whose business is documented as tied to the ticker — not just
    a sector peer. Every entry is backed by a verbatim quote pulled straight
    from a real SEC filing; nothing here is inferred, summarized, or generated.
    """

    related_ticker: str
    related_company_name: str
    relationship_type: str  # "customer" | "supplier" | "partner"
    evidence_quote: str  # verbatim text surrounding the match, copied from the filing
    evidence_source: str  # filing URL
    evidence_filed_at: date
    price_correlation_2y: float | None = None  # computed locally from price history, not from the filing


class RelatedExposureReport(BaseModel):
    ticker: str
    candidates_checked: int
    exposures: list[RelatedExposure] = Field(default_factory=list)
    method: str = "static candidate table + SEC EDGAR filing confirmation"


class SubScores(BaseModel):
    """Sub-scores are the product. The rollup is a labeled convenience."""

    valuation: float | None  # 0-100
    growth: float | None
    financial_health: float | None
    momentum: float | None
    explanations: dict[str, str] = Field(default_factory=dict)
    rollup: float | None = None
    rollup_is_heuristic: bool = True


class Coverage(BaseModel):
    fields_populated: int
    fields_total: int
    news_window_days: int
    news_item_count: int
    provider_fallbacks: list[str] = Field(default_factory=list)
    stale_sections: list[str] = Field(default_factory=list)


class NarrativeSections(BaseModel):
    """The only free-text model. QA scrubs every number here against the reports.

    There is deliberately no `overall_recommendation` field.
    """

    fundamentals_summary: str
    technical_position: str
    news_and_themes: str
    peer_context: str
    risk_summary: str
    related_exposure_summary: str


class StockEvaluationReport(BaseModel):
    ticker: str
    company_name: str
    run_id: str
    generated_at: date
    fundamentals: FundamentalsReport
    technicals: TechnicalReport
    valuation: ValuationContext
    news: NewsSentimentReport
    peers: PeerComparisonReport
    risks: RiskFlagReport
    related_exposure: RelatedExposureReport
    scores: SubScores
    coverage: Coverage
    narrative: NarrativeSections
    qa_violations: list[str] = Field(default_factory=list)
    disclaimer: str = "For informational purposes only. Not financial advice."
