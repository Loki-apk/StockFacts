"""Typed contracts for every stage of the pipeline.

Lock these first (build order step 1). Everything downstream depends on them.
"""

from stocks.models.enums import FlagType, Sentiment, Severity, Theme
from stocks.models.reports import (
    Coverage,
    FilingExcerpt,
    FundamentalsReport,
    MetricWithContext,
    NarrativeSections,
    NewsSentimentItem,
    NewsSentimentReport,
    PeerComparisonReport,
    RawNewsItem,
    RelatedExposure,
    RelatedExposureReport,
    RiskFlag,
    RiskFlagReport,
    StockEvaluationReport,
    SubScores,
    TechnicalReport,
    ValuationContext,
)

__all__ = [
    "FlagType",
    "Sentiment",
    "Severity",
    "Theme",
    "Coverage",
    "FilingExcerpt",
    "FundamentalsReport",
    "MetricWithContext",
    "NarrativeSections",
    "NewsSentimentItem",
    "NewsSentimentReport",
    "PeerComparisonReport",
    "RawNewsItem",
    "RelatedExposure",
    "RelatedExposureReport",
    "RiskFlag",
    "RiskFlagReport",
    "StockEvaluationReport",
    "SubScores",
    "TechnicalReport",
    "ValuationContext",
]
