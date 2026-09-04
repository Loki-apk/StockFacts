"""Step 4-5: compose the final report, compute coverage, run the QA trace check."""

from __future__ import annotations

from stocks.models.reports import (
    Coverage,
    FundamentalsReport,
    NarrativeSections,
    RelatedExposureReport,
    StockEvaluationReport,
)
from stocks.qa import trace_check
from stocks.run_context import RunContext


def _coverage(ctx: RunContext) -> Coverage:
    f = ctx.fundamentals
    tracked = [
        "pe_ratio", "forward_pe", "ev_to_ebitda", "revenue_growth_yoy",
        "eps_growth_yoy", "gross_margin", "profit_margin", "fcf_margin",
        "debt_to_equity", "interest_coverage", "current_ratio", "return_on_equity",
    ]
    populated = sum(1 for name in tracked if f and getattr(f, name) is not None)
    return Coverage(
        fields_populated=populated,
        fields_total=len(tracked),
        news_window_days=ctx.news_report.window_days if ctx.news_report else 0,
        news_item_count=len(ctx.news_report.items) if ctx.news_report else 0,
        provider_fallbacks=sorted(set(ctx.provider_fallbacks)),
        stale_sections=sorted(ctx.fetch_errors.keys()),
    )


def build_report(ctx: RunContext) -> StockEvaluationReport:
    if not ctx.deterministic_layer_usable():
        raise RuntimeError(
            f"insufficient data for {ctx.ticker}: {ctx.fetch_errors or 'no core reports'}"
        )

    narrative = ctx.narrative or NarrativeSections(
        fundamentals_summary="", technical_position="", news_and_themes="",
        peer_context="", risk_summary="", related_exposure_summary="",
    )

    report = StockEvaluationReport(
        ticker=ctx.ticker,
        company_name=ctx.company_name,
        run_id=ctx.run_id,
        generated_at=ctx.generated_at,
        fundamentals=ctx.fundamentals or FundamentalsReport(
            ticker=ctx.ticker, data_as_of=ctx.generated_at
        ),
        technicals=ctx.technicals,
        valuation=ctx.valuation,
        news=ctx.news_report,
        peers=ctx.peers,
        risks=ctx.risk_report,
        related_exposure=ctx.related_exposure or RelatedExposureReport(
            ticker=ctx.ticker, candidates_checked=0, exposures=[]
        ),
        scores=ctx.scores,
        coverage=_coverage(ctx),
        narrative=narrative,
        qa_violations=[],
    )

    report.qa_violations = trace_check(report)
    ctx.qa_violations = report.qa_violations
    return report
