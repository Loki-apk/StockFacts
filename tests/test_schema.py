"""Every report model validates; the assembled report round-trips through JSON."""

from __future__ import annotations

from stocks.assemble import build_report
from stocks.models.reports import StockEvaluationReport
from stocks.scoring import compute_sub_scores


def test_build_report_validates(sample_ctx):
    compute_sub_scores(sample_ctx)
    report = build_report(sample_ctx)
    assert isinstance(report, StockEvaluationReport)

    dumped = report.model_dump_json()
    restored = StockEvaluationReport.model_validate_json(dumped)
    assert restored.ticker == "TEST"
    assert restored.disclaimer == "For informational purposes only. Not financial advice."


def test_no_recommendation_field():
    # The narrative model must not grow an advice field.
    from stocks.models.reports import NarrativeSections

    assert "overall_recommendation" not in NarrativeSections.model_fields
    assert "recommendation" not in NarrativeSections.model_fields


def test_plain_render_is_advice_free_and_handles_gaps(sample_ctx):
    from stocks.assemble import build_report
    from stocks.qa.trace_check import BANNED_TERMS
    from stocks.report.plain import render_plain
    from stocks.scoring import compute_sub_scores

    # knock out several metrics — the plain renderer must not crash
    sample_ctx.fundamentals.pe_ratio = None
    sample_ctx.fundamentals.return_on_equity = None
    sample_ctx.valuation.pe_now = None
    compute_sub_scores(sample_ctx)
    text = render_plain(build_report(sample_ctx))

    assert "explained simply" in text
    assert "Not financial advice" in text
    # advice/opinion language may appear only inside the two disclaimer strings
    from stocks.report.plain import _KID_DISCLAIMER

    body = text.replace(_KID_DISCLAIMER, "").replace(
        "For informational purposes only. Not financial advice.", ""
    )
    assert not BANNED_TERMS.search(body)


def test_fallback_passes_news_through(sample_ctx):
    from datetime import date

    from stocks.agents.crew import _fallback_reports
    from stocks.models.reports import RawNewsItem

    sample_ctx.raw_news = [
        RawNewsItem(
            headline="Company reports Q3 results",
            source="Reuters",
            url="https://example.com/a",
            published_at=date(2026, 9, 1),
        )
    ]
    sample_ctx.news_report = None
    _fallback_reports(sample_ctx)
    assert len(sample_ctx.news_report.items) == 1
    assert str(sample_ctx.news_report.items[0].url) == "https://example.com/a"
    assert sample_ctx.news_report.items[0].theme == "other"


def test_insufficient_data_raises(sample_ctx):
    sample_ctx.fundamentals = None
    from stocks.assemble import build_report

    try:
        build_report(sample_ctx)
    except RuntimeError as exc:
        assert "insufficient data" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected RuntimeError")
