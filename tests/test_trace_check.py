"""QA must catch fabricated numbers and advice language in the narrative."""

from __future__ import annotations

from stocks.assemble import build_report
from stocks.qa import trace_check
from stocks.scoring import compute_sub_scores


def _report(ctx):
    compute_sub_scores(ctx)
    return build_report(ctx)


def test_clean_narrative_passes(sample_ctx):
    report = _report(sample_ctx)
    # sample narrative only cites numbers present in the sub-reports
    assert report.qa_violations == []


def test_fabricated_number_is_flagged(sample_ctx):
    sample_ctx.narrative.fundamentals_summary = (
        "Revenue will grow 47.3% next year and the stock is worth 250.00."
    )
    report = _report(sample_ctx)
    joined = " ".join(report.qa_violations)
    assert "untraceable number" in joined
    assert "47.3" in joined or "250.00" in joined


def test_banned_terms_are_flagged(sample_ctx):
    sample_ctx.narrative.technical_position = (
        "This looks undervalued and now is a good time to buy."
    )
    violations = trace_check(_report(sample_ctx))
    joined = " ".join(violations).lower()
    assert "banned term" in joined
    assert "buy" in joined or "undervalued" in joined or "good time to" in joined


def test_dates_and_window_labels_are_not_flagged(sample_ctx):
    sample_ctx.narrative.technical_position = (
        "As of 2026-09-03 the price was 190.00, above its 50-day and 200-day "
        "simple moving averages over the last 30 days."
    )
    report = _report(sample_ctx)
    assert not any("untraceable" in v for v in report.qa_violations)


def test_numbers_from_headlines_and_risk_text_trace(sample_ctx):
    from datetime import date

    from stocks.models.enums import FlagType, Sentiment, Severity, Theme
    from stocks.models.reports import NewsSentimentItem, RiskFlag

    sample_ctx.news_report.items = [
        NewsSentimentItem(
            headline="Company stock is up 26% in 2026 after 105 job cuts",
            source="Wire",
            url="https://example.com/x",
            published_at=date(2026, 9, 1),
            theme=Theme.OTHER,
            sentiment=Sentiment.NEUTRAL,
            rationale="price move noted",
        )
    ]
    sample_ctx.risk_report.flags = [
        RiskFlag(
            flag_type=FlagType.INSIDER_SELLING,
            description="Net insider selling of ~$156,824,633 over the last 120 days",
            severity=Severity.WATCH,
            source="yfinance",
        )
    ]
    sample_ctx.narrative.news_and_themes = "Shares rose 26% in 2026; the firm cut 105 jobs."
    sample_ctx.narrative.risk_summary = (
        "Insiders sold about $156,824,633 net over the last 120 days."
    )
    report = _report(sample_ctx)
    assert not any("untraceable" in v for v in report.qa_violations)


def test_price_target_allowed_only_in_news(sample_ctx):
    sample_ctx.narrative.news_and_themes = "Bank of America reset its price target."
    sample_ctx.narrative.risk_summary = "Analysts published a new price target."
    violations = trace_check(_report(sample_ctx))
    assert not any("news_and_themes: banned term 'price target'" in v for v in violations)
    assert any("risk_summary: banned term 'price target'" in v for v in violations)


def test_rounded_match_is_allowed(sample_ctx):
    # 24.0 in the reports, "24" in prose -> fine
    sample_ctx.narrative.peer_context = "PE of 24 is near the peer average."
    report = _report(sample_ctx)
    assert not any("untraceable" in v for v in report.qa_violations)
