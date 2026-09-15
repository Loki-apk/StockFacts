"""Scoring is deterministic, bounded 0-100, and degrades to None on missing input."""

from __future__ import annotations

from stockfact.scoring import compute_sub_scores


def test_scores_bounded_and_deterministic(sample_ctx):
    a = compute_sub_scores(sample_ctx)
    b = compute_sub_scores(sample_ctx)
    assert a.model_dump() == b.model_dump()
    for name in ("valuation", "growth", "financial_health", "momentum", "rollup"):
        val = getattr(a, name)
        assert val is None or 0.0 <= val <= 100.0


def test_cheap_vs_history_scores_higher_on_valuation(sample_ctx):
    sample_ctx.valuation.pe_percentile_5y = 10.0  # near its 5y low
    cheap = compute_sub_scores(sample_ctx).valuation
    sample_ctx.valuation.pe_percentile_5y = 90.0  # near its 5y high
    rich = compute_sub_scores(sample_ctx).valuation
    assert cheap > rich


def test_missing_inputs_give_none_not_crash(sample_ctx):
    sample_ctx.fundamentals.revenue_growth_yoy = None
    sample_ctx.fundamentals.debt_to_equity = None
    sample_ctx.fundamentals.current_ratio = None
    sample_ctx.fundamentals.interest_coverage = None
    scores = compute_sub_scores(sample_ctx)
    assert scores.growth is None
    assert scores.financial_health is None
    assert "_coverage" in scores.explanations


def test_rollup_averages_only_present_subscores(sample_ctx):
    sample_ctx.valuation.pe_percentile_5y = None
    sample_ctx.fundamentals.pe_ratio = None
    scores = compute_sub_scores(sample_ctx)
    assert scores.valuation is None
    assert scores.rollup is not None  # still computed from the other three
