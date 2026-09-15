"""Turns already-fetched numbers into 0-100 sub-scores. Pure functions,
nothing fancy — same inputs always give the same score.

If something's missing, the sub-score is just None, not a made-up guess, and
the rollup score only averages what's actually there while noting the gap.
"""

from __future__ import annotations

from stockfact.models.reports import SubScores
from stockfact.run_context import RunContext


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _scale(value: float | None, lo: float, hi: float) -> float | None:
    """Linear map [lo, hi] -> [0, 100], clamped. None passes through."""
    if value is None or hi == lo:
        return None
    return _clamp((value - lo) / (hi - lo) * 100)


def _inv_scale(value: float | None, lo: float, hi: float) -> float | None:
    s = _scale(value, lo, hi)
    return None if s is None else 100 - s


def _blend(*parts: float | None) -> float | None:
    present = [p for p in parts if p is not None]
    return sum(present) / len(present) if present else None


def _val(metric) -> float | None:
    return metric.value if metric is not None else None


def compute_sub_scores(ctx: RunContext) -> SubScores:
    f = ctx.fundamentals
    t = ctx.technicals
    v = ctx.valuation

    # --- valuation: cheap vs. own 5y range -> high score ---
    valuation = None
    if v and v.pe_percentile_5y is not None:
        valuation = 100 - v.pe_percentile_5y  # 0th pctile (cheapest) -> 100
    elif f and _val(f.pe_ratio) is not None:
        valuation = _inv_scale(_val(f.pe_ratio), lo=5, hi=45)  # crude absolute fallback

    # --- growth ---
    growth = _scale(_val(f.revenue_growth_yoy) if f else None, lo=-0.10, hi=0.40)

    # --- financial health ---
    health = _blend(
        _inv_scale(_val(f.debt_to_equity) if f else None, lo=0.0, hi=3.0),
        _scale(_val(f.interest_coverage) if f else None, lo=1.0, hi=12.0),
        _scale(_val(f.current_ratio) if f else None, lo=0.5, hi=2.5),
    )

    # --- momentum (position, not prediction) ---
    momentum = _blend(
        _scale(t.price_vs_sma200 if t else None, lo=-0.30, hi=0.30),
        _scale(100 - abs((t.rsi_14 or 50) - 50) * 2 if t and t.rsi_14 else None, 0, 100),
    )

    subs = {
        "valuation": valuation,
        "growth": growth,
        "financial_health": health,
        "momentum": momentum,
    }
    present = [s for s in subs.values() if s is not None]
    rollup = sum(present) / len(present) if present else None

    explanations = {}
    if v and v.pe_now is not None:
        explanations["valuation"] = (
            f"PE {v.pe_now:.1f} vs 5y median {v.pe_5y_median}"
            f" (pctile {v.pe_percentile_5y})"
        )
    if f and _val(f.revenue_growth_yoy) is not None:
        explanations["growth"] = f"revenue YoY {_val(f.revenue_growth_yoy):+.1%}"
    if f and _val(f.debt_to_equity) is not None:
        explanations["financial_health"] = f"D/E {_val(f.debt_to_equity):.2f}"
    if t and t.price_vs_sma200 is not None:
        explanations["momentum"] = (
            f"{t.price_vs_sma200:+.1%} vs SMA200, RSI {t.rsi_14 and round(t.rsi_14)}"
        )
    if len(present) < 4:
        explanations["_coverage"] = f"{len(present)}/4 sub-scores computed"

    scores = SubScores(
        valuation=None if valuation is None else round(valuation, 1),
        growth=None if growth is None else round(growth, 1),
        financial_health=None if health is None else round(health, 1),
        momentum=None if momentum is None else round(momentum, 1),
        explanations=explanations,
        rollup=None if rollup is None else round(rollup, 1),
    )
    ctx.scores = scores
    return scores
