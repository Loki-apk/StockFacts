"""The last line of defence against hallucination. Deterministic — no LLM.

1. Every significant number in the narrative must trace to a number the pipeline
   already surfaced — a typed field, or text the deterministic layer wrote
   (risk-flag descriptions, news headlines, valuation notes).
2. No advice / prediction language anywhere.
3. A run with violations never renders silently (see report/render.py).
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from stocks.models.reports import NarrativeSections, StockEvaluationReport

_NUM_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?%?")

# Stripped from narrative text before number scanning — these are labels, not claims.
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_WINDOW_LABEL = re.compile(
    r"\b\d+[-\s]?(?:day|week|month|year|yr|hour|quarter)s?\b", re.IGNORECASE
)
_ORDINAL_WINDOW = re.compile(r"\b\d+(?:st|nd|rd|th)\b", re.IGNORECASE)

# Advice / prediction language. Banned in every narrative section.
BANNED_TERMS = re.compile(
    r"\b("
    r"buy|sell|hold|accumulate|trim|divest|"
    r"overweight|underweight|"
    r"undervalued|overvalued|cheap|expensive|"
    r"should|ought to|recommend|advise|"
    r"will (?:rise|fall|climb|drop|reach)|"
    r"expect(?:ed)? to (?:rise|fall|reach|hit)|"
    r"good time to|best time to"
    r")\b",
    re.IGNORECASE,
)

# "price target" / "target price": a red flag when the model volunteers one, but
# legitimate when `news_and_themes` reports that an analyst set/changed one.
_ANALYST_TARGET = re.compile(r"\b(?:price target|target price)\b", re.IGNORECASE)

# tolerance for rounding when matching a narrative number to a source number
_REL_TOL = 0.02
_MIN_TOL = 0.01


def _to_float(token: str) -> float | None:
    s = token.replace("$", "").replace(",", "").strip().rstrip("%")
    try:
        return float(s)
    except ValueError:
        return None


def _candidates(token: str) -> list[float]:
    """Plausible numeric readings of a prose token. A '%' token can mean either
    the raw value (percentile 89.6) or the fraction (0.896)."""
    raw = _to_float(token)
    if raw is None:
        return []
    if token.strip().endswith("%"):
        return [raw, raw / 100]
    return [raw]


def _is_significant(token: str) -> bool:
    """A token worth tracing: a decimal, a percentage, a currency amount, a
    thousands-separated number, or an integer >= 100. Bare small integers
    ('5y', '3 items') are boilerplate."""
    if any(c in token for c in ".$,%"):
        return True
    val = _to_float(token)
    return val is not None and abs(val) >= 100


def _numbers_in_text(text: str) -> set[float]:
    out: set[float] = set()
    for tok in _NUM_RE.findall(text):
        for v in _candidates(tok):
            out.add(v)
            out.add(round(v, 2))
    return out


def collect_all_numbers(*models: BaseModel) -> set[float]:
    """Every numeric leaf in the given models, plus numbers embedded in their
    string fields, plus common derived forms (ratio<->percent)."""
    out: set[float] = set()

    def walk(obj) -> None:
        if isinstance(obj, bool):
            return
        if isinstance(obj, (int, float)):
            out.add(float(obj))
            out.add(round(float(obj), 2))
            if abs(obj) < 1:
                out.add(round(float(obj) * 100, 2))
            return
        if isinstance(obj, str):
            out.update(_numbers_in_text(obj))
            return
        if isinstance(obj, BaseModel):
            for v in obj.__dict__.values():
                walk(v)
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, (list, tuple, set)):
            for v in obj:
                walk(v)

    for m in models:
        walk(m)
    return out


def _matches_known(token: str, known: set[float]) -> bool:
    for val in _candidates(token):
        for k in known:
            tol = max(_MIN_TOL, abs(k) * _REL_TOL)
            # abs-compare too: prose often flips the sign of a stored delta
            # ("9.5% off the high" vs a stored pct_off_high of -0.095)
            if abs(val - k) <= tol or abs(abs(val) - abs(k)) <= tol:
                return True
    return False


def _scannable(text: str) -> str:
    text = _ISO_DATE.sub(" ", text)
    text = _WINDOW_LABEL.sub(" ", text)
    text = _ORDINAL_WINDOW.sub(" ", text)
    return text


def trace_check(report: StockEvaluationReport) -> list[str]:
    known = collect_all_numbers(
        report.fundamentals,
        report.technicals,
        report.valuation,
        report.peers,
        report.risks,
        report.related_exposure,
        report.scores,
        report.news,
    )
    violations: list[str] = []
    narrative: NarrativeSections = report.narrative
    for section, text in narrative.__dict__.items():
        if not isinstance(text, str):
            continue
        scannable = _scannable(text)
        for tok in _NUM_RE.findall(scannable):
            if _is_significant(tok) and not _matches_known(tok, known):
                violations.append(f"{section}: untraceable number {tok!r}")
        for m in BANNED_TERMS.finditer(text):
            violations.append(f"{section}: banned term {m.group(0)!r}")
        if section != "news_and_themes":
            for m in _ANALYST_TARGET.finditer(text):
                violations.append(f"{section}: banned term {m.group(0)!r}")
    return violations
