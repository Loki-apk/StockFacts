"""-> list[RiskFlag], the structured version before the LLM writes anything about them.

Everything here is a fixed threshold check, nothing fuzzy. The Risk Narrative
agent later writes the actual `description` and picks a final `severity`,
but it can't invent a flag that isn't already in this list.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from stockfact.cache import cached
from stockfact.models.enums import FlagType, Severity
from stockfact.models.reports import RiskFlag
from stockfact.providers import get_provider
from stockfact.run_context import RunContext

_EARNINGS_SOON_DAYS = 14
_HIGH_SHORT_PCT = 0.10
_INSIDER_WINDOW_DAYS = 120
_DOWNGRADE_WINDOW_DAYS = 45


@cached("risk_signals")
def _signals_raw(ticker: str) -> dict:
    p = get_provider()

    def _safe(fn):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    return {
        "calendar": _safe(lambda: p.calendar(ticker)),
        "insider": _safe(lambda: p.insider_transactions(ticker)),
        "upgrades_downgrades": _safe(lambda: p.upgrades_downgrades(ticker)),
        "short": _safe(lambda: p.short_interest(ticker)),
    }


def fetch_risk_signals(ctx: RunContext) -> list[RiskFlag]:
    raw = _signals_raw(ctx.ticker)
    if isinstance(raw, dict) and "error" in raw:
        ctx.fetch_errors["risk_signals"] = raw["error"]
        return []

    flags: list[RiskFlag] = []
    today = date.today()
    source = get_provider().name

    flags += _earnings_flag(raw.get("calendar"), today, source)
    flags += _short_interest_flag(raw.get("short"), today, source)
    flags += _insider_flag(raw.get("insider"), today, source)
    flags += _downgrade_flag(raw.get("upgrades_downgrades"), today, source)
    flags += _filing_flag(ctx, today)

    ctx.risk_flags_raw = flags
    return flags


def _earnings_flag(calendar, today, source) -> list[RiskFlag]:
    d = _parse_earnings_date(calendar)
    if d and today <= d <= today + timedelta(days=_EARNINGS_SOON_DAYS):
        return [
            RiskFlag(
                flag_type=FlagType.EARNINGS_UPCOMING,
                description=f"Earnings expected {d.isoformat()}",
                severity=Severity.WATCH,
                source=source,
                as_of=today,
            )
        ]
    return []


def _short_interest_flag(short, today, source) -> list[RiskFlag]:
    if not isinstance(short, dict):
        return []
    pct = short.get("short_percent_of_float")
    if pct is None or pct < _HIGH_SHORT_PCT:
        return []
    return [
        RiskFlag(
            flag_type=FlagType.HIGH_SHORT_INTEREST,
            description=f"Short interest is {pct:.1%} of float",
            severity=Severity.ELEVATED if pct >= 0.20 else Severity.WATCH,
            source=source,
            as_of=today,
        )
    ]


def _insider_flag(insider, today, source) -> list[RiskFlag]:
    if not isinstance(insider, list) or not insider:
        return []
    cutoff = today - timedelta(days=_INSIDER_WINDOW_DAYS)
    buys = sells = 0.0
    for row in insider:
        when = _coerce_date(
            row.get("Start Date") or row.get("startDate") or row.get("transactionDate")
        )
        if not when or when < cutoff:
            continue
        value = _coerce_float(row.get("Value") or row.get("value"))
        text = str(row.get("Text") or row.get("transactionType") or "").lower()
        if value is None:
            continue
        if "sale" in text or "sell" in text or "disposition" in text:
            sells += value
        elif "purchase" in text or "buy" in text or "acqui" in text:
            buys += value

    net = sells - buys
    if sells > 0 and net > 0 and sells >= max(1_000_000.0, 3 * buys):
        return [
            RiskFlag(
                flag_type=FlagType.INSIDER_SELLING,
                description=(
                    f"Net insider selling of ~${net:,.0f} over the last "
                    f"{_INSIDER_WINDOW_DAYS} days"
                ),
                severity=Severity.WATCH,
                source=source,
                as_of=today,
            )
        ]
    if buys > 0 and buys > sells:
        return [
            RiskFlag(
                flag_type=FlagType.INSIDER_BUYING,
                description=f"Net insider buying of ~${buys - sells:,.0f} recently",
                severity=Severity.INFO,
                source=source,
                as_of=today,
            )
        ]
    return []


def _downgrade_flag(rows, today, source) -> list[RiskFlag]:
    if not isinstance(rows, list) or not rows:
        return []
    cutoff = today - timedelta(days=_DOWNGRADE_WINDOW_DAYS)
    downgrades, upgrades = [], []
    for row in rows:
        when = _coerce_date(row.get("GradeDate") or row.get("gradeDate") or row.get("date"))
        if not when or when < cutoff:
            continue
        action = str(row.get("Action") or row.get("action") or "").lower()
        firm = row.get("Firm") or row.get("gradingCompany") or "an analyst"
        if action in {"down", "downgrade"} or "lower" in action:
            downgrades.append(firm)
        elif action in {"up", "upgrade"} or "raise" in action:
            upgrades.append(firm)

    out: list[RiskFlag] = []
    if downgrades:
        out.append(
            RiskFlag(
                flag_type=FlagType.ANALYST_DOWNGRADE,
                description=(
                    f"{len(downgrades)} analyst downgrade(s) in the last "
                    f"{_DOWNGRADE_WINDOW_DAYS} days ({', '.join(downgrades[:3])})"
                ),
                severity=Severity.WATCH if len(downgrades) >= 2 else Severity.INFO,
                source=source,
                as_of=today,
            )
        )
    if upgrades and not downgrades:
        out.append(
            RiskFlag(
                flag_type=FlagType.ANALYST_UPGRADE,
                description=(
                    f"{len(upgrades)} analyst upgrade(s) in the last "
                    f"{_DOWNGRADE_WINDOW_DAYS} days ({', '.join(upgrades[:3])})"
                ),
                severity=Severity.INFO,
                source=source,
                as_of=today,
            )
        )
    return out


def _filing_flag(ctx: RunContext, today) -> list[RiskFlag]:
    if not ctx.filings:
        return []
    has_1a = any(f.section == "Item 1A" for f in ctx.filings)
    if not has_1a:
        return []
    return [
        RiskFlag(
            flag_type=FlagType.FILING_RISK_FACTOR,
            description="See the latest 10-K Item 1A risk factors (excerpted for review)",
            severity=Severity.INFO,
            source="SEC EDGAR",
            as_of=today,
        )
    ]


# --- parsing helpers ---
def _parse_earnings_date(calendar) -> date | None:
    if not isinstance(calendar, dict):
        return None
    val = calendar.get("Earnings Date") or calendar.get("earningsDate")
    if isinstance(val, dict):
        val = next(iter(val.values()), None)
    if isinstance(val, (list, tuple)):
        val = val[0] if val else None
    return _coerce_date(val)


def _coerce_date(val) -> date | None:
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date):
        return val
    if val is None:
        return None
    try:
        return datetime.fromisoformat(str(val)[:19]).date()
    except ValueError:
        try:
            return date.fromisoformat(str(val)[:10])
        except ValueError:
            return None


def _coerce_float(val) -> float | None:
    try:
        return abs(float(val))
    except (TypeError, ValueError):
        return None
