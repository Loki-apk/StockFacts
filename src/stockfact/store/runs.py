"""Saves each evaluation and compares it against the last run for the same ticker.

Honestly, this diff is probably the single most useful feature if you keep
coming back to check the same stock — it only tells you what actually
changed since last time instead of making you reread the whole report.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path

from pydantic import ValidationError

from stockfact.models.reports import StockEvaluationReport

log = logging.getLogger(__name__)

def _db_path() -> Path:
    # Resolved per call, not at import time — STOCKFACT_RUNS_DB must be settable
    # after this module is first imported (tests isolate the DB this way).
    return Path(os.environ.get("STOCKFACT_RUNS_DB", "cache/runs.db"))


def _conn() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS runs ("
        "run_id TEXT PRIMARY KEY, ticker TEXT NOT NULL, "
        "generated_at TEXT NOT NULL, payload TEXT NOT NULL)"
    )
    return conn


def save_report(report: StockEvaluationReport) -> None:
    conn = _conn()
    try:
        conn.execute(
            "REPLACE INTO runs (run_id, ticker, generated_at, payload) VALUES (?, ?, ?, ?)",
            (
                report.run_id,
                report.ticker,
                report.generated_at.isoformat(),
                report.model_dump_json(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def latest_report(ticker: str, before_run_id: str | None = None) -> StockEvaluationReport | None:
    conn = _conn()
    try:
        rows = conn.execute(
            "SELECT run_id, payload FROM runs WHERE ticker = ? "
            "ORDER BY generated_at DESC, rowid DESC",
            (ticker.upper(),),
        ).fetchall()
    finally:
        conn.close()
    for run_id, payload in rows:
        if before_run_id and run_id == before_run_id:
            continue
        try:
            return StockEvaluationReport.model_validate_json(payload)
        except ValidationError as exc:
            # A run stored under an older schema (a field was added/renamed
            # since). Skip it rather than let a stale row crash the whole
            # evaluation — no diff for that pair, not a failure.
            log.warning("skipping unreadable stored run %s: %s", run_id, exc)
            continue
    return None


def diff_reports(
    prev: StockEvaluationReport, curr: StockEvaluationReport
) -> list[str]:
    """Human-readable changelog. Deterministic string diffing only."""
    lines: list[str] = []

    def num_delta(label: str, a: float | None, b: float | None, fmt: str = "{:.2f}") -> None:
        if a is None or b is None or a == b:
            return
        lines.append(f"{label}: {fmt.format(a)} -> {fmt.format(b)}")

    pf, cf = prev.fundamentals, curr.fundamentals
    for field in ("pe_ratio", "forward_pe", "revenue_growth_yoy", "profit_margin"):
        pm, cm = getattr(pf, field), getattr(cf, field)
        num_delta(field, pm.value if pm else None, cm.value if cm else None)

    num_delta("price", prev.technicals.current_price, curr.technicals.current_price)

    prev_flags = {(f.flag_type, f.severity) for f in prev.risks.flags}
    for f in curr.risks.flags:
        if (f.flag_type, f.severity) not in prev_flags:
            lines.append(f"new risk flag: {f.flag_type} ({f.severity}) — {f.description}")
    curr_flags = {f.flag_type for f in curr.risks.flags}
    for f in prev.risks.flags:
        if f.flag_type not in curr_flags:
            lines.append(f"cleared risk flag: {f.flag_type}")

    prev_exposures = {e.related_ticker for e in prev.related_exposure.exposures}
    for e in curr.related_exposure.exposures:
        if e.related_ticker not in prev_exposures:
            lines.append(f"new related exposure: {e.related_ticker} ({e.relationship_type})")

    prev_themes = set(prev.news.dominant_themes)
    for th in curr.news.dominant_themes:
        if th not in prev_themes:
            lines.append(f"news theme appeared: {th}")

    if prev.scores.rollup is not None and curr.scores.rollup is not None:
        num_delta("composite (heuristic)", prev.scores.rollup, curr.scores.rollup, "{:.1f}")

    return lines
