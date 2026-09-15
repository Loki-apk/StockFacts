"""store/runs.py: a stored run from an older schema must never crash a new
evaluation — it's skipped, not fatal (see related_exposure's field addition,
which is exactly the kind of schema change this guards against)."""

from __future__ import annotations

import json

from stockfact.store.runs import _conn, _db_path, latest_report


def test_db_path_reflects_env_var_set_after_import(monkeypatch, tmp_path):
    # Regression: _DB_PATH used to be resolved once at import time, so
    # monkeypatch.setenv (as the autouse _isolated_dbs fixture does) had no
    # effect and tests were silently writing into the real cache/runs.db.
    target = tmp_path / "isolated.db"
    monkeypatch.setenv("STOCKFACT_RUNS_DB", str(target))
    assert _db_path() == target


def test_legacy_schema_row_is_skipped_not_raised(monkeypatch, tmp_path):
    monkeypatch.setenv("STOCKFACT_RUNS_DB", str(tmp_path / "runs.db"))

    conn = _conn()
    conn.execute(
        "INSERT INTO runs (run_id, ticker, generated_at, payload) VALUES (?, ?, ?, ?)",
        ("old1", "TEST", "2026-01-01", json.dumps({"ticker": "TEST"})),  # missing required fields
    )
    conn.commit()
    conn.close()

    assert latest_report("TEST") is None
