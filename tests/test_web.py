"""Smoke tests for the local web UI (no network — deterministic layer, providers
patched via the fixture DBs; the LLM layer is skipped by conftest)."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from stocks.web.app import app  # noqa: E402

client = TestClient(app)


def test_index_serves_html():
    r = client.get("/")
    assert r.status_code == 200
    assert "Stock Evaluation" in r.text
    assert "Full analysis" in r.text


def test_history_endpoint_shape():
    r = client.get("/api/history")
    assert r.status_code == 200
    assert "tickers" in r.json()


def test_report_payload_shape(monkeypatch):
    # stub the pipeline so the route logic is exercised without hitting providers
    import importlib

    from stocks.models.reports import StockEvaluationReport

    webapp = importlib.import_module("stocks.web.app")

    def fake_evaluate(ticker, *, use_llm=None):
        base = StockEvaluationReport.model_validate({**_MINIMAL, "ticker": ticker})
        return base, ["P/E: 20.0 -> 21.0"]

    monkeypatch.setattr(webapp, "evaluate", fake_evaluate)
    r = client.get("/api/report?ticker=test&full=false")
    assert r.status_code == 200
    body = r.json()
    assert body["ticker"] == "TEST"
    assert set(body) >= {"simple_html", "full_html", "json", "scores", "qa_violations"}
    assert body["changed_since_last_run"] == ["P/E: 20.0 -> 21.0"]


_MINIMAL = {
    "ticker": "X",
    "company_name": "X Corp",
    "run_id": "r1",
    "generated_at": "2026-09-03",
    "fundamentals": {"ticker": "X", "data_as_of": "2026-09-03"},
    "technicals": {
        "ticker": "X", "current_price": 10.0, "fifty_two_week_high": 12.0,
        "fifty_two_week_low": 8.0, "pct_off_high": -0.17, "pct_above_low": 0.25,
        "data_as_of": "2026-09-03",
    },
    "valuation": {"ticker": "X", "data_as_of": "2026-09-03"},
    "news": {"ticker": "X", "window_days": 7, "items": [], "dominant_themes": []},
    "peers": {"ticker": "X", "peers": [], "peer_selection_method": "none",
              "metric_comparison": {}},
    "risks": {"ticker": "X", "flags": []},
    "related_exposure": {"ticker": "X", "candidates_checked": 0, "exposures": []},
    "scores": {"valuation": None, "growth": None, "financial_health": None,
               "momentum": None},
    "coverage": {"fields_populated": 0, "fields_total": 12, "news_window_days": 7,
                 "news_item_count": 0},
    "narrative": {"fundamentals_summary": "", "technical_position": "",
                  "news_and_themes": "", "peer_context": "", "risk_summary": "",
                  "related_exposure_summary": ""},
}
