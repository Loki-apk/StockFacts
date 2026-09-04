"""FastAPI app: enter a ticker, get the evaluation as Simple / Full / JSON.

Everything runs in-process against the same pipeline as the CLI. Evaluations are
slow (the LLM layer especially), so requests run in a worker thread and the page
shows a spinner. `?full=false` uses the deterministic layer only — near instant.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

from stocks.main import evaluate
from stocks.report.plain import render_plain
from stocks.report.render import render_markdown
from stocks.store import latest_report

_INDEX = (Path(__file__).parent / "templates" / "index.html").read_text()

app = FastAPI(title="Stock Evaluation")


def _md_to_html(md: str) -> str:
    try:
        import markdown as _md

        return _md.markdown(md, extensions=["tables", "sane_lists"])
    except ImportError:  # pragma: no cover
        return f"<pre>{md}</pre>"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _INDEX


@app.get("/api/history")
def history() -> dict[str, Any]:
    """Tickers already evaluated (from the runs DB), most recent first."""
    import sqlite3

    from stocks.store.runs import _db_path

    db_path = _db_path()
    if not db_path.exists():
        return {"tickers": []}
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT ticker, MAX(generated_at) g FROM runs GROUP BY ticker ORDER BY g DESC"
        ).fetchall()
    finally:
        conn.close()
    return {"tickers": [{"ticker": t, "as_of": g} for t, g in rows]}


@app.get("/api/report")
async def report(
    ticker: str = Query(..., min_length=1, max_length=12),
    full: bool = True,
    cached_ok: bool = False,
) -> JSONResponse:
    ticker = ticker.strip().upper()

    if cached_ok:
        prev = await run_in_threadpool(latest_report, ticker)
        if prev is not None:
            return JSONResponse(_payload(prev, None, from_cache=True))

    try:
        rpt, diff = await run_in_threadpool(evaluate, ticker, use_llm=full)
    except ValueError as exc:  # unresolvable ticker
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    return JSONResponse(_payload(rpt, diff, from_cache=False))


def _payload(rpt, diff, *, from_cache: bool) -> dict[str, Any]:
    return {
        "ticker": rpt.ticker,
        "company_name": rpt.company_name,
        "generated_at": str(rpt.generated_at),
        "run_id": rpt.run_id,
        "from_cache": from_cache,
        "qa_violations": rpt.qa_violations,
        "scores": rpt.scores.model_dump(mode="json"),
        "coverage": rpt.coverage.model_dump(mode="json"),
        "changed_since_last_run": diff or [],
        "simple_html": _md_to_html(render_plain(rpt, diff)),
        "full_html": _md_to_html(render_markdown(rpt, diff)),
        "json": rpt.model_dump(mode="json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="stocks-web")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    import uvicorn

    print(f"\n  Stock Evaluation UI → http://{args.host}:{args.port}\n")
    uvicorn.run(
        "stocks.web.app:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
