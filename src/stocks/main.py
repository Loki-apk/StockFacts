"""Entry point. Wires the pipeline end to end:

    resolve -> fetch layer -> LLM layer -> score -> assemble+QA -> render -> store+diff

Usage:
    crewai run                                    # prompts for a ticker (or STOCKS_TICKER)
    python -m stocks.main AAPL
    python -m stocks.main AAPL --format html --out report.html
    python -m stocks.main AAPL --strict           # exit non-zero on QA violations
    STOCKS_SKIP_LLM=1 python -m stocks.main AAPL   # deterministic layer only
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

from stocks.agents import run_llm_layer
from stocks.assemble import build_report
from stocks.fetch import run_fetch_layer
from stocks.fetch.resolve import resolve
from stocks.models.reports import StockEvaluationReport
from stocks.report.plain import render_plain
from stocks.report.render import render_html, render_markdown
from stocks.scoring import compute_sub_scores
from stocks.store import diff_reports, latest_report, save_report


def evaluate(
    ticker: str, *, use_llm: bool | None = None
) -> tuple[StockEvaluationReport, list[str] | None]:
    """Run the whole pipeline. `use_llm`: True forces the crew, False skips it,
    None follows the environment (STOCKS_SKIP_LLM / whether a key is set)."""
    ctx = resolve(ticker)
    run_fetch_layer(ctx)
    if use_llm is False:
        from stocks.agents.crew import _fallback_reports

        _fallback_reports(ctx)
    else:
        run_llm_layer(ctx)
    compute_sub_scores(ctx)
    report = build_report(ctx)

    prev = latest_report(ticker, before_run_id=report.run_id)
    changelog = diff_reports(prev, report) if prev else None
    save_report(report)
    return report, changelog


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stocks")
    parser.add_argument("ticker")
    parser.add_argument(
        "--format", choices=["md", "html", "json", "plain"], default="md",
        help="'plain' = plain-language explanation anyone can follow",
    )
    parser.add_argument("--out")
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero if the QA trace check finds violations")
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="show pipeline logs and stream each CrewAI agent's thinking",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.verbose:
        os.environ["STOCKS_CREW_VERBOSE"] = "1"

    report, changelog = evaluate(args.ticker)

    if args.format == "json":
        rendered = report.model_dump_json(indent=2)
    elif args.format == "html":
        rendered = render_html(report, changelog)
    elif args.format == "plain":
        rendered = render_plain(report, changelog)
    else:
        rendered = render_markdown(report, changelog)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(rendered)
        print(f"wrote {args.out}")
    else:
        print(rendered)

    if report.qa_violations:
        print(f"\n{len(report.qa_violations)} QA violation(s):", file=sys.stderr)
        for v in report.qa_violations:
            print(f"  - {v}", file=sys.stderr)
        if args.strict:
            return 1
    return 0


def _resolve_ticker() -> str:
    """Ticker from argv, else $STOCKS_TICKER, else an interactive prompt.

    This is the ``crewai run`` entry path — that CLI passes no arguments.
    """
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args:
        return args[0]
    if os.environ.get("STOCKS_TICKER"):
        return os.environ["STOCKS_TICKER"]
    try:
        entered = input("Ticker: ").strip()
    except EOFError:
        entered = ""
    if not entered:
        raise SystemExit("no ticker given (arg, $STOCKS_TICKER, or prompt)")
    return entered


def run() -> None:
    """``crewai run`` / ``uv run run_crew`` entry point."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    report, changelog = evaluate(_resolve_ticker())
    rendered = render_markdown(report, changelog)
    with open("report.md", "w", encoding="utf-8") as fh:
        fh.write(rendered)
    print(rendered)
    print("\nwrote report.md")
    if report.qa_violations:
        print(f"\n{len(report.qa_violations)} QA violation(s) — see the report header.",
              file=sys.stderr)


# crewai/uv script aliases
run_crew = run
kickoff = run


if __name__ == "__main__":
    raise SystemExit(main())
