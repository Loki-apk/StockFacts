# stocks — stock evaluation system

Given a ticker, produce a structured, fully-sourced evaluation — fundamentals,
technical position, news themes, peer context, risk flags — so a person can make
their own decision. **No price prediction. No buy/sell language. Every number
traces to a fetched data point.**

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design and rationale.

## Pipeline

```
resolve → deterministic fetch/compute → 3 LLM agents → score → assemble + QA → render → store + diff
```

The LLM touches language, never arithmetic. Skip it entirely
(`STOCKS_SKIP_LLM=1`) and you still get a valid, terser report.

## Setup

```bash
uv sync --extra dev --extra web    # or: pip install -e ".[dev,web]"
cp .env.example .env               # add API keys (see below)
```

## Web UI (local)

```bash
stocks-web                    # http://127.0.0.1:8000
stocks-web --port 9000 --reload
```

Enter a ticker, get **Simple** / **Full** / **JSON** views, score bars, QA flags,
and a "since last run" diff. Toggle *Full analysis* off for a near-instant
deterministic report (no AI). Previously evaluated tickers show as quick chips.

Environment (`.env`):

| var | purpose |
|---|---|
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | LLM layer |
| `NEWS_LLM`, `RISK_LLM`, `SYNTH_LLM` | model per agent (cheap for news/risk, strong for synth) |
| `FMP_API_KEY` / `FINNHUB_API_KEY` | fallback market data + news (optional) |
| `SEC_USER_AGENT` | `"your name your@email"` — required by SEC EDGAR |

## Run

```bash
crewai run                                     # prompts for a ticker
STOCKS_TICKER=AAPL crewai run                   # or pass it via env
# writes report.md

python -m stocks.main AAPL                       # full CLI
python -m stocks.main AAPL --format plain        # plain-language, no jargon
python -m stocks.main AAPL --format html --out report.html
python -m stocks.main AAPL -v                    # stream each CrewAI agent's thinking
python -m stocks.main AAPL --strict             # non-zero exit on QA violations
STOCKS_SKIP_LLM=1 python -m stocks.main AAPL     # deterministic layer only
```

`--format` is one of `md` (default), `plain`, `html`, `json`. `plain` translates
the same numbers into lemonade-stand language — deterministic, adds no opinion,
still never says whether to buy. `crewai run` respects `STOCKS_CREW_VERBOSE=1`.

## Test

```bash
pytest                       # offline — providers are replaced with fixtures
```

## Layout

| path | role |
|---|---|
| `src/stocks/models/` | Pydantic contracts (lock these first) |
| `src/stocks/providers/` | raw data access behind a protocol (yfinance + fallback, EDGAR, news) |
| `src/stocks/fetch/` | deterministic fetch + compute → one typed report each |
| `src/stocks/scoring/` | deterministic sub-scores |
| `src/stocks/qa/` | numeric traceability + banned-term scan |
| `src/stocks/agents/` | the 3-agent CrewAI layer |
| `src/stocks/assemble.py` | compose final report + coverage + QA |
| `src/stocks/store/` | persist runs, diff against the previous one |
| `src/stocks/report/` | JSON → Markdown / HTML / plain language |
| `src/stocks/web/` | FastAPI local UI over the same pipeline |

## Status

Working end to end. Deterministic layer: ticker resolve, fundamentals (+ interest
coverage), technicals (SMA/RSI/beta/vol), valuation context (self-relative P/E
band — coarse on the yfinance free tier), peer comparison, risk signals (earnings,
short interest, insider clusters, analyst downgrades, filing risk factors), news,
SEC EDGAR filings, deterministic scoring, QA trace-check, diff vs. previous run.
LLM layer: 3 CrewAI agents (news/theme, risk narrative, synthesizer) with a
deterministic fallback when no key is set or a call fails.

FMP (`/stable` API, free tier) is wired and tested: it's the yfinance fallback,
supplies beta directly, and resolves peers (`stock-peers`). `earnings` and
`insider-trading` aren't on the free tier and degrade gracefully. The P/E band
would still sharpen with a paid quarterly-fundamentals feed.

The old `crewai`-declarative scaffold is parked in `legacy_crew_jsonc/`.
