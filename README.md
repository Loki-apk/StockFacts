# stocks

A ticker-in, sourced-evaluation-out system. Give it a symbol; it returns a
structured report — fundamentals, technical position, news themes, peer
context, risk flags — so a person can make their own decision.

**No price prediction. No buy/sell/hold language. Every number in the output
traces to a fetched data point.** These are enforced in code, not just prompted
for — see [`src/stocks/qa/trace_check.py`](src/stocks/qa/trace_check.py).

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design and rationale.

## How it works

```
resolve → deterministic fetch/compute → 3 LLM agents → score → assemble + QA → render → store + diff
```

The deterministic layer does all arithmetic (fundamentals, technicals, scoring,
QA). Three CrewAI agents touch only language — news/theme summarization, risk
narrative, and final synthesis — and are constrained to restate numbers that
already exist in a typed sub-report. Skip the LLM step entirely
(`STOCKS_SKIP_LLM=1`) and you still get a valid, terser report.

## Quickstart

```bash
uv sync --extra dev --extra web    # or: pip install -e ".[dev,web]"
cp .env.example .env               # add API keys (see Configuration below)

python -m stocks.main AAPL         # print a report to stdout
stocks-web                         # or run the web UI: http://127.0.0.1:8000
```

## Configuration

Everything lives in `.env` (gitignored; template in `.env.example`).

| var | purpose |
|---|---|
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `MODEL` | LLM layer — any OpenAI-compatible endpoint |
| `NEWS_LLM`, `RISK_LLM`, `SYNTH_LLM` | optional per-agent model override (cheap for news/risk, strong for synth) |
| `FMP_API_KEY` | Financial Modeling Prep — yfinance fallback + peer resolver (optional) |
| `FINNHUB_API_KEY` | news fallback (optional) |
| `SEC_USER_AGENT` | `"your name your@email"` — required by SEC EDGAR |
| `STOCKS_PRIMARY=fmp` | make FMP the primary market-data provider instead of yfinance |
| `STOCKS_SKIP_LLM=1` | deterministic layer only, no LLM calls |
| `STOCKS_LLM_NO_THINK` | force-disable a reasoning model's hidden thinking (auto-on for Qwen) |

yfinance needs no key and is the default primary provider; everything else is
an optional upgrade.

## Usage

```bash
python -m stocks.main AAPL                       # markdown report
python -m stocks.main AAPL --format plain        # plain-language, no jargon
python -m stocks.main AAPL --format html --out report.html
python -m stocks.main AAPL --format json
python -m stocks.main AAPL -v                    # stream each CrewAI agent's thinking
python -m stocks.main AAPL --strict              # non-zero exit on QA violations
STOCKS_SKIP_LLM=1 python -m stocks.main AAPL     # deterministic layer only

crewai run                                       # prompts for a ticker, writes report.md
STOCKS_TICKER=AAPL crewai run                    # same, ticker via env
```

`--format` is one of `md` (default), `plain`, `html`, `json`. `plain`
translates the same numbers into lemonade-stand language — deterministic, adds
no opinion, still never says whether to buy.

### Web UI

```bash
stocks-web                    # http://127.0.0.1:8000
stocks-web --port 9000 --reload
```

Enter a ticker, get **Simple** / **Full** / **JSON** views, score bars, QA
flags, and a "since last run" diff. Toggle *Full analysis* off for a
near-instant deterministic report (no AI). Previously evaluated tickers show
as quick chips.

## Testing

```bash
pytest    # fully offline — providers are replaced with fixtures
```

## Project layout

| path | role |
|---|---|
| `src/stocks/models/` | Pydantic contracts (lock these first) |
| `src/stocks/providers/` | raw data access behind a protocol (yfinance + FMP fallback, EDGAR, news) |
| `src/stocks/fetch/` | deterministic fetch + compute → one typed report each |
| `src/stocks/scoring/` | deterministic sub-scores |
| `src/stocks/qa/` | numeric traceability + banned-term scan |
| `src/stocks/agents/` | the 3-agent CrewAI layer |
| `src/stocks/assemble.py` | compose the final report + coverage + QA |
| `src/stocks/store/` | persist runs, diff against the previous one |
| `src/stocks/report/` | JSON → Markdown / HTML / plain language |
| `src/stocks/web/` | FastAPI local UI over the same pipeline |

## Status

Working end to end. Deterministic layer: ticker resolve, fundamentals
(+ interest coverage), technicals (SMA/RSI/beta/vol), valuation context
(self-relative P/E band — coarse on the yfinance free tier), peer comparison,
risk signals (earnings, short interest, insider clusters, analyst downgrades,
filing risk factors), news, SEC EDGAR filings, deterministic scoring, QA
trace-check, diff vs. previous run. LLM layer: 3 CrewAI agents (news/theme,
risk narrative, synthesizer) with a deterministic fallback when no key is set
or a call fails.

FMP (`/stable` API, free tier) is wired and tested: it's the yfinance
fallback, supplies beta directly, and resolves peers (`stock-peers`).
`earnings` and `insider-trading` aren't on the free tier and degrade
gracefully. The P/E band would still sharpen with a paid quarterly-fundamentals
feed.

The old declarative `crew.jsonc` scaffold is parked in `legacy_crew_jsonc/`,
superseded by the code layout under `src/stocks/`.

## Disclaimer

For informational purposes only. Not financial advice.
