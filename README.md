# StockFact

I built this to fix one specific thing that bugs me about basically every "stock
analysis" tool out there: they all eventually try to tell you what to do. Buy
this, sell that, price target $X. StockFact doesn't. You give it a ticker, it
hands back a report — fundamentals, technicals, news, peer comparison, risk
flags — and every number in that report traces back to somewhere it was
actually fetched from. No predictions, no "you should," nothing made up.

That's not just something I wrote in this README and hoped for — it's checked
in code, on every single run, before anything gets shown to you. See
[`src/stockfact/qa/trace_check.py`](src/stockfact/qa/trace_check.py) if you
want to see how.

The full technical writeup — why it's built this way, and what changed along
the way — is in [ARCHITECTURE.md](ARCHITECTURE.md).

## How a run actually works

```
resolve → deterministic fetch/compute → 3 LLM agents → score → assemble + QA → render → store + diff
```

Almost all of the real work — fundamentals, technicals, scoring, the QA check
— is plain Python. No LLM involved, because there's no good reason to trust an
LLM with arithmetic it might just get wrong. The only place an LLM touches
this project is turning numbers that already exist into readable sentences:
summarizing news themes, writing up risk narratives, and pulling everything
together into a final write-up. Skip that step entirely
(`STOCKFACT_SKIP_LLM=1`) and you still get a full, valid report — just a
plainer one.

## Getting it running

```bash
uv sync --extra dev --extra web    # or: pip install -e ".[dev,web]"
cp .env.example .env               # fill in whatever keys you have, see below

python -m stockfact.main AAPL      # prints a report to your terminal
stockfact-web                      # or run the local web UI: http://127.0.0.1:8000
```

## What goes in `.env`

Nothing here is required — yfinance needs no key at all, so an empty `.env`
still gets you a working report. You just won't get the LLM write-up or the
extra fallback providers.

| var | what it does |
|---|---|
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `MODEL` | the LLM layer — works with any OpenAI-compatible endpoint |
| `NEWS_LLM`, `RISK_LLM`, `SYNTH_LLM` | optional: a different model per agent (cheap for news/risk, stronger for the synthesizer) |
| `FMP_API_KEY` | Financial Modeling Prep — backup market-data source, and how peers get resolved |
| `FINNHUB_API_KEY` | backup news source |
| `SEC_USER_AGENT` | `"your name your@email"` — SEC EDGAR requires this |
| `STOCKFACT_PRIMARY=fmp` | use FMP instead of yfinance as the primary data source |
| `STOCKFACT_ENABLE_YAHOO_SCRAPE=1` | turns on a third, last-resort data source — see [Providers](#providers), it's off by default on purpose |
| `STOCKFACT_SKIP_LLM=1` | skip the LLM step entirely, deterministic report only |
| `STOCKFACT_LLM_NO_THINK` | force off a reasoning model's hidden "thinking" tokens (auto-detected for Qwen) |

## Providers

Market data comes from yfinance by default. If that fails, it falls back to
Financial Modeling Prep (only if `FMP_API_KEY` is set). And if you turn it on
explicitly, there's a third fallback that scrapes Yahoo's quote page directly
instead of going through an API at all — I added that mostly out of curiosity
about how far you could get without an official API, and it turns out you can
get pretty far: it parses a JSON blob that Yahoo's own frontend embeds in the
page to render itself with.

I kept it off by default, though. It only works because it happens to read a
`<script>` tag Yahoo embeds for their own frontend's use — nothing guarantees
that stays there, and it's reading their public website directly instead of
calling a documented API, which is a shakier thing to depend on than the
other two providers. It also can't get everything: no price history, no
income statements, just what's on that one page. See
[`src/stockfact/providers/yahoo_scrape_provider.py`](src/stockfact/providers/yahoo_scrape_provider.py)
if you want to see how it actually works.

## Using it

```bash
python -m stockfact.main AAPL                        # markdown report
python -m stockfact.main AAPL --format plain          # explained like you're twelve, no jargon
python -m stockfact.main AAPL --format html --out report.html
python -m stockfact.main AAPL --format json
python -m stockfact.main AAPL -v                      # see what each agent is actually doing
python -m stockfact.main AAPL --strict                # fail loudly if the QA check finds anything wrong
STOCKFACT_SKIP_LLM=1 python -m stockfact.main AAPL    # deterministic layer only, fast

crewai run                                            # interactive, just asks for a ticker
```

`--format plain` was one of my favorite parts to build — the exact same
numbers as the markdown report, just translated into plain language with a
lemonade-stand-style analogy where that helps. Still never tells you whether
to buy anything. That's kind of the whole point of this project.

### The web UI

```bash
stockfact-web                    # http://127.0.0.1:8000
stockfact-web --port 9000 --reload
```

Type in a ticker, pick Simple / Full / JSON, and you get score bars, QA
flags, and a diff against the last time you checked that ticker (if you
have). Turn off "Full analysis" for an instant deterministic-only report —
no waiting on an LLM call.

## Running the tests

```bash
pytest    # fully offline, no real network calls — providers are swapped for fixtures
```

## What's actually in here

| path | what it's for |
|---|---|
| `src/stockfact/models/` | the Pydantic models everything else is built around |
| `src/stockfact/providers/` | where market data actually comes from (yfinance, FMP, the optional scraper) |
| `src/stockfact/fetch/` | turns raw provider data into typed reports, no LLM |
| `src/stockfact/scoring/` | the 0-100 sub-scores, pure functions |
| `src/stockfact/qa/` | checks that nothing in the narrative was made up |
| `src/stockfact/agents/` | the 3 LLM agents |
| `src/stockfact/assemble.py` | puts the final report together |
| `src/stockfact/store/` | saves runs, diffs them against the last one |
| `src/stockfact/report/` | turns the final JSON into Markdown / HTML / plain language |
| `src/stockfact/web/` | the FastAPI UI on top of all of the above |

## Where this stands

It works end to end — I've run it against a bunch of real tickers (AAPL,
NVDA, KO, and others) and it holds up. The deterministic layer covers
fundamentals, technicals, valuation-vs-history, peer comparison, risk
signals, news, SEC filings, scoring, and the QA check. The LLM layer is 3
CrewAI agents, and if there's no API key set or a call fails, it quietly
falls back to a plainer deterministic version instead of breaking.

The older, declarative `crew.jsonc` version of this (before I rewrote it as
plain Python + CrewAI's code interface) is parked in `legacy_crew_jsonc/` if
you're curious what the first attempt looked like.

## One more thing

This is informational only. It will never tell you to buy or sell anything,
on purpose — and that's enforced in code, not just promised in this README.
