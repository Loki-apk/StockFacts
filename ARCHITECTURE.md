# StockFact — Architecture

**The idea:** give it a ticker, get back a structured report — fundamentals,
technical position, news themes, peer context, risk flags — and let the
person reading it make their own call.

**What it will never do, on purpose, enforced in code and not just promised here:**

- No price prediction or price targets.
- No buy / sell / hold / "accumulate" / "wait" language, anywhere, ever.
- No number in the output that doesn't trace back to something actually fetched.

Basically: aggregate and contextualize, don't advise.

This file is the technical design writeup — what got built, why, and in what
order. For a quicker overview and how to actually run it, start with
[README.md](README.md).

**Contents:** [High-level flow](#1-high-level-flow) ·
[Project structure](#2-project-structure-code-based-crewai) ·
[RunContext](#3-runcontext--the-shared-store) ·
[Data models](#4-data-models-pydantic) ·
[LLM agents](#5-the-three-llm-agents) ·
[Deterministic scoring](#6-deterministic-scoring) ·
[QA / trace check](#7-qa--trace-check-deterministic) ·
[Providers, caching, resilience](#8-providers-caching-resilience) ·
[Persistence & diffing](#9-persistence--diffing) ·
[Testing](#10-testing--how-you-prove-no-hallucination) ·
[Build order](#11-build-order) ·
[Tech stack](#12-tech-stack)

---

## 1. High-Level Flow

```
User input: ticker
       │
       ▼
┌──────────────────────┐
│ 1. Resolve & validate│  deterministic — ticker exists? company name, sector,
│    (no LLM)          │  industry, currency, exchange
└──────────┬───────────┘
           │  RunContext { ticker, company, sector, industry, currency }
           ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. DETERMINISTIC FETCH + COMPUTE LAYER   (no LLM, runs in parallel)│
│                                                                   │
│   fundamentals_fetch   → FundamentalsReport                        │
│   price_history_fetch  → TechnicalReport  (SMA/RSI/beta in Python) │
│   valuation_context    → ValuationContext (self-history + peers)   │
│   peer_resolve + fetch → PeerComparisonReport                      │
│   risk_signals_fetch   → partial RiskFlagReport (structured flags) │
│   news_fetch           → list[RawNewsItem]  (headline,url,date,src) │
│   filings_fetch        → list[FilingExcerpt] (EDGAR 1A / MD&A)     │
└──────────┬────────────────────────────────────────────────────────┘
           │  RunContext now holds all typed sub-reports + raw news + filings
           ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. LLM AGENTS (CrewAI)   — language-shaped work only              │
│                                                                   │
│   News & Theme Agent   reads RawNewsItem[] + FilingExcerpt[]       │
│                        → NewsSentimentReport (themes, per-item     │
│                          sentiment label, every item keeps url)    │
│                                                                   │
│   Risk Narrative Agent  reads structured flags + filing risk       │
│                         factors → RiskFlagReport (adds description  │
│                         + severity per flag; no new flags)         │
│                                                                   │
│   Synthesizer Agent     reads ALL typed sub-reports                │
│                         → narrative sections, restating only.      │
│                         Strict prompt. No numbers not in inputs.   │
└──────────┬────────────────────────────────────────────────────────┘
           ▼
┌──────────────────────┐
│ 4. Compose + score   │  deterministic — assemble StockEvaluationReport,
│    (no LLM)          │  compute sub-scores in Python
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 5. QA / trace check  │  deterministic — every numeric token in narrative
│    (no LLM)          │  must match a value in a sub-report (exact/rounded).
│                     │  Unmatched → strip + record violation. Imperative-
│                     │  language scan. Coverage check.
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 6. Render            │  JSON (canonical) + Markdown/HTML. Disclaimer top.
│                     │  Every figure carries source + as-of date.
└──────────┬───────────┘
           ▼
   Store run (SQLite) → diff vs. previous run for this ticker
```

The LLM only ever touches **language**, never arithmetic. If you skipped
step 3 entirely, you'd still get a valid (just terser) report out of the
deterministic layer alone.

---

## 2. Project Structure (code-based CrewAI)

CrewAI's declarative `crew.jsonc` format didn't play well with custom tools,
Pydantic models, and a deterministic QA/scoring/render layer bolted on top of
it, so this moved to a plain code layout instead.

```
stockfact/
├── pyproject.toml
├── .env                          # API keys (gitignored)
├── ARCHITECTURE.md
├── src/stockfact/
│   ├── __init__.py
│   ├── main.py                   # CLI / API entry point
│   ├── run_context.py            # RunContext dataclass — the shared store
│   │
│   ├── models/
│   │   ├── reports.py            # all Pydantic report models
│   │   └── enums.py              # FlagType, Severity, Sentiment, Theme
│   │
│   ├── providers/                # raw data access, provider-agnostic
│   │   ├── base.py               # MarketDataProvider protocol
│   │   ├── yfinance_provider.py  # primary
│   │   ├── fmp_provider.py       # fallback (Financial Modeling Prep / Finnhub)
│   │   ├── yahoo_scrape_provider.py  # opt-in last-resort fallback (STOCKFACT_ENABLE_YAHOO_SCRAPE=1)
│   │   ├── edgar.py              # SEC EDGAR filings
│   │   └── news.py               # NewsAPI / GNews / Finnhub news
│   │
│   ├── fetch/                    # deterministic fetch + compute (step 2)
│   │   ├── pipeline.py           # runs the fetch modules below, fills RunContext
│   │   ├── resolve.py            # ticker → company/sector/industry
│   │   ├── fundamentals.py       # → FundamentalsReport
│   │   ├── technicals.py         # price history + SMA/RSI/beta in numpy
│   │   ├── valuation_context.py  # self-history + peer medians
│   │   ├── peers.py              # peer resolution + looped fundamentals
│   │   ├── risk_signals.py       # earnings calendar, insider tx, short int,
│   │   │                         #   analyst rating changes → structured flags
│   │   ├── news.py               # → list[RawNewsItem]
│   │   ├── filings.py            # → list[FilingExcerpt] (Item 1A, MD&A)
│   │   └── related_exposure.py   # → RelatedExposureReport (customer/supplier/
│   │                             #   partner companies, EDGAR-confirmed + 2y correlation)
│   │
│   ├── agents/                   # CrewAI — step 3 only
│   │   └── crew.py               # 3 agents + tasks, built inline; env-driven LLM
│   │
│   ├── scoring/
│   │   └── composite.py          # sub-scores + rollup, pure functions
│   │
│   ├── qa/
│   │   └── trace_check.py        # numeric traceability + language scan
│   │
│   ├── cache/
│   │   └── cache.py              # SQLite, per-datatype TTL
│   │
│   ├── store/
│   │   └── runs.py               # persist reports, diff two runs
│   │
│   ├── assemble.py               # compose StockEvaluationReport + coverage + QA
│   │
│   ├── report/
│   │   ├── render.py             # JSON → Markdown / HTML
│   │   └── plain.py              # JSON → plain-language ("--format plain")
│   │
│   └── web/                      # FastAPI local UI over the same pipeline
│       └── app.py
│
└── tests/
    ├── fixtures/                 # frozen provider responses for golden tickers
    ├── test_schema.py            # every report validates
    ├── test_trace_check.py       # QA catches injected fake numbers
    ├── test_scoring.py           # scoring is deterministic + bounded
    ├── test_related_exposure.py  # customer/supplier evidence + correlation
    ├── test_store.py             # persistence + diff
    ├── test_web.py               # FastAPI routes
    └── test_yahoo_scrape_provider.py  # opt-in provider, mocked HTTP, no live network
```

---

## 3. RunContext — the shared store

Instead of feeding each agent the previous agent's text output, the
deterministic layer fills in one shared object as it goes. Each agent then
only reads the specific fields it actually needs, as real structured data,
through its task inputs — not a transcript of whatever the last agent wrote.

```python
from dataclasses import dataclass, field

@dataclass
class RunContext:
    ticker: str
    company_name: str
    sector: str
    industry: str
    currency: str
    exchange: str
    run_id: str
    generated_at: date

    fundamentals: FundamentalsReport | None = None
    technicals: TechnicalReport | None = None
    valuation: ValuationContext | None = None
    peers: PeerComparisonReport | None = None
    raw_news: list[RawNewsItem] = field(default_factory=list)
    filings: list[FilingExcerpt] = field(default_factory=list)
    risk_flags_raw: list[RiskFlag] = field(default_factory=list)  # pre-narrative

    # filled by LLM step
    news_report: NewsSentimentReport | None = None
    risk_report: RiskFlagReport | None = None
    narrative: NarrativeSections | None = None

    # filled by QA
    qa_violations: list[str] = field(default_factory=list)
    coverage: Coverage | None = None
```

---

## 4. Data Models (Pydantic)

```python
class MetricWithContext(BaseModel):        # the key primitive everything else builds on
    value: float | None
    unit: str                              # "ratio", "pct", "USD", "x"
    self_median_3y: float | None            # this stock's own history
    self_percentile: float | None           # where current sits in its 5y range
    peer_median: float | None
    sector_median: float | None
    as_of: date
    source: str

class FundamentalsReport(BaseModel):
    ticker: str
    pe_ratio: MetricWithContext | None
    forward_pe: MetricWithContext | None
    ev_to_ebitda: MetricWithContext | None
    revenue_growth_yoy: MetricWithContext | None
    eps_growth_yoy: MetricWithContext | None
    gross_margin: MetricWithContext | None
    profit_margin: MetricWithContext | None
    fcf_margin: MetricWithContext | None             # fcf / revenue
    debt_to_equity: MetricWithContext | None
    interest_coverage: MetricWithContext | None
    current_ratio: MetricWithContext | None
    return_on_equity: MetricWithContext | None
    market_cap: float | None
    data_as_of: date
    source: str
    provider_fallback_used: bool = False
    missing_fields: list[str] = []

class TechnicalReport(BaseModel):
    ticker: str
    current_price: float
    fifty_two_week_high: float
    fifty_two_week_low: float
    pct_off_high: float
    pct_above_low: float
    price_vs_sma50: float | None                      # (price/sma) - 1
    price_vs_sma200: float | None
    sma_50: float | None
    sma_200: float | None
    rsi_14: float | None
    realized_vol_30d: float | None
    beta: float | None
    avg_volume_30d: float | None
    volume_vs_90d_avg: float | None
    data_as_of: date

class ValuationContext(BaseModel):
    ticker: str
    pe_now: float | None
    pe_5y_median: float | None
    pe_5y_low: float | None
    pe_5y_high: float | None
    pe_percentile_5y: float | None                    # 0–100
    fwd_pe_vs_peer_median: float | None
    notes: list[str] = []                             # deterministic strings only
    data_as_of: date

class RawNewsItem(BaseModel):                         # pre-LLM, just the raw item
    headline: str
    source: str
    url: str
    published_at: date
    snippet: str | None

class NewsSentimentItem(BaseModel):
    headline: str
    source: str
    url: str                                          # REQUIRED — no item without a link
    published_at: date
    theme: Theme                                      # enum, not free string
    sentiment: Sentiment                              # enum: positive|negative|neutral
    rationale: str                                    # one clause, grounded in headline

class NewsSentimentReport(BaseModel):
    ticker: str
    window_days: int
    items: list[NewsSentimentItem]
    dominant_themes: list[Theme]
    item_count_by_sentiment: dict[str, int]

class FilingExcerpt(BaseModel):
    filing_type: str                                  # "10-K", "10-Q"
    section: str                                      # "Item 1A", "MD&A"
    filed_at: date
    url: str
    text: str                                         # verbatim excerpt

class RiskFlag(BaseModel):
    flag_type: FlagType                               # enum
    description: str
    severity: Severity                                # enum: info|watch|elevated
    source: str
    as_of: date | None

class RiskFlagReport(BaseModel):
    ticker: str
    flags: list[RiskFlag]

class PeerComparisonReport(BaseModel):
    ticker: str
    peers: list[str]
    peer_selection_method: str                        # "sector+size" | "static table"
    metric_comparison: dict[str, dict[str, float | None]]
    sector_median: dict[str, float | None]

class RelatedExposure(BaseModel):
    related_ticker: str
    related_company_name: str
    relationship_type: str                            # "customer" | "supplier" | "partner"
    evidence_quote: str                               # verbatim, copied from the filing
    evidence_source: str                               # filing URL
    evidence_filed_at: date
    # computed locally from price history, not something the filing itself says
    price_correlation_2y: float | None

class RelatedExposureReport(BaseModel):
    ticker: str
    candidates_checked: int
    exposures: list[RelatedExposure]
    method: str = "static candidate table + SEC EDGAR filing confirmation"

class SubScores(BaseModel):
    valuation: float                                  # 0–100
    growth: float
    financial_health: float
    momentum: float
    explanations: dict[str, str]                      # one line each, the inputs
    rollup: float                                     # labeled heuristic
    rollup_is_heuristic: bool = True

class Coverage(BaseModel):
    fields_populated: int
    fields_total: int
    news_window_days: int
    news_item_count: int
    provider_fallbacks: list[str]
    stale_sections: list[str]                         # data_as_of older than TTL*2

class NarrativeSections(BaseModel):                   # ← LLM output, prose only
    fundamentals_summary: str
    technical_position: str
    news_and_themes: str
    peer_context: str
    risk_summary: str
    related_exposure_summary: str
    # NO overall_recommendation field. By design.

class StockEvaluationReport(BaseModel):
    ticker: str
    company_name: str
    run_id: str
    generated_at: date
    fundamentals: FundamentalsReport
    technicals: TechnicalReport
    valuation: ValuationContext
    news: NewsSentimentReport
    peers: PeerComparisonReport
    risks: RiskFlagReport
    related_exposure: RelatedExposureReport
    scores: SubScores
    coverage: Coverage
    narrative: NarrativeSections
    qa_violations: list[str]
    disclaimer: str = "For informational purposes only. Not financial advice."
```

---

## 5. The three LLM agents

| Agent | Reads | Produces | Guardrail |
|---|---|---|---|
| **News & Theme Agent** | `RawNewsItem[]`, `FilingExcerpt[]` | `NewsSentimentReport` | Every output item must carry a `url` present in the input. Theme/sentiment from a fixed enum. `rationale` ≤ 1 clause. |
| **Risk Narrative Agent** | `risk_flags_raw` (structured), filing risk factors | `RiskFlagReport` (fills `description`, `severity`) | May not add flags. `flag_type` count in == out. Severity from enum. |
| **Synthesizer Agent** | all typed sub-reports | `NarrativeSections` | *"Restate only what is in the inputs. Every number you write must appear verbatim (or rounded) in a sub-report. Do not add analysis, forecasts, or recommendations. Do not use the words buy, sell, hold, accumulate, trim, wait."* |

A few notes on how these three actually run:

- They run under CrewAI's `Process.sequential`, though News + Risk could just
  as easily run in parallel with the Synthesizer going last — this crew is
  small and cheap enough that it doesn't matter much either way.
- Every task sets `output_pydantic=` so CrewAI enforces the schema instead of
  hoping the model returns valid JSON.
- Each run tracks a token budget and logs every call, mostly so a runaway
  agent doesn't burn through an API budget silently.
- News and Risk can get away with a cheaper/smaller model since their job is
  narrow; the Synthesizer benefits from a stronger one since it's stitching
  everything together.

---

## 6. Deterministic scoring

```python
def compute_sub_scores(ctx: RunContext) -> SubScores:
    f, t, v = ctx.fundamentals, ctx.technicals, ctx.valuation

    valuation = pct_rank_inverse(v.pe_percentile_5y)          # cheap vs own history → high
    growth    = clamp_scale(f.revenue_growth_yoy.value, lo=-0.1, hi=0.4)
    health    = blend(
        inv(f.debt_to_equity.value, cap=3.0),
        scale(f.interest_coverage.value, lo=1, hi=12),
        scale(f.current_ratio.value, lo=0.5, hi=2.5),
    )
    momentum  = blend(
        scale(t.price_vs_sma200 or 0, lo=-0.3, hi=0.3),
        centered(t.rsi_14, mid=50, half_width=30),
    )
    subs = dict(valuation=valuation, growth=growth,
                financial_health=health, momentum=momentum)
    return SubScores(
        **subs,
        explanations={
            "valuation": f"PE {v.pe_now} vs 5y median {v.pe_5y_median} "
                         f"(pctile {v.pe_percentile_5y:.0f})",
            "growth":    f"revenue YoY {f.revenue_growth_yoy.value:+.1%}",
            "financial_health": f"D/E {f.debt_to_equity.value}, "
                                f"int. coverage {f.interest_coverage.value}x",
            "momentum":  f"{t.price_vs_sma200:+.1%} vs SMA200, RSI {t.rsi_14:.0f}",
        },
        rollup=sum(subs.values()) / 4,
    )
```

Every one of these scores is a pure function of numbers that were already
fetched — nothing here reaches out to a new data source or guesses. If a
sector median isn't actually available, the answer is `None`, never an
invented number. Same for a missing input to a sub-score: it comes back
`None`, and the rollup notes the gap instead of quietly averaging around it.

---

## 7. QA / trace check (deterministic)

```python
NUM_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*%?")
BANNED = re.compile(r"\b(buy|sell|hold|accumulate|trim|overweight|underweight|"
                    r"undervalued|overvalued|should|recommend|target price)\b", re.I)

def trace_check(report: StockEvaluationReport) -> list[str]:
    known = collect_all_numbers(report.fundamentals, report.technicals,
                                report.valuation, report.peers, report.risks)
    violations = []
    for section, text in report.narrative:
        for tok in NUM_RE.findall(text):
            if not matches_known(tok, known, tol=0.02):
                violations.append(f"{section}: untraceable number {tok!r}")
        for m in BANNED.finditer(text):
            violations.append(f"{section}: banned term {m.group()!r}")
    return violations
```

If this finds a violation, one of two things happens: in strict mode (and in
tests) the run just fails outright, and in normal use the offending sentence
gets stripped out and the violation gets listed in the report instead.
Either way, a run with violations never quietly renders as if nothing
happened.

---

## 8. Providers, caching, resilience

- **Primary:** yfinance. **Fallback:** Financial Modeling Prep, if
  `FMP_API_KEY` is set. Both sit behind a `MarketDataProvider` protocol — the
  fetch layer tries the primary first, falls back on an exception or on core
  fields coming back empty, and records `provider_fallback_used` when that
  happens.
- **Last-resort fallback (opt-in):** `yahoo_scrape_provider.py`, turned on
  with `STOCKFACT_ENABLE_YAHOO_SCRAPE=1`. It fetches the public
  `finance.yahoo.com/quote/{ticker}/` page and parses the `quoteSummary` JSON
  Yahoo's own frontend hydrates itself from (sitting inside a
  `<script data-sveltekit-fetched>` tag), instead of calling an API. It
  covers `resolve`, `fundamentals_raw`, `calendar`, `recommendations`,
  `upgrades_downgrades`, and `short_interest` — every other quote sub-page
  blocks plain HTTP requests, so `price_history`, `income_stmt`, and
  `insider_transactions` just raise `ProviderError` and fall through to
  whatever's next in the chain. It's more brittle than the API-based
  providers, since it depends on how Yahoo happens to lay out that page
  rather than a stable contract — which is exactly why it's off by default.
- **SEC EDGAR** (free) for filings, via `https://data.sec.gov/`. Cached
  aggressively since filings only actually change quarterly.
- **Cache** (SQLite, keyed `datatype:ticker`), with its own TTL per type:
  fundamentals 24h · price 30–60min · valuation-context 24h · news 2h ·
  filings 7d · risk signals 12h. Every cached entry stores `fetched_at`, and
  `Coverage.stale_sections` flags anything older than 2× its TTL that still
  had to be served because nothing fresher was available.
- Every provider call is wrapped so failures come back as a structured
  `{"error": ...}` instead of raising into the crew. If every fetch fails,
  the run aborts before the LLM step even starts, with a clear message about
  what went wrong.

---

## 9. Persistence & diffing

- Every `StockEvaluationReport` gets saved as JSON in SQLite, keyed by
  `(ticker, run_id, generated_at)`.
- On a new run, `diff_reports(prev, curr)` builds a changelog:
  `"PE 22.1 → 24.3"`, `"new risk flag: insider_selling (watch)"`,
  `"earnings date moved 2026-10-28 → 2026-10-21"`, `"news themes: +litigation"`.
- This diff is fully deterministic, and honestly it's probably the single
  most useful feature if you keep coming back to check the same ticker —
  you only see what actually changed, not the whole report again.
- Watchlist mode: run N tickers, get back a table of `SubScores` plus each
  one's top risk flag.

---

## 10. Testing — how you actually prove "no hallucination"

1. **Fixtures:** freeze provider JSON for a handful of golden tickers (a
   megacap, a loss-maker, a high-debt name, a recent IPO, a non-USD listing).
2. `test_schema` — every report model validates against every fixture.
3. `test_trace_check` — inject a fabricated number into a narrative string
   and confirm QA catches it; inject "you should buy" and confirm that gets
   caught too.
4. `test_scoring` — scores stay bounded 0–100, come out the same every time
   given the same input, and a `None` input produces a `None` sub-score
   instead of an exception.
5. A full golden-run snapshot test (pipeline on fixtures, LLM step mocked,
   diff the output JSON) was the original plan for step 5 here — it never
   actually got built, since the schema/trace-check/scoring tests plus
   `test_related_exposure` and `test_store` ended up covering the same
   ground in practice.
6. **CI faithfulness check (optional, not built):** run an LLM-as-judge over
   just `NarrativeSections` against the sub-reports it's supposed to be
   restating, scored purely on "does this contain anything not in the
   inputs," across the golden set.

---

## 11. Build order

The order this actually got built in — worth knowing if you're extending it
or building something similar, since it front-loads the parts that let you
catch design mistakes cheaply (contracts, then data, then pure functions)
before the LLM layer goes in last.

1. `models/` + `run_context.py` — lock the contracts first.
2. `providers/yfinance_provider.py` + `cache/` — get raw data flowing.
3. `fetch/` modules one at a time, each with a fixture test. **This ends up being most of the system.**
4. `scoring/` + `qa/` — pure functions, fully tested, no LLM yet.
5. `report/render.py` — you can demo end-to-end here with an empty narrative.
6. `agents/crew.py` — the three LLM agents, last.
7. `store/` + diffing, then watchlist/CLI polish.

By the end of step 5 you already have something working and genuinely
useful, without a single LLM call in it. The LLM layer is the garnish here,
not the engine.

---

## 12. Tech stack

| Layer | Choice |
|---|---|
| Orchestration | CrewAI (code layout, 3-agent crew) + litellm |
| LLM | `MODEL` + `OPENAI_BASE_URL` in `.env` (any OpenAI-compatible endpoint); per-agent override via `NEWS_LLM`/`RISK_LLM`/`SYNTH_LLM` |
| Market data | yfinance primary, optional FMP fallback (`FMP_API_KEY`), optional Yahoo-scrape last resort (`STOCKFACT_ENABLE_YAHOO_SCRAPE=1`) |
| Filings | SEC EDGAR via `edgartools` |
| News | yfinance feed; Finnhub when `FINNHUB_API_KEY` is set |
| Validation | Pydantic + deterministic Python checks |
| Cache / storage | SQLite |
| Numerics | numpy / pandas (SMA, RSI, vol, beta — never the LLM) |
| Render | Jinja2 → Markdown / HTML |
| Tests | pytest + recorded fixtures |
