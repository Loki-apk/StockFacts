# Stock Evaluation System — Architecture 

**Goal:** Given a ticker, produce a structured, fully-sourced evaluation report —
fundamentals, technical position, news themes, peer context, and risk flags —
so a person can make their own decision.

**Non-goals (hard rules, enforced in code):**

- No price prediction or price targets.
- No buy / sell / hold / "accumulate" / "wait" language, anywhere, ever.
- No number in the output that doesn't trace to a fetched data point.

The value proposition is *auditable aggregation with context*, not advice.

This is the technical design record — what was built, why, and in what order.
For a quick overview and how to run it, start with [README.md](README.md).

**Contents:** [What changed from v1](#0-what-changed-from-v1) ·
[High-level flow](#1-high-level-flow) ·
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

## 0. What changed from v1

| v1 | v2 | Why |
|---|---|---|
| 7 LLM agents, one per dimension | Deterministic core + 3 LLM agents | Fundamentals/technicals are tool-call-then-fill; an LLM loop there is cost, latency, and hallucination surface for no gain. |
| `Process.sequential`, data passed as text between agents | Tools write typed objects to a shared `RunContext`; agents read objects | Passing financials through 5 LLM contexts is lossy and expensive. |
| "Best time to buy and sell" | "Valuation vs. history + peers, price position, catalysts, risks" | "Best time to buy" is a prediction wearing a hat. |
| Absolute PE, etc. | Every metric shown vs. its own 3–5y history and peer median | Absolute multiples are close to meaningless. |
| yfinance only | yfinance primary + one paid fallback provider | yfinance scrapes Yahoo; it breaks and rate-limits. |
| Composite score headline | Sub-scores with one-line explanations; rollup labeled a heuristic | A single number invites the over-trust we're trying to avoid. |
| QA as an agent | QA as deterministic Python | Cheaper, reliable, and it's the last line of defense — it shouldn't itself hallucinate. |

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

The LLM touches **language**, never arithmetic. If step 3 were skipped entirely
you'd still get a valid (if terse) report from the deterministic layer.

---

## 2. Project Structure (code-based CrewAI)

The declarative `crew.jsonc` format doesn't carry custom tools + Pydantic +
a deterministic QA/scoring/render layer well. Move to the code layout.

```
stocks/
├── pyproject.toml
├── .env                          # API keys (gitignored)
├── ARCHITECTURE.md
├── src/stocks/
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
│   │   ├── yahoo_scrape_provider.py  # opt-in last-resort fallback (STOCKS_ENABLE_YAHOO_SCRAPE=1)
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

Instead of feeding each agent the previous agent's text, the deterministic layer
fills one object. Agents receive **only the fields they need**, as structured
data, via task inputs.

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

Additions/changes from v1 in **bold**.

```python
class MetricWithContext(BaseModel):        # ← NEW, the key primitive
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
    ev_to_ebitda: MetricWithContext | None          # ← NEW
    revenue_growth_yoy: MetricWithContext | None
    eps_growth_yoy: MetricWithContext | None         # ← NEW
    gross_margin: MetricWithContext | None           # ← NEW
    profit_margin: MetricWithContext | None
    fcf_margin: MetricWithContext | None             # ← NEW (fcf / revenue)
    debt_to_equity: MetricWithContext | None
    interest_coverage: MetricWithContext | None      # ← NEW
    current_ratio: MetricWithContext | None          # ← NEW
    return_on_equity: MetricWithContext | None       # ← NEW
    market_cap: float | None
    data_as_of: date
    source: str
    provider_fallback_used: bool = False             # ← NEW
    missing_fields: list[str] = []

class TechnicalReport(BaseModel):
    ticker: str
    current_price: float
    fifty_two_week_high: float
    fifty_two_week_low: float
    pct_off_high: float
    pct_above_low: float                              # ← NEW
    price_vs_sma50: float | None                      # ← NEW ((price/sma)-1)
    price_vs_sma200: float | None                     # ← NEW
    sma_50: float | None
    sma_200: float | None
    rsi_14: float | None
    realized_vol_30d: float | None                    # ← NEW
    beta: float | None
    avg_volume_30d: float | None
    volume_vs_90d_avg: float | None                   # ← NEW
    data_as_of: date

class ValuationContext(BaseModel):                    # ← NEW report
    ticker: str
    pe_now: float | None
    pe_5y_median: float | None
    pe_5y_low: float | None
    pe_5y_high: float | None
    pe_percentile_5y: float | None                    # 0–100
    fwd_pe_vs_peer_median: float | None
    notes: list[str] = []                             # deterministic strings only
    data_as_of: date

class RawNewsItem(BaseModel):                         # ← NEW, pre-LLM
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

class FilingExcerpt(BaseModel):                       # ← NEW
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

class RelatedExposure(BaseModel):                     # ← NEW
    related_ticker: str
    related_company_name: str
    relationship_type: str                            # "customer" | "supplier" | "partner"
    evidence_quote: str                               # verbatim, copied from the filing
    evidence_source: str                               # filing URL
    evidence_filed_at: date
    price_correlation_2y: float | None                # computed locally, not from the filing

class RelatedExposureReport(BaseModel):                # ← NEW
    ticker: str
    candidates_checked: int
    exposures: list[RelatedExposure]
    method: str = "static candidate table + SEC EDGAR filing confirmation"

class SubScores(BaseModel):                           # ← replaces bare composite
    valuation: float                                  # 0–100
    growth: float
    financial_health: float
    momentum: float
    explanations: dict[str, str]                      # one line each, the inputs
    rollup: float                                     # labeled heuristic
    rollup_is_heuristic: bool = True

class Coverage(BaseModel):                            # ← NEW
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

Notes:

- Use CrewAI `Process.sequential` for just these three, or run News + Risk in
  parallel and Synthesizer after. This crew is small and cheap.
- `output_pydantic=` on every task so CrewAI enforces the schema.
- Set a per-run token budget; log every call.
- Consider a smaller/cheaper model for News & Risk, a stronger one for Synthesizer.

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

Every score is a pure function of already-fetched numbers. No sector-relative
normalization without a real sector median in hand (fall back to `None`, don't
invent one). Missing input → that sub-score is `None`, not a guess, and the
rollup notes it.

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

Policy: on any violation, either (a) hard-fail the run in strict mode, or
(b) strip the offending sentence and list the violation in the report. Default to
(a) in tests, (b) in production. A run with violations never renders silently.

---

## 8. Providers, caching, resilience

- **Primary:** yfinance. **Fallback:** one of Financial Modeling Prep / Finnhub /
  Alpha Vantage / Tiingo (you have keys — pick one paid tier). A `MarketDataProvider`
  protocol; the fetch layer tries primary, falls back on exception or empty core
  fields, and sets `provider_fallback_used`.
- **Last-resort fallback (opt-in):** `yahoo_scrape_provider.py`, enabled with
  `STOCKS_ENABLE_YAHOO_SCRAPE=1`. Fetches the public `finance.yahoo.com/quote/{ticker}/`
  page and parses the `quoteSummary` JSON Yahoo's own frontend hydrates from
  (embedded in a `<script data-sveltekit-fetched>` tag) rather than calling an
  API. Covers `resolve`, `fundamentals_raw`, `calendar`, `recommendations`,
  `upgrades_downgrades`, `short_interest`; every other quote sub-page bot-walls
  plain HTTP requests, so `price_history`, `income_stmt`, and
  `insider_transactions` raise `ProviderError` and fall through. More brittle
  than the API-based providers (depends on Yahoo's page markup) and off by
  default for that reason.
- **SEC EDGAR** (free) for filings — `https://data.sec.gov/`. Cache aggressively;
  filings change quarterly.
- **Cache** (SQLite, keyed `datatype:ticker`), per-type TTL:
  fundamentals 24h · price 30–60min · valuation-context 24h · news 2h ·
  filings 7d · risk signals 12h. Store `fetched_at`; `Coverage.stale_sections`
  flags anything older than 2×TTL that had to be served anyway.
- Every provider call wrapped: returns `{"error": ...}` structurally, never raises
  into the crew. An all-error fetch aborts before the LLM step with a clear message.

---

## 9. Persistence & diffing

- Store every `StockEvaluationReport` as JSON in SQLite keyed `(ticker, run_id, generated_at)`.
- On a new run, `diff_reports(prev, curr)` emits a changelog:
  `"PE 22.1 → 24.3"`, `"new risk flag: insider_selling (watch)"`,
  `"earnings date moved 2026-10-28 → 2026-10-21"`, `"news themes: +litigation"`.
- The diff is deterministic and is the highest-value recurring-use feature.
- Watchlist mode: run N tickers, output a table of `SubScores` + top risk flag.

---

## 10. Testing — how you prove "no hallucination"

1. **Fixtures:** freeze provider JSON for ~5 golden tickers (a megacap, a
   loss-maker, a high-debt name, a recent IPO, a non-USD listing).
2. `test_schema` — every report model validates on every fixture.
3. `test_trace_check` — inject a fabricated number into a narrative string,
   assert QA catches it; inject "you should buy", assert QA catches it.
4. `test_scoring` — scores bounded 0–100, deterministic across runs, `None`
   inputs produce `None` sub-scores not exceptions.
5. `test_golden_run` — full pipeline on fixtures (LLM step mocked or recorded),
   snapshot the output JSON; a diff is a review gate.
6. **CI faithfulness check (optional):** LLM-as-judge on just `NarrativeSections`
   vs. the sub-reports, scored for "contains only restated facts", run on the
   golden set.

---

## 11. Build order

The order this was actually built in — useful if you're extending it or
building something similar, since it front-loads the parts that let you catch
design mistakes cheaply (contracts, then data, then pure functions) before
adding the LLM layer at the end.

1. `models/` + `run_context.py` — lock the contracts first.
2. `providers/yfinance_provider.py` + `cache/` — get raw data flowing.
3. `fetch/` modules one at a time, each with a fixture test. **This is most of the system.**
4. `scoring/` + `qa/` — pure functions, fully tested, no LLM yet.
5. `report/render.py` — you can demo end-to-end here with an empty narrative.
6. `agents/crew.py` — the three LLM agents, last.
7. `store/` + diffing, then watchlist/CLI polish.

You have a working, useful, non-hallucinating tool after step 5. The LLM is the
garnish, not the engine.

---

## 12. Tech stack

| Layer | Choice |
|---|---|
| Orchestration | CrewAI (code layout, 3-agent crew) + litellm |
| LLM | `MODEL` + `OPENAI_BASE_URL` in `.env` (any OpenAI-compatible endpoint); per-agent override via `NEWS_LLM`/`RISK_LLM`/`SYNTH_LLM` |
| Market data | yfinance + optional FMP fallback (`FMP_API_KEY`) |
| Filings | SEC EDGAR via `edgartools` |
| News | yfinance feed; Finnhub when `FINNHUB_API_KEY` is set |
| Validation | Pydantic + deterministic Python checks |
| Cache / storage | SQLite |
| Numerics | numpy / pandas (SMA, RSI, vol, beta — never the LLM) |
| Render | Jinja2 → Markdown / HTML |
| Tests | pytest + recorded fixtures |
