"""The 3-agent CrewAI layer — ARCHITECTURE.md step 3.

Reads typed sub-reports off the RunContext, writes NewsSentimentReport /
RiskFlagReport / NarrativeSections back onto it. Language work only, never
arithmetic.

If CrewAI is unavailable, no LLM key is set, or the step errors, the caller
still gets a valid (terser) report from the deterministic layer via
``_fallback_reports``.
"""

from __future__ import annotations

import json
import logging
import os

from stocks.models.enums import Sentiment, Theme
from stocks.models.reports import (
    NarrativeSections,
    NewsSentimentItem,
    NewsSentimentReport,
    RiskFlagReport,
)
from stocks.run_context import RunContext

log = logging.getLogger(__name__)


def _normalize_model(model: str) -> str:
    """litellm needs a provider prefix. Bare names -> the OpenAI-compatible path
    (covers custom base URLs set via OPENAI_BASE_URL / OPENAI_API_BASE)."""
    return model if "/" in model else f"openai/{model}"


# generous, because reasoning models (e.g. Qwen3.x) spend tokens on
# reasoning_content before emitting the answer
_MAX_TOKENS = int(os.environ.get("STOCKS_LLM_MAX_TOKENS", "8000"))


def _verbose() -> bool:
    """Stream each agent's thinking + tool calls. `-v` on the CLI sets this."""
    return os.environ.get("STOCKS_CREW_VERBOSE") == "1"


def _no_think(model: str) -> bool:
    """Whether to ask the server to skip chain-of-thought. Reasoning models burn
    the whole completion budget on `reasoning_content` and never emit the JSON
    crewai's structured output needs. Default on for Qwen3.x; override with
    STOCKS_LLM_NO_THINK=0/1."""
    env = os.environ.get("STOCKS_LLM_NO_THINK")
    if env is not None:
        return env == "1"
    return "qwen" in model.lower()


def _llm(var: str, default: str | None = None):
    """Build a crewai LLM. Honours OPENAI_BASE_URL for custom OpenAI-compatible
    endpoints so a single MODEL/base/key trio in .env drives all three agents."""
    from crewai import LLM

    model = _normalize_model(
        os.environ.get(var) or default or os.environ.get("MODEL") or "gpt-4o-mini"
    )
    base = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    kwargs: dict = {"model": model, "max_tokens": _MAX_TOKENS, "temperature": 0.1}
    if model.startswith("openai/") and base:
        kwargs["base_url"] = base
        kwargs["api_key"] = os.environ.get("OPENAI_API_KEY")
    if _no_think(model):
        # vLLM / Qwen convention; harmless to other OpenAI-compatible servers
        kwargs["additional_params"] = {
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
        }
    return LLM(**kwargs)


def _has_llm_key() -> bool:
    return any(
        os.environ.get(k)
        for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY")
    )


def _dump(model) -> str:
    if model is None:
        return "null"
    if hasattr(model, "model_dump_json"):
        return model.model_dump_json()
    return json.dumps(model, default=str)


# --------------------------------------------------------------------------- #
# crew assembly
# --------------------------------------------------------------------------- #
def _build_agents():
    from crewai import Agent

    news = Agent(
        role="Financial news classifier",
        goal=(
            "Assign exactly one theme and one sentiment from the allowed enums to "
            "each supplied news item and write a one-clause rationale grounded "
            "strictly in the headline. Can drop a URL. Never invent items."
        ),
        backstory=(
            "A meticulous wire-desk editor. You classify; you do not speculate "
            "about price impact and you never call a stock cheap, expensive, or a "
            "buy or sell. Ambiguous headline -> neutral."
        ),
        llm=_llm("NEWS_LLM"),
        verbose=_verbose(),
        allow_delegation=False,
    )
    risk = Agent(
        role="Risk factor summariser",
        goal=(
            "Given a fixed list of structured risk flags plus verbatim filing "
            "excerpts, write a plain 1-2 sentence description and confirm a "
            "severity (info | watch | elevated) for each flag. Add no flags."
        ),
        backstory=(
            "A compliance analyst. You restate documented risks in clear language. "
            "You do not forecast outcomes or probabilities and you give no advice."
        ),
        llm=_llm("RISK_LLM"),
        verbose=_verbose(),
        allow_delegation=False,
    )
    synth = Agent(
        role="Report synthesiser",
        goal=(
            "Produce five short prose sections that restate ONLY what is present "
            "in the provided typed sub-reports."
        ),
        backstory=(
            "You restate. Every number you write must appear verbatim or rounded "
            "in a sub-report. You add no analysis, no forecasts, no interpretation, "
            "no numbers of your own. You never use the words buy, sell, hold, "
            "accumulate, trim, wait, undervalued, overvalued, cheap, expensive, or "
            "'target price'. If the data does not support a sentence, you omit it."
        ),
        llm=_llm("SYNTH_LLM"),
        verbose=_verbose(),
        allow_delegation=False,
    )
    return news, risk, synth


def _news_task(agent, ctx: RunContext):
    from crewai import Task

    themes = ", ".join(t.value for t in Theme)
    raw = json.dumps([i.model_dump(mode="json") for i in ctx.raw_news])
    filings = json.dumps([f.model_dump(mode="json") for f in ctx.filings])
    return Task(
        description=(
            f"Classify each news item for {ctx.ticker} ({ctx.company_name}).\n"
            f"Allowed themes: {themes}\n"
            f"Allowed sentiment: positive, negative, neutral\n"
            f"Return one item per input item, copying headline/source/url/"
            f"published_at exactly.\n\n"
            f"NEWS ITEMS (JSON):\n{raw}\n\n"
            f"FILING EXCERPTS (context only, do not classify):\n{filings}"
        ),
        expected_output=(
            "A NewsSentimentReport: items[] each carrying a url from the input; "
            "dominant_themes[] (<=3); item_count_by_sentiment."
        ),
        agent=agent,
        output_pydantic=NewsSentimentReport,
    )


def _risk_task(agent, ctx: RunContext):
    from crewai import Task

    flags = json.dumps([f.model_dump(mode="json") for f in ctx.risk_flags_raw])
    filings = json.dumps([f.model_dump(mode="json") for f in ctx.filings])
    return Task(
        description=(
            f"For each structured flag below for {ctx.ticker}, write `description` "
            f"(1-2 plain sentences) and confirm `severity`. Do NOT add flags. Keep "
            f"the same flag_type set. Ground wording in the filing excerpts where "
            f"relevant.\n\nSTRUCTURED FLAGS (JSON):\n{flags}\n\n"
            f"FILING EXCERPTS (JSON):\n{filings}"
        ),
        expected_output="A RiskFlagReport whose flags[] has the same flag_type set as the input.",
        agent=agent,
        output_pydantic=RiskFlagReport,
    )


def _synth_task(agent, ctx: RunContext, context_tasks):
    from crewai import Task

    return Task(
        description=(
            f"Write the five narrative sections for {ctx.ticker} "
            f"({ctx.company_name}) using ONLY these sub-reports. Restate; do not "
            f"analyse or forecast. Every number must appear in the inputs.\n\n"
            f"FUNDAMENTALS:\n{_dump(ctx.fundamentals)}\n\n"
            f"TECHNICALS:\n{_dump(ctx.technicals)}\n\n"
            f"VALUATION CONTEXT:\n{_dump(ctx.valuation)}\n\n"
            f"PEERS:\n{_dump(ctx.peers)}\n\n"
            f"RELATED EXPOSURE (companies documented in SEC filings as a "
            f"customer/supplier/partner, plus their 2y price correlation — "
            f"restate only what's here, do not editorialize about whether "
            f"the correlation is meaningful):\n{_dump(ctx.related_exposure)}\n\n"
            f"(news and risk reports come from the previous tasks)"
        ),
        expected_output=(
            "A NarrativeSections object: fundamentals_summary, technical_position, "
            "news_and_themes, peer_context, risk_summary, "
            "related_exposure_summary. No recommendation, no banned terms."
        ),
        agent=agent,
        context=list(context_tasks),
        output_pydantic=NarrativeSections,
    )


def _run_crew(ctx: RunContext) -> None:
    from crewai import Crew, Process

    news_agent, risk_agent, synth_agent = _build_agents()
    tasks = []
    news_task = risk_task = None

    if ctx.raw_news:
        news_task = _news_task(news_agent, ctx)
        tasks.append(news_task)
    if ctx.risk_flags_raw:
        risk_task = _risk_task(risk_agent, ctx)
        tasks.append(risk_task)

    synth_context = [t for t in (news_task, risk_task) if t is not None]
    synth_task = _synth_task(synth_agent, ctx, synth_context)
    tasks.append(synth_task)

    crew = Crew(agents=[news_agent, risk_agent, synth_agent], tasks=tasks,
                process=Process.sequential, verbose=_verbose())
    result = crew.kickoff()

    by_name: dict[str, object] = {}
    for to in result.tasks_output:
        if getattr(to, "pydantic", None) is not None:
            by_name[type(to.pydantic).__name__] = to.pydantic

    ctx.news_report = by_name.get("NewsSentimentReport") or _passthrough_news(ctx)
    ctx.risk_report = by_name.get("RiskFlagReport") or RiskFlagReport(
        ticker=ctx.ticker, flags=list(ctx.risk_flags_raw)
    )
    narrative = by_name.get("NarrativeSections")
    if narrative is None:
        raise RuntimeError("synthesizer produced no structured output")
    ctx.narrative = narrative


# --------------------------------------------------------------------------- #
# fallback (no LLM)
# --------------------------------------------------------------------------- #
def _passthrough_news(ctx: RunContext) -> NewsSentimentReport:
    from stocks.fetch.news import NEWS_WINDOW_DAYS

    items = [
        NewsSentimentItem(
            headline=n.headline,
            source=n.source,
            url=n.url,
            published_at=n.published_at,
            theme=Theme.OTHER,
            sentiment=Sentiment.NEUTRAL,
            rationale="unclassified (LLM layer skipped)",
        )
        for n in ctx.raw_news
    ]
    return NewsSentimentReport(
        ticker=ctx.ticker,
        window_days=NEWS_WINDOW_DAYS,
        items=items,
        dominant_themes=[],
        item_count_by_sentiment={"neutral": len(items)},
    )


def _fallback_reports(ctx: RunContext) -> None:
    ctx.news_report = _passthrough_news(ctx)
    ctx.risk_report = RiskFlagReport(ticker=ctx.ticker, flags=list(ctx.risk_flags_raw))
    ctx.narrative = NarrativeSections(
        fundamentals_summary="(LLM synthesis skipped — see typed fundamentals above.)",
        technical_position="(LLM synthesis skipped — see typed technicals above.)",
        news_and_themes="(LLM synthesis skipped — headlines listed above, unclassified.)",
        peer_context="(LLM synthesis skipped — see the peer table above.)",
        risk_summary="(LLM synthesis skipped — see structured risk flags above.)",
        related_exposure_summary=(
            "(LLM synthesis skipped — see the related exposure table above.)"
        ),
    )


def run_llm_layer(ctx: RunContext, *, allow_fallback: bool = True) -> RunContext:
    if os.environ.get("STOCKS_SKIP_LLM") or not _has_llm_key():
        if not os.environ.get("STOCKS_SKIP_LLM"):
            log.info("no LLM API key found — running deterministic layer only")
        _fallback_reports(ctx)
        return ctx
    try:
        _run_crew(ctx)
    except Exception as exc:  # noqa: BLE001
        if not allow_fallback:
            raise
        log.warning("LLM layer failed (%s) — using deterministic fallback", exc)
        _fallback_reports(ctx)
    return ctx
