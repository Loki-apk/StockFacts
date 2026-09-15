"""Turns a StockEvaluationReport into Markdown or HTML.

The sections in the output map 1:1 to the data model, the disclaimer always
goes at the top, and every number is shown with where it came from and when
it was fetched. There's no "recommendation" section — on purpose, and there
never will be.
"""

from __future__ import annotations

from stockfact.models.reports import StockEvaluationReport


def _metric_line(label: str, m) -> str:
    if m is None or m.value is None:
        return f"- **{label}:** n/a"
    ctx = []
    if m.self_median_3y is not None:
        ctx.append(f"3y median {m.self_median_3y:.2f}")
    if m.peer_median is not None:
        ctx.append(f"peer median {m.peer_median:.2f}")
    if m.self_percentile_5y is not None:
        ctx.append(f"{m.self_percentile_5y:.0f}th pctile of own 5y range")
    suffix = f" ({'; '.join(ctx)})" if ctx else ""
    return f"- **{label}:** {m.value:.2f}{suffix}  \n  _source: {m.source}, as of {m.as_of}_"


def render_markdown(report: StockEvaluationReport, diff: list[str] | None = None) -> str:
    r = report
    out: list[str] = []
    out.append(f"> {r.disclaimer}\n")
    out.append(f"# {r.company_name} ({r.ticker})")
    out.append(f"_Generated {r.generated_at} · run {r.run_id}_\n")

    if r.qa_violations:
        out.append("## ⚠️ QA violations (review before trusting this report)\n")
        out += [f"- {v}" for v in r.qa_violations]
        out.append("")

    if diff:
        out.append("## Since last run\n")
        out += [f"- {d}" for d in diff]
        out.append("")

    out.append("## Fundamentals\n")
    f = r.fundamentals
    for label, attr in [
        ("P/E", "pe_ratio"), ("Forward P/E", "forward_pe"), ("EV/EBITDA", "ev_to_ebitda"),
        ("Revenue growth YoY", "revenue_growth_yoy"), ("EPS growth YoY", "eps_growth_yoy"),
        ("Gross margin", "gross_margin"), ("Profit margin", "profit_margin"),
        ("FCF margin", "fcf_margin"), ("Debt/Equity", "debt_to_equity"),
        ("Interest coverage", "interest_coverage"), ("Current ratio", "current_ratio"),
        ("Return on equity", "return_on_equity"),
    ]:
        out.append(_metric_line(label, getattr(f, attr)))
    if f.missing_fields:
        out.append(f"\n_Missing: {', '.join(f.missing_fields)}_")
    if r.narrative:
        out.append(f"\n{r.narrative.fundamentals_summary}")

    out.append("\n## Technical position\n")
    t = r.technicals
    out.append(f"- **Price:** {t.current_price:.2f} ({r.fundamentals.source})")
    out.append(f"- **52w range:** {t.fifty_two_week_low:.2f} – {t.fifty_two_week_high:.2f} "
               f"({t.pct_off_high:+.1%} off high)")
    if t.price_vs_sma50 is not None:
        sma200 = "n/a" if t.price_vs_sma200 is None else format(t.price_vs_sma200, "+.1%")
        out.append(f"- **vs SMA50 / SMA200:** {t.price_vs_sma50:+.1%} / {sma200}")
    if t.rsi_14 is not None:
        out.append(f"- **RSI(14):** {t.rsi_14:.0f}")
    if r.narrative:
        out.append(f"\n{r.narrative.technical_position}")

    out.append("\n## Valuation context\n")
    v = r.valuation
    out.append(f"- **P/E now:** {v.pe_now}")
    out.append(f"- **5y median / low / high:** {v.pe_5y_median} / {v.pe_5y_low} / {v.pe_5y_high}")
    for n in v.notes:
        out.append(f"- _{n}_")

    out.append("\n## News & themes\n")
    n = r.news
    out.append(f"_Window: {n.window_days}d · {len(n.items)} items · "
               f"themes: {', '.join(n.dominant_themes) or 'none'}_\n")
    for item in n.items:
        out.append(f"- [{item.headline}]({item.url}) — {item.source}, {item.published_at} "
                   f"· {item.theme}/{item.sentiment}")
    if r.narrative:
        out.append(f"\n{r.narrative.news_and_themes}")

    out.append("\n## Peer comparison\n")
    out.append(f"_Peers ({r.peers.peer_selection_method}): {', '.join(r.peers.peers) or 'none'}_")
    if r.narrative:
        out.append(f"\n{r.narrative.peer_context}")

    out.append("\n## Related exposure\n")
    re_ = r.related_exposure
    if not re_.exposures:
        out.append(f"- none confirmed ({re_.candidates_checked} candidate(s) checked)")
    for exp in re_.exposures:
        corr = (
            f", 2y price correlation {exp.price_correlation_2y:+.2f}"
            if exp.price_correlation_2y is not None
            else ""
        )
        out.append(
            f"- **{exp.related_company_name} ({exp.related_ticker})** — "
            f"documented {exp.relationship_type}{corr}  \n"
            f"  _\"{exp.evidence_quote}\"_ — [source]({exp.evidence_source}), "
            f"filed {exp.evidence_filed_at}"
        )
    if r.narrative:
        out.append(f"\n{r.narrative.related_exposure_summary}")

    out.append("\n## Risk flags\n")
    if not r.risks.flags:
        out.append("- none raised")
    for fl in r.risks.flags:
        out.append(f"- **{fl.flag_type}** ({fl.severity}) — {fl.description} _({fl.source})_")
    if r.narrative:
        out.append(f"\n{r.narrative.risk_summary}")

    out.append("\n## Scores\n")
    s = r.scores
    for k in ("valuation", "growth", "financial_health", "momentum"):
        out.append(f"- **{k}:** {getattr(s, k)}  "
                   f"{'— ' + s.explanations[k] if k in s.explanations else ''}")
    out.append(f"- **rollup (heuristic, not a rating):** {s.rollup}")

    cov = r.coverage
    cov_line = (
        f"\n_Coverage: {cov.fields_populated}/{cov.fields_total} fields · "
        f"news {cov.news_item_count} items / {cov.news_window_days}d"
    )
    if cov.provider_fallbacks:
        cov_line += f" · fallbacks: {', '.join(cov.provider_fallbacks)}"
    if cov.stale_sections:
        cov_line += f" · stale: {', '.join(cov.stale_sections)}"
    out.append(cov_line + "_")

    return "\n".join(out) + "\n"


def render_html(report: StockEvaluationReport, diff: list[str] | None = None) -> str:
    """Minimal wrapper. TODO: Jinja2 template with proper styling."""
    md = render_markdown(report, diff)
    try:
        import markdown as _md

        body = _md.markdown(md, extensions=["tables"])
    except ImportError:
        body = f"<pre>{md}</pre>"
    return f"<!doctype html><meta charset=utf-8><title>{report.ticker}</title>{body}"
