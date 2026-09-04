"""Plain-language rendering — the same numbers, explained like you're twelve.

Deterministic. Every sentence is a restatement of a typed field with a simple
framing and, where it helps, a lemonade-stand analogy. It adds no opinion and
never says whether to buy — that stays true to the system's hard rules.
"""

from __future__ import annotations

from stocks.models.reports import StockEvaluationReport

_KID_DISCLAIMER = (
    "This page just explains facts about a company in simple words. "
    "It does NOT tell anyone to buy or sell anything. "
    "Money decisions are for a grown-up you trust."
)


def _big_money(n: float | None) -> str:
    if n is None:
        return "an unknown amount"
    n = float(n)
    for scale, word in ((1e12, "trillion"), (1e9, "billion"), (1e6, "million")):
        if abs(n) >= scale:
            return f"${n / scale:.1f} {word}"
    return f"${n:,.0f}"


def _pct(x: float | None, digits: int = 0) -> str | None:
    return None if x is None else f"{x * 100:.{digits}f}%"


def _cents_on_the_dollar(ratio: float | None) -> str | None:
    """0.28 -> 'about 28 cents'; 1.49 -> 'about $1.49'."""
    if ratio is None:
        return None
    if abs(ratio) >= 1:
        return f"about ${ratio:.2f}"
    return f"about {round(ratio * 100)} cents"


def _bar(score: float | None, width: int = 20) -> str:
    if score is None:
        return "(no score)"
    filled = round(score / 100 * width)
    return "█" * filled + "░" * (width - filled) + f"  {score:.0f}/100"


def render_plain(report: StockEvaluationReport, diff: list[str] | None = None) -> str:
    r = report
    f, t, v, s = r.fundamentals, r.technicals, r.valuation, r.scores
    out: list[str] = []

    out.append(f"# {r.company_name} — explained simply")
    out.append(f"_(stock symbol: {r.ticker}. Numbers as of {r.generated_at}.)_\n")
    out.append(f"> {_KID_DISCLAIMER}\n")

    out.append("## Think of it like a giant lemonade stand\n")
    out.append(
        f"{r.company_name} is a company you can own a tiny slice of by buying a "
        f'"share." Buying the *whole* company right now would cost about '
        f"**{_big_money(f.market_cap)}**.\n"
    )

    # --- how much money it makes ---
    out.append("## Does it make money?\n")
    pm = f.profit_margin.value if f.profit_margin else None
    if pm is not None:
        out.append(
            f"- For every **$1** the company gets from selling things, it keeps "
            f"**{_cents_on_the_dollar(pm)}** as profit. The rest pays for making "
            f"and selling the product."
        )
    rg = f.revenue_growth_yoy.value if f.revenue_growth_yoy else None
    if rg is not None:
        direction = "bigger" if rg >= 0 else "smaller"
        out.append(
            f"- Its sales this year are about **{_pct(abs(rg))} {direction}** than "
            f"last year."
        )
    roe = f.return_on_equity.value if f.return_on_equity else None
    if roe is not None:
        out.append(
            f"- For every $1 the owners have put in, the company earned "
            f"**{_cents_on_the_dollar(roe)}** back over the past year."
        )
    out.append("")

    # --- does it owe money ---
    out.append("## Does it owe money?\n")
    de = f.debt_to_equity.value if f.debt_to_equity else None
    if de is not None:
        out.append(
            f"- For every **$1** the owners put in, the company has borrowed about "
            f"**{round(de * 100)} cents**. "
            + ("That's not a lot." if de < 0.5 else
               "That's a moderate amount." if de < 1.5 else
               "That's a fair bit of borrowing.")
        )
    ic = f.interest_coverage.value if f.interest_coverage else None
    if ic is not None:
        out.append(
            f"- It earns about **{ic:.0f} times** what it needs to pay in loan "
            f"interest, so paying that is "
            + ("comfortable." if ic >= 4 else "something to watch.")
        )
    if de is None and ic is None:
        out.append("- We don't have clear numbers on its borrowing right now.")
    out.append("")

    # --- the price ---
    out.append("## What about the price?\n")
    out.append(f"- One share costs **${t.current_price:,.2f}** today.")
    out.append(
        f"- Over the past year the price has swung between **${t.fifty_two_week_low:,.2f}** "
        f"and **${t.fifty_two_week_high:,.2f}**. "
        + (
            "Right now it's near the top of that range."
            if t.pct_off_high > -0.05
            else "Right now it's near the bottom of that range."
            if t.pct_above_low < 0.05
            else "Right now it's somewhere in the middle."
        )
    )
    if v.pe_now is not None:
        line = (
            f"- People are paying about **${v.pe_now:.0f}** for every **$1** the "
            f"company earns in a year."
        )
        if v.pe_5y_median is not None:
            usually = round(v.pe_5y_median)
            past = f" Over the last 5 years that number was usually around **${usually}**"
            if v.pe_now > v.pe_5y_median * 1.1:
                line += past + ", so people are paying more than usual."
            elif v.pe_now < v.pe_5y_median * 0.9:
                line += past + ", so people are paying less than usual."
            else:
                line += past + " too — about the same as now."
        out.append(line)
    out.append("")

    # --- news ---
    out.append("## What's in the news?\n")
    if not r.news.items:
        out.append("- Nothing notable in the last few days.")
    else:
        # show the items with a clear mood first — they're the interesting ones
        ranked = sorted(
            r.news.items, key=lambda i: 0 if i.sentiment != "neutral" else 1
        )
        for item in ranked[:6]:
            mood = {"positive": "🙂", "negative": "🙁", "neutral": "😐"}.get(
                item.sentiment, "😐"
            )
            out.append(
                f"- {mood} [{item.headline}]({item.url})  "
                f"\n  _({item.source}, {item.published_at})_"
            )
    out.append("")

    # --- related exposure ---
    out.append("## Companies connected to this one\n")
    exposures = r.related_exposure.exposures
    if not exposures:
        out.append(
            "- We don't have a documented customer/supplier/partner link on file "
            "for this one yet."
        )
    else:
        rel_word = {
            "customer": "is a documented customer of",
            "supplier": "is a documented supplier to",
            "partner": "is a documented partner of",
        }
        for exp in exposures:
            phrase = rel_word.get(exp.relationship_type, "is documented as connected to")
            line = f"- **{exp.related_company_name}** {phrase} {r.company_name}"
            if exp.price_correlation_2y is not None:
                strength = (
                    "closely" if abs(exp.price_correlation_2y) >= 0.6
                    else "somewhat" if abs(exp.price_correlation_2y) >= 0.3
                    else "loosely"
                )
                direction = "the same direction" if exp.price_correlation_2y >= 0 else "opposite directions"
                line += (
                    f" — over the past 2 years their share prices have moved "
                    f"{strength} in {direction}"
                )
            out.append(line + ".")
        out.append(
            "\n_This is a documented business link plus a historical price "
            "pattern, not a prediction that it will continue._"
        )
    out.append("")

    # --- things to keep an eye on ---
    out.append("## Things grown-ups keep an eye on\n")
    if not r.risks.flags:
        out.append("- Nothing unusual flagged right now.")
    else:
        plain_flag = {
            "earnings_upcoming": (
                "The company will soon report how it did — results can move the price."
            ),
            "insider_selling": "Some company insiders have been selling their own shares.",
            "insider_buying": "Some company insiders have been buying their own shares.",
            "high_short_interest": "A lot of investors are betting against the stock.",
            "analyst_downgrade": "Some experts recently became less positive about it.",
            "analyst_upgrade": "Some experts recently became more positive about it.",
            "litigation": "The company is dealing with a lawsuit.",
            "filing_risk_factor": (
                "The company's official yearly report lists risks to its business."
            ),
        }
        for fl in r.risks.flags:
            out.append(f"- {plain_flag.get(fl.flag_type, fl.description)}")
    out.append("")

    # --- helper scores ---
    out.append("## Rough helper scores (0–100)\n")
    out.append(
        "_A higher number is not a verdict on the company — each bar just "
        "measures one single thing._\n"
    )
    for label, score, what in [
        ("Price vs. value", s.valuation, "low price for what it earns → higher"),
        ("Growth", s.growth, "sales getting bigger faster → higher"),
        ("Money health", s.financial_health, "can pay its bills easily → higher"),
        ("Recent price trend", s.momentum, "price drifting up lately → higher"),
    ]:
        out.append(f"- **{label}** {_bar(score)}  \n  _{what}_")
    out.append("")

    if diff:
        out.append("## What changed since last time\n")
        for d in diff:
            out.append(f"- {d}")
        out.append("")

    out.append("---")
    out.append(f"_{r.disclaimer}_")
    if r.qa_violations:
        out.append(
            f"\n_(Note for grown-ups: {len(r.qa_violations)} automatic fact-check "
            f"flag(s) on this run — see the full report.)_"
        )
    return "\n".join(out) + "\n"
