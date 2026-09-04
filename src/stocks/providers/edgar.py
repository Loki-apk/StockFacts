"""SEC EDGAR filings via ``edgartools``.

Free, authoritative. The SEC requires a descriptive identity ("name email"),
taken from ``SEC_USER_AGENT``. Used for real risk factors (10-K Item 1A) and
MD&A (Item 7) rather than guessing at risks.
"""

from __future__ import annotations

import os
import re
from datetime import date
from typing import Any

_IDENTITY = os.environ.get("SEC_USER_AGENT", "stocks-eval you@example.com")
_MAX_SECTION_CHARS = 8000

_identity_set = False


def _ensure_identity() -> None:
    global _identity_set
    if not _identity_set:
        from edgar import set_identity

        set_identity(_IDENTITY)
        _identity_set = True


def _latest_10k(ticker: str):
    _ensure_identity()
    from edgar import Company

    filing = Company(ticker).get_filings(form="10-K").latest(1)
    if filing is None:
        raise LookupError(f"no 10-K on file for {ticker!r}")
    return filing


def _clean(text: str | None) -> str:
    return re.sub(r"[ \t]*\n[ \t]*", "\n", re.sub(r"[ \t]+", " ", text or "")).strip()


def risk_factor_excerpts(ticker: str) -> list[dict[str, Any]]:
    """Latest 10-K -> Item 1A + MD&A excerpts, shaped for ``FilingExcerpt``.

    Each section is truncated to ~8k chars — enough to ground the LLM's risk
    wording, not the whole document.
    """
    filing = _latest_10k(ticker)
    filed_at = _filed_date(filing)
    url = getattr(filing, "filing_url", None) or getattr(filing, "homepage_url", "")
    tenk = filing.obj()

    sections = {
        "Item 1A": _clean(getattr(tenk, "risk_factors", None)),
        "MD&A": _clean(getattr(tenk, "management_discussion", None)),
    }
    out: list[dict[str, Any]] = []
    for label, body in sections.items():
        if len(body) < 400:
            continue
        out.append(
            {
                "filing_type": "10-K",
                "section": label,
                "filed_at": filed_at,
                "url": str(url) or "https://www.sec.gov/",
                "text": body[:_MAX_SECTION_CHARS],
            }
        )
    return out


def _filed_date(filing) -> str:
    d = getattr(filing, "filing_date", None)
    if isinstance(d, date):
        return d.isoformat()
    return str(d)[:10] if d else date.today().isoformat()


def find_relationship_mention(candidate_ticker: str, needle: str) -> dict[str, Any] | None:
    """Grep the candidate's latest 10-K for ``needle`` (a short form of another
    company's name). -> ``{quote, url, filed_at}`` on the first hit, else
    ``None``.

    Fully deterministic: ``Filing.grep`` does a literal case-insensitive text
    search and returns the real surrounding text, verbatim — nothing here is
    summarized, inferred, or LLM-generated. This is what backs
    ``RelatedExposure.evidence_quote``: a customer/supplier/partner claim only
    survives if the searched company's name is actually named in the
    candidate's own filing.
    """
    filing = _latest_10k(candidate_ticker)
    matches = filing.grep(needle)
    if not matches:
        return None
    quote = _clean(matches[0].context)
    url = str(getattr(filing, "filing_url", None) or getattr(filing, "homepage_url", "")) or "https://www.sec.gov/"
    return {"quote": quote, "url": url, "filed_at": _filed_date(filing)}


# --- back-compat shims for earlier call sites -------------------------------

def latest_filing(ticker: str, form: str = "10-K") -> dict[str, Any]:
    if form != "10-K":
        raise NotImplementedError("only 10-K is wired")
    filing = _latest_10k(ticker)
    return {
        "filed_at": _filed_date(filing),
        "primary_doc_url": str(getattr(filing, "filing_url", "")),
        "_filing": filing,
    }


def extract_sections(doc_url: str, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:  # noqa: ARG001
    raise NotImplementedError("use risk_factor_excerpts(ticker)")
