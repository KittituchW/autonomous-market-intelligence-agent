"""Quality gates and deterministic fallback for generated briefings.

The LLM crew is allowed to be creative in synthesis, but the saved briefing
needs boring production guarantees: stable sections, valid citations, no hidden
reasoning text, and source URLs copied from retrieval rather than generated.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


REQUIRED_SECTIONS = ("Summary", "Key Signals", "Watch", "Sources")

_CITATION_RE = re.compile(r"\[(\d+)\]")
_SOURCES_HEADING_RE = re.compile(
    r"^\s*(?:#+\s*)?\**\s*sources?\s*:?\s*\**\s*$", re.IGNORECASE
)
_THINK_BLOCK_RE = re.compile(
    r"<think\b[^>]*>.*?</think>",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class BriefingQuality:
    ok: bool
    warnings: list[str]


def strip_reasoning(text: str) -> str:
    """Remove model reasoning artifacts that should never land in a briefing."""
    cleaned = _THINK_BLOCK_RE.sub("", text or "")
    cleaned = re.sub(r"^\s*Final Answer:\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def split_sources_block(briefing: str) -> tuple[str, str]:
    """Split a briefing into body and source block at the first Sources heading."""
    lines = (briefing or "").splitlines()
    for idx, line in enumerate(lines):
        if _SOURCES_HEADING_RE.match(line.strip()):
            body = "\n".join(lines[:idx]).rstrip()
            source_block = "\n".join(lines[idx:]).strip()
            return body, source_block
    return (briefing or "").rstrip(), ""


def cited_source_ids(briefing: str, sources: list[dict]) -> list[int]:
    """Return valid source ids cited in the briefing body, preserving order."""
    body, _ = split_sources_block(briefing)
    valid_ids = {int(s["id"]) for s in sources if str(s.get("id", "")).isdigit()}
    seen: set[int] = set()
    ids: list[int] = []
    for match in _CITATION_RE.finditer(body):
        source_id = int(match.group(1))
        if source_id in valid_ids and source_id not in seen:
            seen.add(source_id)
            ids.append(source_id)
    return ids


def _source_line(source: dict) -> str:
    sentiment = f" [{source['sentiment']}]" if source.get("sentiment") else ""
    return (
        f"{source.get('id')}. [{source.get('ticker', '?')}/"
        f"{source.get('source', '?')}]{sentiment} {source.get('url', '')}"
    ).rstrip()


def replace_sources_block(briefing: str, sources: list[dict]) -> str:
    """Replace any generated Sources block with retrieval-backed source URLs.

    Only cited sources are listed. If the body has no valid citations, use the
    first three retrieved sources so the file still points to evidence.
    """
    if not sources:
        return strip_reasoning(briefing)

    cleaned = strip_reasoning(briefing)
    body, _ = split_sources_block(cleaned)
    ids = cited_source_ids(body, sources)
    if not ids:
        ids = [int(s["id"]) for s in sources[:3] if str(s.get("id", "")).isdigit()]

    by_id = {int(s["id"]): s for s in sources if str(s.get("id", "")).isdigit()}
    real_lines = [
        _source_line(by_id[source_id])
        for source_id in ids
        if source_id in by_id
    ]
    if not real_lines:
        return body

    return body.rstrip() + "\n\n**Sources:**\n" + "\n".join(real_lines)


def validate_briefing(briefing: str, sources: list[dict]) -> BriefingQuality:
    """Check whether a saved briefing meets minimum production invariants."""
    warnings: list[str] = []
    text = briefing or ""
    lower = text.lower()

    for section in REQUIRED_SECTIONS:
        if section.lower() not in lower:
            warnings.append(f"missing section: {section}")

    if "<think" in lower or "</think>" in lower:
        warnings.append("reasoning artifact leaked")

    valid_ids = {int(s["id"]) for s in sources if str(s.get("id", "")).isdigit()}
    body, source_block = split_sources_block(text)
    body_ids = [int(m.group(1)) for m in _CITATION_RE.finditer(body)]
    invalid_ids = sorted(
        {source_id for source_id in body_ids if source_id not in valid_ids}
    )
    if invalid_ids:
        warnings.append(f"invalid citation ids: {invalid_ids}")
    if sources and not body_ids:
        warnings.append("no body citations")
    if sources and not source_block:
        warnings.append("missing sources block")

    by_id = {
        int(s["id"]): s for s in sources if str(s.get("id", "")).isdigit()
    }
    news_source_ids = {
        source_id
        for source_id, source in by_id.items()
        if source.get("data_type") == "news"
    }
    social_source_ids = {
        source_id
        for source_id, source in by_id.items()
        if source.get("data_type") == "social"
    }
    news_line_ids = _line_citation_ids(body, "News")
    retail_line_ids = _line_citation_ids(body, "Retail sentiment")
    if news_source_ids and not news_line_ids:
        warnings.append("news signal missing citation")
    elif news_source_ids and not any(i in news_source_ids for i in news_line_ids):
        warnings.append("news signal lacks news-source citation")
    if social_source_ids and not retail_line_ids:
        warnings.append("retail sentiment signal missing citation")
    elif social_source_ids and not any(
        i in social_source_ids for i in retail_line_ids
    ):
        warnings.append("retail sentiment signal lacks social-source citation")

    return BriefingQuality(ok=not warnings, warnings=warnings)


def _line_citation_ids(body: str, label: str) -> list[int]:
    pattern = re.compile(
        rf"^\s*-\s*{re.escape(label)}\s*:\s*(.+)$",
        re.IGNORECASE | re.MULTILINE,
    )
    match = pattern.search(body or "")
    if not match:
        return []
    return [int(m.group(1)) for m in _CITATION_RE.finditer(match.group(1))]


def _first_source(sources: list[dict], *, data_type: str | None = None) -> dict | None:
    for source in sources:
        if data_type is None or source.get("data_type") == data_type:
            return source
    return None


def _source_summary(source: dict | None, fallback: str) -> str:
    if not source:
        return fallback
    snippet = (source.get("snippet") or "").strip().replace("\n", " ")
    if len(snippet) > 180:
        snippet = snippet[:177].rstrip() + "..."
    return snippet or fallback


def _sentiment_phrase(sources: list[dict]) -> str:
    counts = Counter(
        s.get("sentiment")
        for s in sources
        if s.get("data_type") == "social" and s.get("sentiment")
    )
    if not counts:
        return "no explicit retrieved social sentiment label"
    parts = [f"{count} {sentiment}" for sentiment, count in sorted(counts.items())]
    return ", ".join(parts)


def build_fallback_briefing(
    ticker: str,
    sources: list[dict],
    reason: str | None = None,
) -> str:
    """Build an evidence-only briefing when the LLM crew fails quality gates."""
    news = _first_source(sources, data_type="news")
    social = _first_source(sources, data_type="social")
    risk_source = news or social

    cited = [source for source in (news, social, risk_source) if source]
    seen: set[int] = set()
    cited_unique = []
    for source in cited:
        source_id = int(source.get("id", 0))
        if source_id and source_id not in seen:
            seen.add(source_id)
            cited_unique.append(source)
    if not cited_unique:
        cited_unique = sources[:3]

    citation_ids = [
        int(s["id"]) for s in cited_unique if str(s.get("id", "")).isdigit()
    ]
    first_citation = f" [{citation_ids[0]}]" if citation_ids else ""
    news_citation = f" [{news['id']}]" if news else ""
    social_citation = f" [{social['id']}]" if social else ""
    risk_citation = f" [{risk_source['id']}]" if risk_source else ""

    note = (
        " The LLM crew output failed validation, so this fallback uses retrieved "
        "snippets only."
        if reason
        else ""
    )
    summary = (
        f"{ticker} briefing generated from retrieval-backed evidence only.{note} "
        "The most relevant retrieved item says: "
        f"{_source_summary(cited_unique[0] if cited_unique else None, 'no usable snippet was retrieved')}"
        f"{first_citation}"
    )

    body = "\n\n".join(
        [
            f"**Summary:**\n{summary}",
            "**Key Signals:**\n"
            "- News: "
            f"{_source_summary(news, 'No news source was retrieved for this run.')}"
            f"{news_citation}\n"
            "- Retail sentiment: Retrieved social labels show "
            f"{_sentiment_phrase(sources)}.{social_citation}\n"
            "- Risks: Evidence quality is limited to the retrieved snippets; "
            "avoid treating this as a full investment thesis without newer "
            f"primary sources.{risk_citation}",
            "**Watch:** Track fresh company filings, earnings commentary, and "
            f"updated source retrieval before acting on {ticker}.",
        ]
    )
    if not cited_unique:
        return body + "\n\n**Sources:**\nNo retrieved sources."
    return replace_sources_block(body, cited_unique)
