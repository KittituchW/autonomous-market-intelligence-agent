"""Parse briefings/YYYY-MM-DD/*.md into structured dicts for delivery.

The CrewAI writer produces markdown with a known shape:

    # AAPL Daily Briefing
    _Generated: 2026-05-01_
    ---
    **Summary:** ...
    **Key Signals:** ...
    **Watch:** ...
    **Sources:** ...

We split that into fields so email and Notion workflows can render proper
sections instead of dumping the whole blob. If the writer ever drifts off
template, body=full_markdown is the safety net.
"""
import os
import re
from datetime import datetime

from amia.config import TICKERS

BRIEFINGS_DIR = "briefings"


def _split_sections(body: str) -> dict:
    """Split a briefing body on bold-heading markers like **Summary:** or **Watch:**.

    We use a regex that matches the start of a line containing **Word:** so we
    don't accidentally split on inline bold. Returns a dict of section_name -> text.
    """
    # find every bold heading and its position
    pattern = re.compile(r"^\s*\*\*([A-Za-z ]+):\*\*\s*", re.MULTILINE)
    matches = list(pattern.finditer(body))

    sections = {}
    for i, match in enumerate(matches):
        name = match.group(1).strip().lower().replace(" ", "_")
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections[name] = body[start:end].strip()

    return sections


def parse_briefing_file(path: str) -> dict:
    """Read one briefing markdown file, return a dict ready for JSON."""
    with open(path, "r") as f:
        raw = f.read()

    # ticker is the first line after the # heading
    ticker_match = re.search(r"^#\s+([A-Z]+)\s+Daily Briefing", raw, re.MULTILINE)
    ticker = ticker_match.group(1) if ticker_match else os.path.basename(path).replace(".md", "")

    # everything after the first --- divider is the briefing body
    parts = raw.split("---", 1)
    body = parts[1].strip() if len(parts) > 1 else raw

    sections = _split_sections(body)

    # one-line preview for email subject lines / Notion title hints.
    # take the first sentence of summary, capped at 140 chars.
    summary = sections.get("summary", "")
    first_sentence = summary.split(". ")[0].strip()
    preview = (first_sentence[:137] + "...") if len(first_sentence) > 140 else first_sentence

    return {
        "ticker": ticker,
        "title": f"{ticker} Daily Briefing",
        "preview": preview,
        "summary": sections.get("summary", ""),
        "key_signals": sections.get("key_signals", ""),
        "watch": sections.get("watch", ""),
        "sources": sections.get("sources", ""),
        "body_markdown": body,
    }


def load_today(run_date: str | None = None) -> dict:
    """Load all briefings for a given date (default: today). Used by /deliver.

    Returns:
        {
            "date": "2026-05-01",
            "count": 5,
            "briefings": [ {ticker, title, preview, summary, key_signals, watch, sources, body_markdown}, ... ]
        }
    """
    if run_date is None:
        run_date = datetime.now().strftime("%Y-%m-%d")

    day_dir = os.path.join(BRIEFINGS_DIR, run_date)
    if not os.path.isdir(day_dir):
        return {"date": run_date, "count": 0, "briefings": [], "error": f"no folder {day_dir}"}

    briefings = []
    # iterate in TICKERS order so emails are always AAPL, TSLA, NVDA, MSFT, AMZN
    for ticker in TICKERS:
        path = os.path.join(day_dir, f"{ticker}.md")
        try:
            briefings.append(parse_briefing_file(path))
        except FileNotFoundError:
            continue

    return {
        "date": run_date,
        "count": len(briefings),
        "briefings": briefings,
    }


def build_email_html(payload: dict) -> str:
    """Compose a single HTML email body from the day's briefings.

    Plain HTML on purpose: Gmail strips most CSS, and we want this to be
    readable on phone too. Each briefing gets a card-style block with the
    ticker as a heading and the four sections as paragraphs.
    """
    date = payload["date"]
    briefings = payload["briefings"]

    if not briefings:
        return f"<p>No briefings found for {date}.</p>"

    # table of contents at the top for quick jump to a ticker
    toc = " &nbsp;|&nbsp; ".join(
        f'<a href="#{b["ticker"]}" style="color:#0a66c2;text-decoration:none;">{b["ticker"]}</a>'
        for b in briefings
    )

    blocks = []
    for b in briefings:
        block = f"""
        <div style="margin:24px 0;padding:16px;border:1px solid #eee;border-radius:8px;">
          <h2 id="{b['ticker']}" style="margin:0 0 4px 0;font-size:20px;">{b['ticker']}</h2>
          <p style="color:#666;font-size:13px;margin:0 0 12px 0;">{b['preview']}</p>

          <p><strong>Summary</strong><br>{b['summary'].replace(chr(10), '<br>')}</p>
          <p><strong>Key signals</strong><br>{b['key_signals'].replace(chr(10), '<br>')}</p>
          <p><strong>Watch</strong><br>{b['watch'].replace(chr(10), '<br>')}</p>
          <p style="font-size:12px;color:#888;"><strong>Sources</strong><br>{b['sources'].replace(chr(10), '<br>')}</p>
        </div>
        """
        blocks.append(block)

    html = f"""
    <div style="font-family:-apple-system,Helvetica,Arial,sans-serif;max-width:680px;margin:auto;color:#222;">
      <h1 style="font-size:24px;margin-bottom:4px;">AMIA Daily Briefing</h1>
      <p style="color:#666;margin-top:0;">{date} &middot; {len(briefings)} tickers</p>
      <p>{toc}</p>
      {''.join(blocks)}
      <hr>
      <p style="font-size:12px;color:#888;">Generated by AMIA. News, StockTwits, HackerNews. LangGraph + CrewAI + local Gemma.</p>
    </div>
    """
    return html


def main() -> None:
    # quick CLI: python digest.py [YYYY-MM-DD]
    import sys
    import json

    date_arg = sys.argv[1] if len(sys.argv) > 1 else None
    payload = load_today(date_arg)
    print(json.dumps(payload, indent=2)[:2000])
    print(f"\n--- {payload['count']} briefings loaded ---")


if __name__ == "__main__":
    main()
