"""Day 12 task 4: per-provider token usage logging.

Writes one JSONL line per LLM call to logs/token_usage.jsonl. Lets you eyeball
"how close am I to Groq's 100k TPD" without hitting Groq's limits API.

Wired in two places:
  1. graph.py planner: explicit log() call inside _invoke_with_fallback
  2. crew.py: litellm.success_callback fires on every CrewAI agent call

Run `python usage_log.py` to print today's totals per provider.
"""
import json
from datetime import datetime
from pathlib import Path
from threading import Lock
from collections import defaultdict

LOG_PATH = Path("logs/token_usage.jsonl")
_lock = Lock()


def log(
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    source: str = "",
) -> None:
    """Append one usage record. Cheap, file-locked, no deps."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now().isoformat(),
        "provider": provider,
        "model": model,
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "total_tokens": int((prompt_tokens or 0) + (completion_tokens or 0)),
        "source": source,
    }
    with _lock:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")


def litellm_callback(kwargs, completion_response, start_time, end_time) -> None:
    """litellm.success_callback signature. Crew agent calls land here."""
    try:
        model = kwargs.get("model", "?")
        provider = model.split("/")[0] if "/" in model else "?"
        usage = getattr(completion_response, "usage", None)
        if usage is None:
            return
        log(
            provider=provider,
            model=model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
            source="crew",
        )
    except Exception:
        # never break the call path on a logging hiccup
        pass


def daily_summary(date_str: str | None = None) -> dict:
    """Return {(provider, model): total_tokens} for the given YYYY-MM-DD."""
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")
    if not LOG_PATH.exists():
        return {}
    totals: dict[tuple[str, str], int] = defaultdict(int)
    for line in LOG_PATH.read_text().splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not e["ts"].startswith(date_str):
            continue
        key = (e["provider"], e["model"])
        totals[key] += e["total_tokens"]
    return dict(totals)


# free tier daily limits, used for the % bar in the CLI summary.
# Day 13: model split means we now hit 4 different Groq endpoints. Numbers
# below reflect Groq free tier as of May 2026; verify on the dashboard if
# the % bar starts looking off.
KNOWN_LIMITS = {
    ("groq", "groq/openai/gpt-oss-120b"): 200_000,
    ("groq", "openai/gpt-oss-120b"): 200_000,  # planner logs without prefix
    ("groq", "groq/meta-llama/llama-4-scout-17b-16e-instruct"): 500_000,
    ("groq", "groq/qwen/qwen3-32b"): 500_000,
    ("groq", "groq/llama-3.1-8b-instant"): 500_000,
    ("groq", "llama-3.1-8b-instant"): 500_000,  # subquery logs without prefix
    # legacy keys kept so historical entries still render with a bar
    ("groq", "groq/llama-3.3-70b-versatile"): 100_000,
    # gemini free tier is 1500 req/day, no documented TPD; show usage only
}


def print_summary(date_str: str | None = None) -> None:
    """Pretty-print today's usage with bar against known free-tier limits."""
    totals = daily_summary(date_str)
    if not totals:
        print("No usage logged today.")
        return
    print(f"\nToken usage for {date_str or datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 60)
    for (provider, model), tokens in sorted(totals.items()):
        limit = KNOWN_LIMITS.get((provider, model))
        if limit:
            pct = tokens / limit
            bar = "#" * int(pct * 30) + "." * (30 - int(pct * 30))
            print(f"{model:50} {tokens:>7} / {limit} [{bar}] {pct:.0%}")
        else:
            print(f"{model:50} {tokens:>7} tokens (no known limit)")


if __name__ == "__main__":
    import sys
    date = sys.argv[1] if len(sys.argv) > 1 else None
    print_summary(date)
