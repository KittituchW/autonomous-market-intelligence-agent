"""Per-provider token usage logging.

Writes one JSONL line per LLM call to logs/token_usage.jsonl so you can
eyeball "how close am I to Groq's free-tier TPD" without hitting Groq's
limits API. Wired in two places:
  1. graph.py planner: explicit log() call inside _invoke_with_fallback
  2. crew.py: litellm.success_callback fires on every CrewAI agent call

Run `python -m amia.observability.usage_log` to print today's totals.
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
    with open(LOG_PATH) as f:
        for line in f:
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


# Groq free-tier daily token limits, keyed without the leading "groq/" so
# the planner (which logs bare ids) and crew (which logs prefixed ids) both
# resolve to the same limit. Verify on the Groq dashboard if % bars look off.
KNOWN_LIMITS = {
    ("groq", "openai/gpt-oss-120b"): 200_000,
    ("groq", "meta-llama/llama-4-scout-17b-16e-instruct"): 500_000,
    ("groq", "qwen/qwen3-32b"): 500_000,
    ("groq", "llama-3.1-8b-instant"): 500_000,
    # Legacy key, kept so historical entries still render with a bar.
    ("groq", "llama-3.3-70b-versatile"): 100_000,
    # Gemini free tier is 1500 req/day, no documented TPD; show usage only.
}


def _lookup_limit(provider: str, model: str) -> int | None:
    bare = model.split("/", 1)[1] if model.startswith(f"{provider}/") else model
    return KNOWN_LIMITS.get((provider, bare))


def print_summary(date_str: str | None = None) -> None:
    """Pretty-print today's usage with bar against known free-tier limits."""
    totals = daily_summary(date_str)
    if not totals:
        print("No usage logged today.")
        return
    print(f"\nToken usage for {date_str or datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 60)
    for (provider, model), tokens in sorted(totals.items()):
        limit = _lookup_limit(provider, model)
        if limit:
            pct = tokens / limit
            bar = "#" * int(pct * 30) + "." * (30 - int(pct * 30))
            print(f"{model:50} {tokens:>7} / {limit} [{bar}] {pct:.0%}")
        else:
            print(f"{model:50} {tokens:>7} tokens (no known limit)")


def main() -> None:
    import sys
    date = sys.argv[1] if len(sys.argv) > 1 else None
    print_summary(date)


if __name__ == "__main__":
    main()
