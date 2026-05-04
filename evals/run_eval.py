"""Day 10 eval runner.

Runs every question in eval_set_v1.json through the full AMIA pipeline,
scores retrieval (did the right source types come back?) and reasoning
(did the briefing mention the expected keywords?), and writes a
date-stamped results file.

This is v1 scoring. Keyword overlap is cheap and noisy. Day 11 can swap
in an LLM judge if the signal is too weak.
"""
import json
import time
from pathlib import Path
from datetime import datetime

# add project root to path so we can import graph.py from inside evals/
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph import run_for_question
from tracing import flush as flush_tracing

EVAL_PATH = Path(__file__).parent / "eval_set_v1.json"
RESULTS_PATH = Path(__file__).parent / f"results_{datetime.now().strftime('%Y%m%d')}.json"

# Sleep between questions so we do not trip Groq's rate limit running 10 back to back.
# Groq free tier on llama-3.3-70b-versatile: 30 RPM, 6k TPM, 100k TPD.
# Each question fires planner + crew, ~5-8 calls and ~10k tokens per question.
# 10s gives TPM headroom. Day 12 also wired a Gemini fallback in crew.py so a
# TPD hit no longer kills the run mid-way.
SLEEP_BETWEEN_QUESTIONS = 10


def categorise_source(source_dict: dict) -> str:
    """Map a source dict from retrieve_with_sources to one of the eval categories.

    The eval file uses "news", "stocktwits", "hackernews".
    Real source dicts have:
      - source: publisher name for news (e.g. "Reuters") OR "stocktwits" / "hackernews"
      - data_type: "news" or "social"
    So we use data_type for news, and source for the social split.
    """
    data_type = source_dict.get("data_type", "")
    source = source_dict.get("source", "")
    if data_type == "news":
        return "news"
    if source == "stocktwits":
        return "stocktwits"
    if source == "hackernews":
        return "hackernews"
    return "unknown"


def score_retrieval(retrieved_sources: list[dict], expected_sources: list[str]) -> bool:
    """All expected source categories must appear at least once in retrieval."""
    found = {categorise_source(s) for s in retrieved_sources}
    return all(src in found for src in expected_sources)


def score_reasoning(answer_text: str, expected_keywords: list[str]) -> float:
    """Cheap keyword overlap. 0.0 to 1.0. False positives possible (e.g. 'not bullish')."""
    text_lower = answer_text.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in text_lower)
    return hits / len(expected_keywords) if expected_keywords else 0.0


def run_eval():
    questions = json.loads(EVAL_PATH.read_text())
    results = []

    for i, q in enumerate(questions, start=1):
        print(f"\n[{i}/{len(questions)}] {q['id']} ({q['category']}): {q['question']}")

        try:
            output = run_for_question(q["question"])
            briefing = output["briefing"]
            retrieved = output["retrieved_sources"]
            ticker = output["ticker"]
            error = None
        except Exception as e:
            print(f"  ERROR: {e}")
            briefing = ""
            retrieved = []
            ticker = "?"
            error = str(e)

        retrieval_pass = score_retrieval(retrieved, q["expected_sources"])
        reasoning_score = score_reasoning(briefing, q["expected_keywords"])
        found_categories = sorted({categorise_source(s) for s in retrieved})

        print(f"  ticker={ticker} retrieval={retrieval_pass} reasoning={reasoning_score:.2f} "
              f"sources_found={found_categories}")

        results.append({
            "id": q["id"],
            "category": q["category"],
            "ticker_used": ticker,
            "expected_sources": q["expected_sources"],
            "found_source_categories": found_categories,
            "retrieval_pass": retrieval_pass,
            "reasoning_score": reasoning_score,
            "expected_keywords": q["expected_keywords"],
            "briefing": briefing,
            "error": error,
        })

        # be polite to Groq's rate limiter
        if i < len(questions):
            time.sleep(SLEEP_BETWEEN_QUESTIONS)

    # aggregates
    total = len(results)
    retrieval_rate = sum(r["retrieval_pass"] for r in results) / total
    reasoning_avg = sum(r["reasoning_score"] for r in results) / total

    # split by category so we can see if cross-source is dragging things down
    news_only = [r for r in results if r["category"] == "news_only"]
    cross = [r for r in results if r["category"] == "cross_source"]

    def _avg(items, key):
        return sum(item[key] for item in items) / len(items) if items else 0.0

    summary = {
        "run_at": datetime.now().isoformat(),
        "total_questions": total,
        "retrieval_pass_rate": round(retrieval_rate, 3),
        "reasoning_avg": round(reasoning_avg, 3),
        "news_only": {
            "n": len(news_only),
            "retrieval_pass_rate": round(_avg(news_only, "retrieval_pass"), 3),
            "reasoning_avg": round(_avg(news_only, "reasoning_score"), 3),
        },
        "cross_source": {
            "n": len(cross),
            "retrieval_pass_rate": round(_avg(cross, "retrieval_pass"), 3),
            "reasoning_avg": round(_avg(cross, "reasoning_score"), 3),
        },
        "results": results,
    }

    RESULTS_PATH.write_text(json.dumps(summary, indent=2))

    # pretty print
    print("\n" + "=" * 60)
    print("EVAL SUMMARY")
    print("=" * 60)
    print(f"Overall:      retrieval={retrieval_rate:.0%}  reasoning={reasoning_avg:.0%}")
    print(f"News-only:    retrieval={summary['news_only']['retrieval_pass_rate']:.0%}  "
          f"reasoning={summary['news_only']['reasoning_avg']:.0%}")
    print(f"Cross-source: retrieval={summary['cross_source']['retrieval_pass_rate']:.0%}  "
          f"reasoning={summary['cross_source']['reasoning_avg']:.0%}")
    print(f"\nResults written to: {RESULTS_PATH}")


if __name__ == "__main__":
    try:
        run_eval()
    finally:
        # make sure langfuse uploads before process exits
        flush_tracing()
