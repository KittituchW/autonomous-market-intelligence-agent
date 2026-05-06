"""Day 3 LangGraph: plan -> retrieve -> write (CrewAI 4-agent crew), with one retry on empty retrieval.

Day 11 updates:
  - retrieve_node now passes ticker to retrieve_with_sources for the metadata filter
    + StockTwits/HackerNews quotas. Stops the cross-ticker leakage the Day 9
    trace caught (AMZN run getting NVDA / MSFT articles).
  - write_node now strips the writer's hallucinated Sources block and replaces
    it with the real URLs from state["sources"]. The Gemma 3 1B writer was
    inventing Bloomberg/Reuters URLs from scratch; this kills that for good.
"""
import os
import re
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from retrieval import search_news, retrieve_with_sources
from crew import build_crew
from usage_log import log as _usage_log

load_dotenv()

# planner uses Groq directly. The crew handles its own LLMs in the write node.
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

# Day 12: planner fallback. langchain_groq raises groq.RateLimitError on 429,
# but litellm-style errors leak through too, so we match on the message.
# Same Gemini model the crew critic uses.
fallback_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)


def _invoke_with_fallback(prompt: str) -> str:
    """Call the planner LLM, fall back to Gemini on any rate limit / 429.

    The crew has its own fallback wired via litellm globals in crew.py.
    This function only covers the plan_node call which goes through
    langchain_groq, not litellm.

    Day 12 task 4: every call (success OR fallback) is logged to
    logs/token_usage.jsonl so `python usage_log.py` can show daily totals.
    """
    try:
        resp = llm.invoke(prompt)
        usage = resp.response_metadata.get("token_usage", {}) or {}
        _usage_log(
            provider="groq",
            model="llama-3.3-70b-versatile",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            source="planner",
        )
        return resp.content
    except Exception as e:
        msg = str(e).lower()
        is_rate_limit = (
            "rate_limit" in msg
            or "ratelimit" in msg
            or "429" in msg
            or "tpd" in msg
            or "tpm" in msg
        )
        if not is_rate_limit:
            raise
        print("[plan] Groq rate limited, falling back to Gemini")
        resp = fallback_llm.invoke(prompt)
        usage = (resp.response_metadata or {}).get("usage_metadata", {}) or {}
        _usage_log(
            provider="gemini",
            model="gemini-2.0-flash",
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            source="planner-fallback",
        )
        return resp.content


class State(TypedDict):
    ticker: str
    plan: str
    context: str
    sources: list  # structured citations passed into the crew
    briefing: str
    retries: int  # tracks how many times retrieve has been re-run


def plan_node(state: State) -> dict:
    """Ask the LLM what to research about this ticker. Plan drives retrieval."""
    prompt = (
        f"You are a market analyst. List 3 specific questions to research "
        f"about {state['ticker']} today. Cover news, retail sentiment, and "
        f"technical/product developments. Output as a numbered list, nothing else."
    )
    plan = _invoke_with_fallback(prompt)
    print(f"\n[plan] {plan}\n")
    return {"plan": plan}


def retrieve_node(state: State) -> dict:
    """Run the plan as a retrieval query, return both formatted text and structured sources.

    Day 11: passes ticker so retrieve_with_sources hard-filters on metadata
    AND tops up with StockTwits + HackerNews quotas.
    """
    query = f"{state['ticker']} {state['plan']}"
    context, sources = retrieve_with_sources(
        query, ticker=state["ticker"], top_k=8
    )
    source_kinds = {s.get("source") for s in sources}
    data_kinds = {s.get("data_type") for s in sources}
    print(
        f"\n[retrieve] {len(context)} chars, {len(sources)} sources, "
        f"data_types={data_kinds}, sources={source_kinds}\n"
    )
    return {
        "context": context,
        "sources": sources,
        "retries": state.get("retries", 0),
    }


# Matches a Sources/Source heading line in any of the forms the writer emits:
#   "**Sources:**", "Sources:", "**Source**", "## Sources", etc. Case-insensitive.
_SOURCES_HEADING = re.compile(
    r"^\s*(?:#+\s*)?\**\s*sources?\s*:?\s*\**\s*$", re.IGNORECASE
)


def _replace_sources_block(briefing: str, sources: list[dict]) -> str:
    """Strip whatever Sources block the writer produced, append the real URLs.

    The Gemma 3 1B writer kept hallucinating Bloomberg / Reuters URLs that did
    not exist in retrieval. The real URLs are sitting right there in
    state["sources"]. So we just overwrite.
    """
    if not sources:
        return briefing

    real_lines = []
    for s in sources:
        url = s.get("url")
        if not url:
            continue
        sentiment = f" [{s['sentiment']}]" if s.get("sentiment") else ""
        real_lines.append(
            f"{s['id']}. [{s.get('ticker', '?')}/{s.get('source', '?')}]"
            f"{sentiment} {url}"
        )
    if not real_lines:
        return briefing
    real_block = "**Sources:**\n" + "\n".join(real_lines)

    # cut from the FIRST Sources heading onward (writer often emits it twice)
    lines = briefing.split("\n")
    cut_idx = None
    for i, line in enumerate(lines):
        if _SOURCES_HEADING.match(line.strip()):
            cut_idx = i
            break
    if cut_idx is not None:
        lines = lines[:cut_idx]
    cleaned = "\n".join(lines).rstrip()
    return cleaned + "\n\n" + real_block


def write_node(state: State) -> dict:
    """Hand off to the CrewAI crew, then post-process URLs.

    Day 11: the writer cannot be trusted to copy URLs verbatim, so we replace
    its Sources block with the real URLs from retrieval before saving.
    """
    crew = build_crew(
        ticker=state["ticker"],
        context=state["context"],
        sources=state["sources"],
    )
    # kickoff runs research -> analyst -> critic -> writer in sequence
    result = crew.kickoff()
    # CrewAI 0.51 returns a CrewOutput object, get the raw string
    raw = str(result)
    cleaned = _replace_sources_block(raw, state.get("sources", []))
    return {"briefing": cleaned}


def should_retry(state: State) -> str:
    """If retrieval came back empty and we have not retried yet, loop back to plan once."""
    has_context = bool(state.get("context", "").strip())
    retries = state.get("retries", 0)
    if not has_context and retries < 1:
        print("[router] empty context, retrying plan...")
        return "retry"
    return "ok"


def increment_retry(state: State) -> dict:
    """Bump the retry counter so we do not loop forever."""
    return {"retries": state.get("retries", 0) + 1}


# build the graph
graph = StateGraph(State)
graph.add_node("plan", plan_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("write", write_node)
graph.add_node("bump_retry", increment_retry)

graph.set_entry_point("plan")
graph.add_edge("plan", "retrieve")
# conditional edge: empty context routes back to bump_retry -> plan, else to write
graph.add_conditional_edges(
    "retrieve",
    should_retry,
    {"retry": "bump_retry", "ok": "write"},
)
graph.add_edge("bump_retry", "plan")
graph.add_edge("write", END)

app = graph.compile()


# tickers AMIA covers, used by run_for_question to detect which ticker to run
TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN"]


def run_for_question(question: str, default_ticker: str = "AAPL") -> dict:
    """Eval-friendly entry point.

    The graph itself is ticker-driven (the planner generates its own questions
    inside the graph). For Day 10 evals we want to feed in a free-form
    question, still run the full pipeline, and get back both the briefing and
    the structured sources retrieve_with_sources produced.

    Behavior:
        1. Scan the question text for one of the 5 known tickers
        2. Fall back to default_ticker if none found
        3. Run the graph with langfuse session_id "amia-eval" so eval traces
           are grouped separately from the daily runs

    Returns:
        {
          "ticker": detected ticker,
          "briefing": final briefing text,
          "retrieved_sources": list[dict] from retrieve_with_sources,
        }
    """
    upper_q = question.upper()
    detected = next((t for t in TICKERS if t in upper_q), default_ticker)

    # local import so eval still works in environments where langfuse is off
    from tracing import build_config

    config = build_config(ticker=detected, session_id="amia-eval")
    result = app.invoke(
        {"ticker": detected, "retries": 0, "sources": []},
        config=config,
    )
    return {
        "ticker": detected,
        "briefing": result.get("briefing", ""),
        "retrieved_sources": result.get("sources", []),
    }


if __name__ == "__main__":
    # smoke test on one ticker, full crew runs inside write_node.
    # Day 9: build_config returns callbacks + Langfuse metadata so this
    # smoke run shows up grouped in the Langfuse UI. flush at the end so
    # the trace actually uploads before python exits.
    from tracing import build_config, flush as flush_tracing

    config = build_config(ticker="TSLA")
    result = app.invoke(
        {"ticker": "TSLA", "retries": 0, "sources": []},
        config=config,
    )
    print("\n" + "=" * 60)
    print("FINAL BRIEFING")
    print("=" * 60)
    print(result["briefing"])
    flush_tracing()
