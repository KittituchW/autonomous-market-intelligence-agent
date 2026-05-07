"""LangGraph: plan -> retrieve -> write (CrewAI 4-agent crew), with one retry on empty retrieval."""
import os
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from amia.config import TICKERS
from amia.config.models import PLANNER_MODEL_BARE, GEMINI_MODEL_BARE
from amia.observability.usage_log import log as _usage_log
from amia.pipeline.crew import build_crew
from amia.quality.briefing_quality import (
    build_fallback_briefing,
    replace_sources_block,
    validate_briefing,
)
from amia.retrieval.index import retrieve_with_sources

load_dotenv()

# Planning is the choke point: weak decomposition cascades into bad retrieval
# queries, so we pay for the bigger reasoning model here.
llm = ChatGroq(
    model=PLANNER_MODEL_BARE,
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

# Planner fallback. langchain_groq raises groq.RateLimitError on 429, but
# litellm-style errors leak through too, so we match on the message.
fallback_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL_BARE,
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

_RATE_LIMIT_TOKENS = ("rate_limit", "ratelimit", "429", "tpd", "tpm")


def _invoke_with_fallback(prompt: str) -> str:
    """Call the planner LLM, fall back to Gemini on any rate limit / 429.

    The crew has its own fallback wired via litellm globals in crew.py.
    This function only covers the plan_node call which goes through
    langchain_groq, not litellm.
    """
    try:
        resp = llm.invoke(prompt)
        usage = resp.response_metadata.get("token_usage", {}) or {}
        _usage_log(
            provider="groq",
            model=PLANNER_MODEL_BARE,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            source="planner",
        )
        return resp.content
    except Exception as e:
        msg = str(e).lower()
        if not any(token in msg for token in _RATE_LIMIT_TOKENS):
            raise
        print("[plan] Groq rate limited, falling back to Gemini")
        resp = fallback_llm.invoke(prompt)
        usage = (resp.response_metadata or {}).get("usage_metadata", {}) or {}
        _usage_log(
            provider="gemini",
            model=GEMINI_MODEL_BARE,
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
    quality_warnings: list[str]
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

    Passing ticker hard-filters on metadata and adds StockTwits + HackerNews quotas.
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


def write_node(state: State) -> dict:
    """Hand off to the CrewAI crew, then post-process URLs.

    The writer cannot be trusted to copy URLs verbatim, so we replace its
    Sources block with the real URLs from retrieval before saving.
    """
    ticker = state["ticker"]
    sources = state.get("sources", [])
    try:
        crew = build_crew(
            ticker=ticker,
            context=state["context"],
            sources=sources,
            verbose=os.getenv("AMIA_CREW_VERBOSE", "").strip() == "1",
        )
        # kickoff runs research -> analyst -> critic -> writer in sequence
        result = crew.kickoff()
        # CrewAI 0.51 returns a CrewOutput object, get the raw string
        raw = str(result)
    except Exception as e:
        warning = f"crew failed: {e.__class__.__name__}: {e}"
        print(f"[write] {warning}; using deterministic fallback")
        fallback = build_fallback_briefing(ticker, sources, reason=warning)
        return {"briefing": fallback, "quality_warnings": [warning]}

    cleaned = replace_sources_block(raw, sources)
    quality = validate_briefing(cleaned, sources)
    if not quality.ok:
        print(f"[write] briefing failed quality gate: {quality.warnings}")
        fallback = build_fallback_briefing(
            ticker,
            sources,
            reason="; ".join(quality.warnings),
        )
        return {"briefing": fallback, "quality_warnings": quality.warnings}

    return {"briefing": cleaned, "quality_warnings": []}


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


def run_for_question(question: str, default_ticker: str = "AAPL") -> dict:
    """Eval-friendly entry point.

    The graph itself is ticker-driven (the planner generates its own questions
    inside the graph). For evals we feed in a free-form question, still run
    the full pipeline, and get back both the briefing and the structured
    sources retrieve_with_sources produced.

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
    from amia.observability.tracing import build_config

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


def main() -> None:
    # smoke test on one ticker, full crew runs inside write_node.
    from amia.observability.tracing import build_config, flush as flush_tracing

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


if __name__ == "__main__":
    main()
