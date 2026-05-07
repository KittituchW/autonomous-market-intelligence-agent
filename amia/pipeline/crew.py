"""4-agent CrewAI crew. Researcher + Analyst on Groq, Critic on Gemini, Writer on local Gemma."""
import os
import litellm
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from amia.retrieval.subquery import compare_news_and_social
from amia.observability.usage_log import litellm_callback as _usage_callback
from amia.config.models import (
    RESEARCHER_MODEL,
    ANALYST_MODEL,
    CRITIC_MODEL,
    WRITER_MODEL,
    WRITER_FALLBACK_MODEL,
    OLLAMA_BASE_URL,
    GEMINI_FALLBACK_MODEL,
)

load_dotenv()

# Every successful crew agent call lands in logs/token_usage.jsonl.
# Guard against duplicate registration when this module is imported twice
# (e.g. test runners), which would produce duplicate log entries.
if not isinstance(litellm.success_callback, list):
    litellm.success_callback = []
if _usage_callback not in litellm.success_callback:
    litellm.success_callback.append(_usage_callback)

# When Groq hits TPD or TPM, LiteLLM automatically falls back to Gemini.
litellm.num_retries = 2
litellm.fallbacks = [
    {RESEARCHER_MODEL: [GEMINI_FALLBACK_MODEL]},
    {ANALYST_MODEL: [GEMINI_FALLBACK_MODEL]},
    # On 8GB RAM gemma3:4b can OOM or time out mid-generation; runtime
    # fallback to Gemini keeps the briefing flowing.
    {WRITER_MODEL: [WRITER_FALLBACK_MODEL]},
]


def _force_text_tool_mode(llm: LLM) -> LLM:
    """Use CrewAI's text tool loop instead of provider-native tool calls.

    Groq can reject optional native tool calls with tool_use_failed when the
    model chooses to answer directly. CrewAI's ReAct loop allows the same agent
    tool while keeping the first response as plain text when no tool is needed.
    """
    object.__setattr__(llm, "supports_function_calling", lambda: False)
    return llm


researcher_llm = _force_text_tool_mode(
    LLM(
        model=RESEARCHER_MODEL,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )
)

analyst_llm = LLM(
    model=ANALYST_MODEL,
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

# Critic uses a different vendor for diversity so it does not anchor on the analyst.
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY", "")
critic_llm = LLM(model=CRITIC_MODEL, temperature=0)


# Writer runs on local Gemma by default. Three escape hatches in order of cost:
#   1. AMIA_WRITER=cloud env var -> skip Ollama entirely, use Gemini
#   2. Pre-flight check at module load -> swap to Gemini if Ollama is down
#      or gemma3:4b is not pulled
#   3. LiteLLM fallback (above) -> retry on Gemini if the writer crashes mid-run
def _pick_writer_llm():
    """Choose the writer LLM. Local Gemma if reachable, else Gemini."""
    forced = os.getenv("AMIA_WRITER", "").strip().lower()
    if forced == "cloud":
        print("[writer] AMIA_WRITER=cloud, using Gemini fallback")
        return LLM(
            model=WRITER_FALLBACK_MODEL,
            temperature=0.3,
            api_key=os.getenv("GEMINI_API_KEY"),
        )
    if forced == "local":
        # explicit local: skip the reachability probe
        return LLM(
            model=WRITER_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.3,
        )

    # Default: probe Ollama. 3-second timeout is enough -- if Ollama is up,
    # /api/tags responds in <100ms.
    try:
        import urllib.request
        import json as _json
        with urllib.request.urlopen(
            f"{OLLAMA_BASE_URL}/api/tags", timeout=3
        ) as resp:
            tags = _json.loads(resp.read())
        loaded = {m["name"] for m in tags.get("models", [])}
        wanted = WRITER_MODEL.split("/", 1)[-1]  # "gemma3:4b"
        if wanted not in loaded:
            print(
                f"[writer] {wanted} not pulled in Ollama "
                f"(found {sorted(loaded)}), falling back to Gemini"
            )
            return LLM(
                model=WRITER_FALLBACK_MODEL,
                temperature=0.3,
                api_key=os.getenv("GEMINI_API_KEY"),
            )
        return LLM(
            model=WRITER_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.3,
        )
    except Exception as e:
        print(f"[writer] Ollama unreachable ({e}), falling back to Gemini")
        return LLM(
            model=WRITER_FALLBACK_MODEL,
            temperature=0.3,
            api_key=os.getenv("GEMINI_API_KEY"),
        )


ollama_llm = _pick_writer_llm()


def _format_sources_block(sources: list[dict]) -> str:
    """Turn structured sources into a citation-ready block the agents can reference by id."""
    lines = []
    for s in sources:
        sentiment = f" [sentiment: {s['sentiment']}]" if s.get("sentiment") else ""
        lines.append(
            f"[{s['id']}] ({s['data_type']}/{s['source']}){sentiment} "
            f"{s['published_at']} - {s['url']}\n    {s['snippet']}..."
        )
    return "\n".join(lines)


def build_crew(ticker: str, context: str, sources: list[dict], verbose: bool = True) -> Crew:
    """Wire up 4 agents with their assigned models and pass them the same context + sources.

    Set verbose=True for single-ticker debugging, False for the multi-ticker loop
    so the terminal stays readable.
    """

    sources_block = _format_sources_block(sources)

    # 1. Researcher: scans context, picks the material signals.
    # compare_news_and_social tool handles cross-source questions.
    researcher = Agent(
        role="Financial Researcher",
        goal=f"Identify the 3 most material developments about {ticker} from the retrieved context",
        backstory=(
            "You are a senior buyside researcher. You skim noise fast and only "
            "flag developments that move price or change narrative. You always "
            "cite sources by their [id] number. Usually answer from the supplied "
            "context. Only call compare_news_and_social when the supplied context "
            "does not clearly show the news vs retail sentiment divergence."
        ),
        llm=researcher_llm,
        tools=[compare_news_and_social],
        allow_delegation=False,
        verbose=verbose,
    )

    # 2. Analyst: turns research into catalysts and risks
    analyst = Agent(
        role="Quantitative Analyst",
        goal="Convert research findings into 2 catalysts and 2 risks with reasoning",
        backstory=(
            "You think in catalysts (what could push the stock up) and risks "
            "(what could push it down). You weigh retail sentiment from "
            "StockTwits as a signal but never as the main driver. You cite [id]."
        ),
        llm=analyst_llm,
        allow_delegation=False,
        verbose=verbose,
    )

    # 3. Critic: challenges the analyst, runs on a different vendor for diversity
    critic = Agent(
        role="Skeptical Critic",
        goal="Find weak claims, confirmation bias, and missing counter-evidence in the analyst output",
        backstory=(
            "You assume every conclusion is wrong until proven. You are blunt "
            "and specific. If a catalyst lacks evidence in the sources, you say "
            "so by [id]. If the StockTwits sentiment is shallow or low-volume, "
            "you flag that. You never rewrite the analyst, you only critique."
        ),
        llm=critic_llm,
        allow_delegation=False,
        verbose=verbose,
    )

    # 4. Writer: composes the final briefing locally on Gemma 3 4B
    writer = Agent(
        role="Briefing Writer",
        goal="Produce the final 1-paragraph + 3-bullet briefing with citations",
        backstory=(
            "You write in plain English, no buzzwords. You incorporate the "
            "critic's feedback. You always end with a 'Sources:' block listing "
            "the cited [id] entries with their URLs."
        ),
        llm=ollama_llm,
        allow_delegation=False,
        verbose=verbose,
    )

    # tasks chain in order: research -> analyse -> critique -> write
    research_task = Task(
        description=(
            f"Research the most important developments about {ticker} today.\n\n"
            f"CONTEXT (read this carefully):\n{context}\n\n"
            f"NUMBERED SOURCES (cite by id):\n{sources_block}\n\n"
            "Pick the 3 most material signals. For each, write 2-3 sentences "
            "and cite the source [id]. Use only ids from NUMBERED SOURCES. "
            "Do not fabricate facts, ids, or URLs."
        ),
        expected_output="3 numbered findings, each with source [id] citations",
        agent=researcher,
    )

    analyst_task = Task(
        description=(
            "Using the researcher's findings, identify 2 catalysts (upside) "
            "and 2 risks (downside) for the stock. For each, give one sentence "
            "of reasoning and cite the supporting source [id]. Use StockTwits "
            "sentiment as a secondary signal, not primary."
        ),
        expected_output="2 catalysts and 2 risks, each one sentence with [id] citations",
        agent=analyst,
        context=[research_task],
    )

    critic_task = Task(
        description=(
            "Critique the analyst's catalysts and risks. For each one, say "
            "whether the evidence in the sources actually supports it. Flag "
            "any missing counter-evidence. Be blunt. Cite [id]."
        ),
        expected_output="A bullet list of weak points and missing evidence, with [id] citations",
        agent=critic,
        context=[research_task, analyst_task],
    )

    write_task = Task(
        description=(
            f"Write the final daily briefing on {ticker}. Use this exact format:\n\n"
            "**Summary:** one paragraph, 3-4 sentences, plain English.\n\n"
            "**Key Signals:**\n"
            "- News: ...\n"
            "- Retail sentiment: ...\n"
            "- Risks: ...\n\n"
            "**Watch:** one line on what to track tomorrow.\n\n"
            "**Sources:** numbered list of cited [id] entries with URLs.\n\n"
            "Incorporate the critic's pushback. Only cite sources actually "
            "discussed by the researcher or analyst. The News bullet must cite "
            "at least one news source id, the Retail sentiment bullet must cite "
            "at least one social source id, and the Sources block must reuse "
            "the exact retrieved URLs. No fluff, no buzzwords."
        ),
        expected_output="A clean markdown briefing with the 4 sections above",
        agent=writer,
        context=[research_task, analyst_task, critic_task],
    )

    return Crew(
        agents=[researcher, analyst, critic, writer],
        tasks=[research_task, analyst_task, critic_task, write_task],
        process=Process.sequential,  # one task at a time, output flows forward
        verbose=verbose,
        tracing=True,  # skip the interactive y/N trace prompt at end of run
        manager_llm=analyst_llm,  # silences the "LLM is explicitly disabled. Using MockLLM" warning
    )


def main() -> None:
    # quick smoke test on TSLA, pulls real context from the index
    from amia.retrieval.index import retrieve_with_sources

    context, sources = retrieve_with_sources(
        "TSLA news earnings sentiment risks", top_k=8
    )
    print(f"Pulled {len(sources)} sources for the crew\n")

    crew = build_crew("TSLA", context, sources)
    result = crew.kickoff()

    print("\n" + "=" * 60)
    print("FINAL BRIEFING")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()
