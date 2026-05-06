"""Day 3-5: 4-agent CrewAI crew. Researcher + Analyst on Groq, Critic on Groq 8B, Writer on local Gemma.

Day 5 update: researcher now has access to compare_news_and_social, a sub-question
engine that routes between news-only and social-only retrieval.

Day 12 update: Groq -> Gemini fallback wired via LiteLLM globals so a TPD/TPM
hit on Groq doesn't kill an eval run mid-way. The crew calls litellm under
the hood, so setting litellm.fallbacks once here propagates to every agent.
"""
import os
import litellm
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from subquery import compare_news_and_social
from usage_log import litellm_callback as _usage_callback
from models import (
    RESEARCHER_MODEL,
    ANALYST_MODEL,
    CRITIC_MODEL,
    WRITER_MODEL,
    WRITER_FALLBACK_MODEL,
    OLLAMA_BASE_URL,
    GEMINI_FALLBACK_MODEL,
)

load_dotenv()

# Day 12 task 4: every successful crew agent call lands in logs/token_usage.jsonl
# litellm.success_callback is a list of either string names or callables.
# Append our callable so it runs alongside any existing callbacks.
if not isinstance(litellm.success_callback, list):
    litellm.success_callback = []
litellm.success_callback.append(_usage_callback)

# CrewAI 0.51 wants its own LLM wrapper (LiteLLM under the hood),
# not langchain_groq / langchain_google_genai / langchain_ollama objects directly.
# Model strings follow LiteLLM convention: <provider>/<model_id>.

# --- Day 12: rate-limit reliability ---
# When Groq hits TPD (100k/day on free tier) or TPM (6k/min), automatically
# fall back to Gemini. LiteLLM checks these globals on every completion call.
# Day 13: fallback list now covers the NEW per-agent model ids.
litellm.num_retries = 2  # transient errors get a free retry first
litellm.fallbacks = [
    {RESEARCHER_MODEL: [GEMINI_FALLBACK_MODEL]},
    {ANALYST_MODEL: [GEMINI_FALLBACK_MODEL]},
    # Writer runtime fallback. If gemma3:4b times out or OOMs mid-generation
    # (8GB RAM is tight when Chrome + Cursor + Qdrant are also resident),
    # LiteLLM transparently retries on Gemini 3.1 Flash Lite.
    {WRITER_MODEL: [WRITER_FALLBACK_MODEL]},
]

# --- Day 13: per-agent model split ---
# Previously researcher + analyst shared one Llama 3.3 70B instance. We split
# them so each agent gets the model that fits its job:
#   - Researcher reads many docs and extracts cited facts -> Llama 4 Scout
#     (long context + high throughput on Groq)
#   - Analyst does numerical reasoning over the findings -> Qwen3-32B
#     (consistently strongest open model on math at this size)
# Needs GROQ_API_KEY env var set, which load_dotenv handles above.
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

# Critic uses a different vendor (Gemini) for diversity so it does not anchor
# on the analyst. Top wants this kept on Gemini 3.1 Flash Lite Preview.
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY", "")
critic_llm = LLM(model=CRITIC_MODEL, temperature=0)

# --- Local writer with cloud fallback ---
# Day 13: writer runs on local Gemma 3 4B by default. On 8GB RAM the 4B model
# is borderline -- if Chrome/Cursor/Qdrant are also resident it can OOM or
# time out. Three escape hatches, in order of cost:
#   1. AMIA_WRITER=cloud env var -> skip Ollama entirely, use Gemini
#   2. Pre-flight check at module load -> if Ollama is down or gemma3:4b
#      is not pulled, swap to Gemini before the crew is built
#   3. LiteLLM fallback (above) -> if the writer crashes mid-run, the next
#      call retries on Gemini transparently
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
    # Day 5: gets compare_news_and_social as a tool for hard cross-source questions.
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
        tools=[compare_news_and_social],  # Day 5 sub-question engine wrapped as tool
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
            "and cite the source [id]. Do not fabricate."
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
            "discussed by the researcher or analyst. No fluff, no buzzwords."
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


if __name__ == "__main__":
    # quick smoke test on TSLA, pulls real context from the index
    from retrieval import retrieve_with_sources

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
