"""SubQuestionQueryEngine over the Qdrant index with two filtered sub-engines.

The engine decomposes a hard question into sub-questions, routes each one to
either the news engine or the social engine (whichever is more relevant),
then synthesises one answer.
"""
import os
from dotenv import load_dotenv
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.llms.groq import Groq
from amia.retrieval.index import get_index
from amia.config.models import RETRIEVE_MODEL_BARE

load_dotenv()

# Decomposition + synthesis LLM. Passed explicitly so we don't touch
# Settings.llm and conflict with the index module. Kept on a small model
# because this engine is called as a tool from the Researcher (Llama 4 Scout),
# which does the real synthesis itself.
groq_llm = Groq(
    model=RETRIEVE_MODEL_BARE,
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)


def _news_engine():
    """Query engine restricted to news articles via metadata filter."""
    filters = MetadataFilters(filters=[MetadataFilter(key="data_type", value="news")])
    return get_index().as_query_engine(
        similarity_top_k=5,
        filters=filters,
        llm=groq_llm,
    )


def _social_engine():
    """Query engine restricted to StockTwits + HackerNews."""
    filters = MetadataFilters(filters=[MetadataFilter(key="data_type", value="social")])
    return get_index().as_query_engine(
        similarity_top_k=5,
        filters=filters,
        llm=groq_llm,
    )


def build_subquestion_engine() -> SubQuestionQueryEngine:
    """Build the engine that picks news_search vs social_search per sub-question."""
    news_tool = QueryEngineTool(
        query_engine=_news_engine(),
        metadata=ToolMetadata(
            name="news_search",
            description=(
                "Searches financial news articles about AAPL, TSLA, NVDA, MSFT, AMZN. "
                "Use for analyst views, earnings, market events, and official company news."
            ),
        ),
    )
    social_tool = QueryEngineTool(
        query_engine=_social_engine(),
        metadata=ToolMetadata(
            name="social_search",
            description=(
                "Searches retail trader and tech-community posts (StockTwits + HackerNews). "
                "Use for retail sentiment, trader chatter, and tech/product community signal."
            ),
        ),
    )

    return SubQuestionQueryEngine.from_defaults(
        query_engine_tools=[news_tool, social_tool],
        llm=groq_llm,
        use_async=False,  # 8GB RAM, keep it simple
        verbose=True,  # prints the decomposition into sub-questions, useful for debugging
    )


# Cache the engine so we do not rebuild on every call (cheap but pointless).
_engine = None


def ask(question: str) -> str:
    """Ask a multi-source question, return the synthesised answer string."""
    global _engine
    if _engine is None:
        _engine = build_subquestion_engine()
    response = _engine.query(question)
    return str(response)


# --- CrewAI tool wrapper ---
# So the researcher agent can call the sub-question engine in the middle of its task.
from crewai.tools import tool  # noqa: E402


@tool("compare_news_and_social")
def compare_news_and_social(question: str) -> str:
    """Compare news framing against retail/tech-community sentiment.

    Use this when a question requires comparing what analysts/news say versus what
    retail traders on StockTwits or HackerNews users are saying. The tool will
    decompose the question, route sub-questions to news vs social, and synthesise
    one answer. Pass a single natural-language question as the only argument.
    """
    return ask(question)


def main() -> None:
    # 3 test questions per Day 5 plan
    test_questions = [
        "Compare StockTwits sentiment vs news sentiment on TSLA over the last week.",
        "What are analysts saying about NVDA earnings, and how does retail trader sentiment differ?",
        "Has the AAPL narrative shifted between news and HackerNews?",
    ]
    for q in test_questions:
        print("\n" + "=" * 70)
        print(f"Q: {q}")
        print("=" * 70)
        print(ask(q))


if __name__ == "__main__":
    main()
