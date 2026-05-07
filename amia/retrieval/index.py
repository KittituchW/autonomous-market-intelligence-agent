import os
import json
import time
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

# Disk cache for retrieve_with_sources. News doesn't move that fast intraday,
# and dev iteration runs the same (ticker, plan) query repeatedly. 6h TTL
# means the daily ingest still gets fresh retrieval.
_CACHE_DIR = Path("cache/retrieve")
_CACHE_TTL_SECONDS = 60 * 60 * 6


def _cache_key(query: str, ticker: str | None, top_k: int, social_quota: int) -> Path:
    raw = f"{query}|{ticker}|{top_k}|{social_quota}".encode()
    digest = hashlib.sha256(raw).hexdigest()[:16]
    return _CACHE_DIR / f"{digest}.json"


def _cache_get(path: Path):
    try:
        age = time.time() - path.stat().st_mtime
    except FileNotFoundError:
        return None
    if age > _CACHE_TTL_SECONDS:
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data["formatted"], data["sources"]


def _cache_put(path: Path, formatted: str, sources: list[dict]) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"formatted": formatted, "sources": sources}, f)
    except OSError:
        # cache write failures must not break retrieval
        pass

# use nomic-embed-text via Ollama
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = None

# connect to local Qdrant
qdrant_client = QdrantClient(host="localhost", port=6333, check_compatibility=False)

# explicitly build the vector store and storage context
vector_store = QdrantVectorStore(client=qdrant_client, collection_name="amia_news")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

def _load_dir(dir_path: str, data_type: str) -> list[Document]:
    """Generic loader: read every JSON file in a folder, turn each item into a Document."""
    docs = []
    if not os.path.isdir(dir_path):
        return docs

    for filename in os.listdir(dir_path):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(dir_path, filename)
        with open(filepath) as f:
            items = json.load(f)

        for item in items:
            # news has 'title', social does not always, so build text defensively
            title = item.get("title", "")
            content = item.get("content", "")
            text = f"{title}\n\n{content}".strip() if title else content
            if not text:
                continue

            metadata = {
                "ticker": item.get("ticker", "?"),
                "source": item.get("source", data_type),
                "published_at": item.get("published_at", ""),
                "url": item.get("url", ""),
                "data_type": data_type,
            }
            # carry sentiment through if it exists, useful later for the analyst
            if "sentiment" in item:
                metadata["sentiment"] = item["sentiment"]

            docs.append(Document(text=text, metadata=metadata))
    return docs


def load_documents() -> list[Document]:
    """Load news + social docs. Each file is JSON list of dicts."""
    news_docs = _load_dir("data/news", data_type="news")
    social_docs = _load_dir("data/social/filtered", data_type="social")
    print(f"Loaded {len(news_docs)} news docs, {len(social_docs)} social docs")
    return news_docs + social_docs


def build_index() -> VectorStoreIndex:
    """Build the vector index and store it in Qdrant."""
    # wipe old collection
    if qdrant_client.collection_exists("amia_news"):
        qdrant_client.delete_collection("amia_news")
        print("Deleted old collection.")

    # recreate it with the right vector size for nomic-embed-text (768 dims)
    qdrant_client.create_collection(
        collection_name="amia_news",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    print("Created fresh collection.")

    documents = load_documents()
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )
    print("Index built and stored in Qdrant.")
    return index

def load_index() -> VectorStoreIndex:
    """Load existing index from Qdrant."""
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )

# cache the index at module level so we do not rebuild it on every search
_index = None

def get_index() -> VectorStoreIndex:
    """Return a cached index, building it on first call only."""
    global _index
    if _index is None:
        _index = load_index()
    return _index


def _format_nodes(nodes) -> str:
    """Turn retrieved nodes into a readable string. Uses .get so a missing key does not crash."""
    results = []
    for i, node in enumerate(nodes):
        meta = node.metadata or {}
        results.append(
            f"[{i+1}] {meta.get('ticker', '?')} | {meta.get('source', '?')} | {meta.get('published_at', '?')}\n"
            f"{node.text[:300]}...\n"
            f"URL: {meta.get('url', '?')}"
        )
    return "\n\n".join(results)


def search_news(query: str, top_k: int = 5) -> str:
    """Search the index and return top_k results as formatted text."""
    index = get_index()
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    return _format_nodes(nodes)


def retrieve_with_sources(
    query: str,
    ticker: str | None = None,
    top_k: int = 8,
    social_quota: int = 2,
) -> tuple[str, list[dict]]:
    """Like search_news but also returns a structured list of sources for citations.

    Hard filters by ticker, blends in StockTwits + HackerNews quotas so
    cross-source briefings have real retail signal, dedupes by URL to kill
    same-article-three-times duplicates.

    Args:
        query:        embedding query string
        ticker:       if set, results are hard-filtered to this ticker on metadata
        top_k:        primary retrieval count (mixed sources, ranked by similarity)
        social_quota: how many StockTwits and HackerNews docs to force in on top
                      of top_k. 2 + 2 = 4, so final list is up to top_k + 4 docs
                      after dedupe.

    Returns:
        formatted: human-readable text passed to agents as context
        sources:   list of dicts with ticker, source, url, sentiment, data_type
                   so the writer agent can cite them by index
    """
    cache_path = _cache_key(query, ticker, top_k, social_quota)
    cached = _cache_get(cache_path)
    if cached is not None:
        return cached

    index = get_index()

    # 1. main retrieve, ticker-filtered if a ticker was passed in
    base_filter = None
    if ticker:
        base_filter = MetadataFilters(
            filters=[MetadataFilter(key="ticker", value=ticker)]
        )
    main_retriever = index.as_retriever(similarity_top_k=top_k, filters=base_filter)
    main_nodes = main_retriever.retrieve(query)

    # 2. force StockTwits + HackerNews to surface, otherwise vector similarity
    # picks news every time and cross-source questions return 0 social hits.
    social_nodes: list = []
    if ticker and social_quota > 0:
        for source in ("stocktwits", "hackernews"):
            filters = MetadataFilters(filters=[
                MetadataFilter(key="ticker", value=ticker),
                MetadataFilter(key="source", value=source),
            ])
            retriever = index.as_retriever(similarity_top_k=social_quota, filters=filters)
            social_nodes.extend(retriever.retrieve(query))

    # 3. combine and dedupe by URL; the same article can be tagged under
    # multiple tickers and would otherwise appear several times.
    all_nodes = main_nodes + social_nodes
    seen: set[str] = set()
    deduped = []
    for n in all_nodes:
        url = (n.metadata or {}).get("url", "")
        # fall back to text prefix when URL is missing so we still dedupe
        key = url if url else n.text[:120]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(n)

    sources = []
    for i, node in enumerate(deduped):
        meta = node.metadata or {}
        sources.append({
            "id": i + 1,
            "ticker": meta.get("ticker", "?"),
            "source": meta.get("source", "?"),
            "data_type": meta.get("data_type", "?"),
            "sentiment": meta.get("sentiment"),  # only present on StockTwits
            "published_at": meta.get("published_at", ""),
            "url": meta.get("url", ""),
            "snippet": node.text[:200],
        })

    formatted = _format_nodes(deduped)
    _cache_put(cache_path, formatted, sources)
    return formatted, sources


def summarise_ticker(ticker: str, top_k: int = 5) -> str:
    """Return the top results for a specific ticker, hard-filtered on metadata."""
    index = get_index()
    # metadata filter so we only get rows where ticker matches exactly
    filters = MetadataFilters(filters=[MetadataFilter(key="ticker", value=ticker)])
    retriever = index.as_retriever(similarity_top_k=top_k, filters=filters)
    # still pass a query string so vector search ranks within the filtered set
    nodes = retriever.retrieve(f"{ticker} latest news earnings outlook")
    return _format_nodes(nodes)


# Filter helpers: let agents ask for one source type only, used by the
# sub-question query engine and for comparing news vs social signals.

def _filtered_search(query: str, filters: MetadataFilters, top_k: int) -> str:
    """Generic filtered search, used by the helpers below."""
    index = get_index()
    retriever = index.as_retriever(similarity_top_k=top_k, filters=filters)
    nodes = retriever.retrieve(query)
    return _format_nodes(nodes)


def search_news_only(query: str, top_k: int = 5) -> str:
    """Search only across news articles. Skips StockTwits and HackerNews."""
    filters = MetadataFilters(filters=[MetadataFilter(key="data_type", value="news")])
    return _filtered_search(query, filters, top_k)


def search_social_only(query: str, top_k: int = 5) -> str:
    """Search only across social sources (StockTwits + HackerNews)."""
    filters = MetadataFilters(filters=[MetadataFilter(key="data_type", value="social")])
    return _filtered_search(query, filters, top_k)


def search_hn_only(query: str, top_k: int = 5) -> str:
    """Search only HackerNews stories. Useful for tech sentiment vs retail trader sentiment."""
    filters = MetadataFilters(filters=[MetadataFilter(key="source", value="hackernews")])
    return _filtered_search(query, filters, top_k)


def search_stocktwits_only(query: str, top_k: int = 5) -> str:
    """Search only StockTwits messages. Useful for retail trader sentiment."""
    filters = MetadataFilters(filters=[MetadataFilter(key="source", value="stocktwits")])
    return _filtered_search(query, filters, top_k)

def main() -> None:
    build_index()


if __name__ == "__main__":
    main()
