from retrieval import search_news, summarise_ticker

test_queries = [
    "NVDA earnings forecast",
    "Tesla demand slowdown",
    "Apple AI features",
    "Microsoft Azure cloud growth",
    "Amazon AWS competition",
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)
    print(search_news(query, top_k=3))