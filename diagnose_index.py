"""Day 11 diagnostic: count docs per source in Qdrant vs on disk.

If Qdrant has 0 stocktwits docs but disk has 70+, the index is stale.
Fix: rerun `python retrieval.py` to rebuild.
"""
import json
import os
from collections import Counter
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333, check_compatibility=False)

# 1. count what's actually in Qdrant
print("=" * 60)
print("QDRANT PAYLOAD COUNTS")
print("=" * 60)
points, _ = client.scroll(
    collection_name="amia_news",
    limit=10000,
    with_payload=True,
    with_vectors=False,
)
print(f"Total points in Qdrant: {len(points)}")

source_counter = Counter()
ticker_counter = Counter()
data_type_counter = Counter()
ticker_x_source = Counter()

for p in points:
    payload = p.payload or {}
    # llamaindex stores metadata under different paths depending on version,
    # try a few options
    meta = payload.get("metadata") or payload
    src = meta.get("source", "?")
    tic = meta.get("ticker", "?")
    dt = meta.get("data_type", "?")
    source_counter[src] += 1
    ticker_counter[tic] += 1
    data_type_counter[dt] += 1
    ticker_x_source[(tic, src)] += 1

print(f"\nBy source: {dict(source_counter)}")
print(f"By ticker: {dict(ticker_counter)}")
print(f"By data_type: {dict(data_type_counter)}")
print(f"\nTicker x source matrix:")
for (t, s), c in sorted(ticker_x_source.items()):
    print(f"  {t:6} {s:15} {c}")

# 2. count what's on disk
print("\n" + "=" * 60)
print("DISK COUNTS")
print("=" * 60)
disk_counter = Counter()
for fn in os.listdir("data/news"):
    if fn.endswith(".json"):
        d = json.load(open(f"data/news/{fn}"))
        for item in d:
            disk_counter[("news", item.get("source", "?"))] += 1
for fn in os.listdir("data/social/filtered"):
    if fn.endswith(".json"):
        d = json.load(open(f"data/social/filtered/{fn}"))
        for item in d:
            disk_counter[("social", item.get("source", "?"))] += 1

for k, v in sorted(disk_counter.items()):
    print(f"  {k[0]:8} {k[1]:15} {v}")

# 3. verdict
print("\n" + "=" * 60)
print("VERDICT")
print("=" * 60)
qdrant_st = source_counter.get("stocktwits", 0)
disk_st = sum(v for k, v in disk_counter.items() if k[1] == "stocktwits")
qdrant_hn = source_counter.get("hackernews", 0)
disk_hn = sum(v for k, v in disk_counter.items() if k[1] == "hackernews")

print(f"StockTwits: disk={disk_st}, qdrant={qdrant_st}")
print(f"HackerNews: disk={disk_hn}, qdrant={qdrant_hn}")

if qdrant_st == 0 and disk_st > 0:
    print("\n>>> Index is stale. Run `python retrieval.py` to rebuild. <<<")
elif qdrant_st < disk_st * 0.5:
    print("\n>>> Index is partial. Run `python retrieval.py` to rebuild. <<<")
else:
    print("\nIndex looks complete. Bug is elsewhere.")
