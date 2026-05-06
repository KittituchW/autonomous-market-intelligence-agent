import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from newsapi import NewsApiClient
from dotenv import load_dotenv

load_dotenv()

# init NewsAPI client
newsapi = NewsApiClient(api_key=os.getenv("NEWSAPI_KEY"))

# the 5 tickers we care about
TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN"]

BLOCKED_DOMAINS = [
    "rlsbb.to", "pypi.org", "cointelegraph.com",
    "marketbeat.com", "post.rlsbb.to"
]

# Day 12 task 2: NewsAPI free tier is 100 requests/day. We pre-check the
# counter and bail before the call, so a quota-bust does not waste a request
# on a 429. Counter resets at UTC midnight.
NEWSAPI_DAILY_LIMIT = 100
_NEWSAPI_COUNTER = Path("logs/newsapi_count.json")


def _load_counter() -> dict:
    if not _NEWSAPI_COUNTER.exists():
        return {"date": "", "count": 0}
    try:
        return json.loads(_NEWSAPI_COUNTER.read_text())
    except Exception:
        return {"date": "", "count": 0}


def _save_counter(state: dict) -> None:
    _NEWSAPI_COUNTER.parent.mkdir(parents=True, exist_ok=True)
    _NEWSAPI_COUNTER.write_text(json.dumps(state))


def _check_and_bump_quota() -> bool:
    """Bump today's NewsAPI counter by 1 if under the limit. Return True if ok."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    state = _load_counter()
    if state.get("date") != today:
        state = {"date": today, "count": 0}
    if state["count"] + 1 > NEWSAPI_DAILY_LIMIT:
        return False
    state["count"] += 1
    _save_counter(state)
    return True


def fetch_news_for_ticker(ticker: str, num_articles: int = 40) -> list[dict]:
    """Pull recent news articles for one ticker from NewsAPI."""
    if not _check_and_bump_quota():
        print(
            f"[ingest] NewsAPI daily quota reached ({NEWSAPI_DAILY_LIMIT}/day). "
            f"Skipping {ticker}. Counter resets at UTC midnight."
        )
        return []

    week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

    response = newsapi.get_everything(
        q=f'"{ticker}" AND (stock OR shares OR earnings OR market)',  # exact ticker + finance context, OR group properly grouped
        from_param=week_ago,
        language="en",
        sort_by="relevancy",
        page_size=num_articles,
    )

    articles = response.get("articles", [])
    cleaned = []
    for article in articles:
        if not article.get("content"):
            continue
        # skip known junk domains
        url = article.get("url", "")
        if any(domain in url for domain in BLOCKED_DOMAINS):
            continue
        cleaned.append({
            "ticker": ticker,
            "title": article["title"],
            "source": article["source"]["name"],
            "published_at": article["publishedAt"],
            "url": article["url"],
            "content": article["content"],
        })
    return cleaned

def save_articles(ticker: str, articles: list[dict]):
    """Save articles for one ticker as a JSON file."""
    # ensure the output dir exists, so a fresh clone does not crash
    os.makedirs("data/news", exist_ok=True)
    output_path = f"data/news/{ticker}.json"
    with open(output_path, "w") as f:
        json.dump(articles, f, indent=2)
    print(f"Saved {len(articles)} articles for {ticker} -> {output_path}")

if __name__ == "__main__":
    total = 0
    for ticker in TICKERS:
        articles = fetch_news_for_ticker(ticker, num_articles=40)
        save_articles(ticker, articles)
        total += len(articles)
    print(f"\nTotal articles saved: {total}")