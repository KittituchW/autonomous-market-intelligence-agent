import html
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import requests

from amia.config import TICKERS

RAW_DIR = "data/social/raw"
FILTERED_DIR = "data/social/filtered"

STOCKTWITS_DAYS = 3
HACKERNEWS_DAYS = 30

BLOCKED_TERMS = [
    "epstein",
    "trump",
    "clinton",
    "hillary",
    "iran",
    "pedophile",
    "democrat",
    "republican",
    "politics",
    "election",
    "shib",
    "floki",
    "lunc",
    "crypto pump",
]

COMPANY_TERMS = {
    "AAPL": ["apple", "iphone", "mac", "macbook", "ios", "app store", "wwdc", "vision pro", "airtag"],
    "TSLA": ["tesla", "elon", "autopilot", "fsd", "robotaxi", "ev", "cybertruck", "model y", "model 3"],
    "NVDA": ["nvidia", "gpu", "cuda", "blackwell", "ai chip", "data center", "h100", "h200", "geforce"],
    "MSFT": ["microsoft", "azure", "copilot", "openai", "windows", "office", "github", "xbox"],
    "AMZN": ["amazon", "aws", "anthropic", "prime", "alexa", "kindle", "warehouse", "ecommerce"],
}

HN_QUERIES = {
    "AAPL": ["apple ai", "iphone", "ios", "app store", "wwdc", "vision pro"],
    "TSLA": ["tesla", "elon musk", "autopilot", "fsd", "robotaxi", "ev"],
    "NVDA": ["nvidia", "gpu", "cuda", "ai chips", "blackwell"],
    "MSFT": ["microsoft", "azure", "copilot", "openai", "windows"],
    "AMZN": ["amazon", "aws", "anthropic", "cloud", "warehouse automation"],
}

BROAD_HN_QUERY_TITLE_TERMS = {
    "AAPL": {
        "queries": ["ios", "app store"],
        "title_terms": ["apple", "iphone", "mac", "app store", "wwdc"],
    }
}

INVESTOR_TERMS = [
    "earnings",
    "revenue",
    "sales",
    "guidance",
    "margin",
    "demand",
    "supply",
    "lawsuit",
    "regulation",
    "antitrust",
    "price",
    "stock",
    "shares",
    "acquisition",
    "ai strategy",
]

CASHTAG_RE = re.compile(r"\$([A-Za-z][A-Za-z0-9]*)")
HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"\b[\w']+\b")


def stocktwits_headers(ticker: str) -> dict:
    """Browser-like headers help the public StockTwits endpoint return JSON."""
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": f"https://stocktwits.com/symbol/{ticker}",
    }


def parse_datetime(value: str):
    """Parse API timestamps into timezone-aware UTC datetimes."""
    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def clean_text(value) -> str:
    """Decode HTML entities, remove simple tags, and normalize whitespace."""
    if value is None:
        return ""

    text = html.unescape(str(value))
    text = HTML_TAG_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def extract_cashtags(text: str) -> list[str]:
    """Return unique uppercase cashtags without the $ symbol."""
    seen = set()
    cashtags = []

    for match in CASHTAG_RE.findall(text):
        tag = match.upper()
        if tag not in seen:
            seen.add(tag)
            cashtags.append(tag)

    return cashtags


def text_has_term(text: str, term: str) -> bool:
    """Match single words by word boundary and phrases by simple containment."""
    term = term.lower()
    if " " in term:
        return term in text
    return re.search(rf"\b{re.escape(term)}\b", text) is not None


def has_blocked_term(text: str) -> bool:
    lowered = text.lower()
    return any(text_has_term(lowered, term) for term in BLOCKED_TERMS)


def is_recent(post: dict, max_age_days: int) -> bool:
    published_at = parse_datetime(post.get("published_at", ""))
    if published_at is None:
        return False

    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    return published_at >= cutoff


def is_company_relevant(post: dict) -> bool:
    """Return True when title/content mention this ticker's company story."""
    ticker = post.get("ticker", "").upper()
    text = f"{post.get('title', '')} {post.get('content', '')}".lower()

    for term in COMPANY_TERMS.get(ticker, []):
        if text_has_term(text, term):
            return True

    # StockTwits gets one extra relevance signal: exact target cashtag.
    if post.get("source") == "stocktwits" and ticker in post.get("cashtags", []):
        return True

    return False


def title_has_company_term(post: dict) -> bool:
    title = post.get("title", "").lower()
    ticker = post.get("ticker", "").upper()
    return any(text_has_term(title, term) for term in COMPANY_TERMS.get(ticker, []))


def is_investor_relevant_hn(post: dict) -> bool:
    """Return True when an HN story has investor/market framing."""
    text = f"{post.get('title', '')} {post.get('content', '')}".lower()
    return any(text_has_term(text, term) for term in INVESTOR_TERMS)


def broad_hn_query_has_weak_title(post: dict) -> bool:
    ticker = post.get("ticker", "").upper()
    config = BROAD_HN_QUERY_TITLE_TERMS.get(ticker)
    if not config:
        return False

    query = post.get("query_used", "").lower()
    if query not in config["queries"]:
        return False

    title = post.get("title", "").lower()
    return not any(text_has_term(title, term) for term in config["title_terms"])


def stocktwits_quality_score(post: dict) -> int:
    """Score useful company-specific posts higher without overvaluing followers."""
    content = post.get("content", "")
    sentiment = post.get("sentiment", "neutral")
    followers = post.get("user_followers", 0) or 0
    cashtag_count = len(post.get("cashtags", []))

    score = 20
    score += min(word_count(content) * 2, 30)

    if is_company_relevant(post):
        score += 20
    if sentiment in ("bullish", "bearish"):
        score += 8

    # Keep follower influence small so spammy posts cannot score highly.
    if followers >= 1000 and word_count(content) >= 8 and cashtag_count <= 2:
        score += 5
    elif followers >= 100 and word_count(content) >= 8 and cashtag_count <= 2:
        score += 3

    if cashtag_count >= 3:
        score -= 15
    if has_blocked_term(content):
        score -= 30

    return max(0, min(score, 100))


def hackernews_quality_score(post: dict) -> int:
    """Score company-relevant, discussed stories higher."""
    content = post.get("content", "")
    points = post.get("points", 0) or 0
    comments = post.get("num_comments", 0) or 0

    score = 15
    score += min(word_count(content), 25)
    score += min(points, 25)
    score += min(comments * 2, 20)

    if is_company_relevant(post):
        score += 20
    if is_investor_relevant_hn(post):
        score += 15
    if title_has_company_term(post):
        score += 15
    if broad_hn_query_has_weak_title(post):
        score -= 20

    return max(0, min(score, 100))


def get_stocktwits_rejection_reasons(post: dict) -> list[str]:
    reasons = []
    ticker = post.get("ticker", "").upper()
    content = post.get("content", "")
    cashtags = post.get("cashtags", [])

    if not is_recent(post, STOCKTWITS_DAYS):
        reasons.append("too_old")
    if ticker not in cashtags:
        reasons.append("missing_target_cashtag")
    if word_count(content) < 5:
        reasons.append("too_short")
    if len(cashtags) >= 4:
        reasons.append("too_many_cashtags")
    if has_blocked_term(content):
        reasons.append("blocked_term")
    if not is_company_relevant(post):
        reasons.append("not_company_relevant")

    return reasons


def get_hackernews_rejection_reasons(post: dict) -> list[str]:
    reasons = []
    content = post.get("content", "")
    points = post.get("points", 0) or 0
    comments = post.get("num_comments", 0) or 0

    if not is_recent(post, HACKERNEWS_DAYS):
        reasons.append("too_old")
    if word_count(content) < 5:
        reasons.append("too_short")
    if points < 3 and comments < 1:
        reasons.append("low_hn_engagement")
    if not is_company_relevant(post):
        reasons.append("not_company_relevant")
    if not is_investor_relevant_hn(post):
        reasons.append("not_investor_relevant")

    return reasons


def keep_stocktwits_post(post: dict) -> bool:
    return len(get_stocktwits_rejection_reasons(post)) == 0


def keep_hackernews_story(post: dict) -> bool:
    return len(get_hackernews_rejection_reasons(post)) == 0


def save_json(path: str, items: list[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(items, f, indent=2)
    print(f"Saved {len(items)} -> {path}")


def fetch_stocktwits_raw(ticker: str) -> list[dict]:
    """Pull recent StockTwits messages for a ticker. Public API, no auth."""
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"

    # One retry helps with short-lived rate limits or Cloudflare warmup.
    for attempt in range(2):
        try:
            resp = requests.get(url, headers=stocktwits_headers(ticker), timeout=10)
        except requests.RequestException as e:
            print(f"StockTwits {ticker} attempt {attempt + 1} failed: {e}")
            time.sleep(2)
            continue

        if resp.status_code == 200:
            try:
                return resp.json().get("messages", [])
            except ValueError:
                print(f"StockTwits {ticker} returned non-JSON content")
                return []

        challenge = resp.headers.get("cf-mitigated") == "challenge"
        reason = "Cloudflare challenge" if challenge else resp.text[:120]
        print(f"StockTwits {ticker} attempt {attempt + 1}: HTTP {resp.status_code} :: {reason}")
        time.sleep(2)

    return []


def build_stocktwits_post(ticker: str, msg: dict) -> dict:
    content = clean_text(msg.get("body", ""))
    sentiment_block = msg.get("entities", {}).get("sentiment") or {}
    sentiment = clean_text(sentiment_block.get("basic", "neutral")).lower()
    if sentiment not in ("bullish", "bearish"):
        sentiment = "neutral"

    post = {
        "ticker": ticker,
        "source": "stocktwits",
        "sentiment": sentiment,
        "cashtags": extract_cashtags(content),
        "published_at": msg.get("created_at", ""),
        "user_followers": msg.get("user", {}).get("followers", 0) or 0,
        "url": f"https://stocktwits.com/message/{msg.get('id')}",
        "content": content,
    }

    post["quality_score"] = stocktwits_quality_score(post)
    post["rejection_reasons"] = get_stocktwits_rejection_reasons(post)
    post["is_usable"] = keep_stocktwits_post(post)
    return post


def process_stocktwits(ticker: str, raw_messages: list[dict]) -> tuple[list[dict], list[dict]]:
    raw_posts = [build_stocktwits_post(ticker, msg) for msg in raw_messages]
    filtered_posts = [post for post in raw_posts if post["is_usable"]]
    return raw_posts, filtered_posts


def _fetch_hn_query(ticker: str, query: str, cutoff_epoch: int) -> list[dict]:
    endpoint = "https://hn.algolia.com/api/v1/search_by_date"
    params = {
        "query": query,
        "tags": "story",
        "hitsPerPage": 20,
        "numericFilters": f"created_at_i>{cutoff_epoch}",
    }
    try:
        resp = requests.get(endpoint, params=params, timeout=10)
    except requests.RequestException as e:
        print(f"HackerNews {ticker} query '{query}' failed: {e}")
        return []
    if resp.status_code != 200:
        print(f"HackerNews {ticker} query '{query}' failed: HTTP {resp.status_code}")
        return []
    hits = resp.json().get("hits", [])
    for hit in hits:
        hit["query_used"] = query
    return hits


def fetch_hackernews_raw(ticker: str) -> list[dict]:
    """Pull recent HN stories using ticker-specific product/company queries."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=HACKERNEWS_DAYS)
    cutoff_epoch = int(cutoff.timestamp())
    queries = HN_QUERIES[ticker]
    raw_hits: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(queries)) as pool:
        for hits in pool.map(lambda q: _fetch_hn_query(ticker, q, cutoff_epoch), queries):
            raw_hits.extend(hits)
    return raw_hits


def build_hackernews_post(ticker: str, hit: dict) -> dict:
    object_id = hit.get("objectID", "")
    title = clean_text(hit.get("title", ""))
    content = clean_text(hit.get("story_text") or title)

    post = {
        "ticker": ticker,
        "source": "hackernews",
        "object_id": object_id,
        "title": title,
        "query_used": hit.get("query_used", ""),
        "points": hit.get("points", 0) or 0,
        "num_comments": hit.get("num_comments", 0) or 0,
        "published_at": hit.get("created_at", ""),
        "url": hit.get("url") or f"https://news.ycombinator.com/item?id={object_id}",
        "content": content,
    }

    post["quality_score"] = hackernews_quality_score(post)
    post["rejection_reasons"] = get_hackernews_rejection_reasons(post)
    post["is_usable"] = keep_hackernews_story(post)
    return post


def process_hackernews(ticker: str, raw_hits: list[dict]) -> tuple[list[dict], list[dict]]:
    raw_posts = [build_hackernews_post(ticker, hit) for hit in raw_hits]
    filtered_posts = []
    seen_ids = set()

    for post in raw_posts:
        object_id = post.get("object_id", "")
        if not post["is_usable"] or not object_id or object_id in seen_ids:
            continue
        seen_ids.add(object_id)
        filtered_posts.append(post)

    return raw_posts, filtered_posts


def top_rejection_reasons(posts: list[dict]) -> str:
    counter = Counter()
    for post in posts:
        counter.update(post.get("rejection_reasons", []))

    if not counter:
        return "none"

    return ", ".join(f"{reason}={count}" for reason, count in counter.most_common(3))


def main() -> None:
    total_usable = 0

    for ticker in TICKERS:
        stocktwits_raw = fetch_stocktwits_raw(ticker)
        stocktwits_posts, stocktwits_filtered = process_stocktwits(ticker, stocktwits_raw)

        hackernews_raw = fetch_hackernews_raw(ticker)
        hackernews_posts, hackernews_filtered = process_hackernews(ticker, hackernews_raw)

        save_json(f"{RAW_DIR}/stocktwits_{ticker}.json", stocktwits_posts)
        save_json(f"{FILTERED_DIR}/stocktwits_{ticker}.json", stocktwits_filtered)
        save_json(f"{RAW_DIR}/hackernews_{ticker}.json", hackernews_posts)
        save_json(f"{FILTERED_DIR}/hackernews_{ticker}.json", hackernews_filtered)

        total_usable += len(stocktwits_filtered) + len(hackernews_filtered)

        print(f"\n{ticker} summary")
        print(f"  StockTwits raw: {len(stocktwits_posts)}")
        print(f"  StockTwits usable: {len(stocktwits_filtered)}")
        print(f"  HackerNews raw: {len(hackernews_posts)}")
        print(f"  HackerNews usable: {len(hackernews_filtered)}")
        print(f"  StockTwits top rejection reasons: {top_rejection_reasons(stocktwits_posts)}")
        print(f"  HackerNews top rejection reasons: {top_rejection_reasons(hackernews_posts)}")

    print(f"\nTotal usable social docs saved: {total_usable}")


if __name__ == "__main__":
    main()
