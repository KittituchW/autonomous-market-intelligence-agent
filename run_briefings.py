"""Day 4 CLI: loop the LangGraph + Crew pipeline over all 5 tickers, save to disk.

Usage:
    python run_briefings.py            # all 5 tickers
    python run_briefings.py TSLA NVDA  # subset
"""
import os
import sys
import time
import traceback
from datetime import datetime
from graph import app
from tracing import build_config, flush as flush_tracing

TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN"]


def run_one(ticker: str, run_date: str) -> str:
    """Run the full graph for one ticker and return the briefing markdown.

    Day 9 (v3): build_config returns the LangChain run config with the
    Langfuse callback plus metadata that sets session id, tags, and user id.
    All 5 tickers in one run share session amia-<run_date>.
    """
    config = build_config(ticker=ticker, run_date=run_date)
    result = app.invoke({"ticker": ticker, "retries": 0, "sources": []}, config=config)
    return result["briefing"]


def save_briefing(ticker: str, briefing: str, run_date: str) -> str:
    """Save the briefing to briefings/YYYY-MM-DD/<ticker>.md, return the path."""
    out_dir = os.path.join("briefings", run_date)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ticker}.md")

    # add a small header so the file is self-describing
    header = f"# {ticker} Daily Briefing\n\n_Generated: {run_date}_\n\n---\n\n"
    with open(out_path, "w") as f:
        f.write(header + briefing)

    return out_path


def main():
    # let user pass tickers on CLI, otherwise run all 5
    tickers = sys.argv[1:] if len(sys.argv) > 1 else TICKERS
    run_date = datetime.now().strftime("%Y-%m-%d")

    print(f"Running briefings for {len(tickers)} tickers on {run_date}")
    print(f"Tickers: {', '.join(tickers)}")
    print("=" * 60)

    successes = []
    failures = []
    start = time.time()

    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] {ticker} starting...")
        ticker_start = time.time()
        try:
            briefing = run_one(ticker, run_date)
            path = save_briefing(ticker, briefing, run_date)
            elapsed = time.time() - ticker_start
            print(f"[{i}/{len(tickers)}] {ticker} done in {elapsed:.1f}s -> {path}")
            successes.append((ticker, path))
        except Exception as e:
            elapsed = time.time() - ticker_start
            print(f"[{i}/{len(tickers)}] {ticker} FAILED in {elapsed:.1f}s: {e}")
            # full trace to a log file so we can debug later without spamming stdout
            err_dir = os.path.join("briefings", run_date)
            os.makedirs(err_dir, exist_ok=True)
            with open(os.path.join(err_dir, f"{ticker}_ERROR.log"), "w") as f:
                f.write(f"Error for {ticker}:\n{e}\n\n")
                f.write(traceback.format_exc())
            failures.append((ticker, str(e)))

        # short pause between tickers to dodge Groq rate limits (free tier is ~30 RPM)
        # each ticker uses 4 cloud calls, so 5 tickers = 20 calls in a few minutes
        if i < len(tickers):
            time.sleep(5)

    total = time.time() - start
    print("\n" + "=" * 60)
    print(f"Run complete in {total:.1f}s")
    print(f"Succeeded: {len(successes)}/{len(tickers)}")
    for t, p in successes:
        print(f"  {t} -> {p}")
    if failures:
        print(f"Failed: {len(failures)}")
        for t, err in failures:
            print(f"  {t}: {err[:120]}")

    # Day 9: force any buffered Langfuse traces to upload before we exit.
    # without this, the last trace can be lost when python tears down the SDK.
    flush_tracing()


if __name__ == "__main__":
    main()
