"""CLI: loop the LangGraph + Crew pipeline over all 5 tickers, save to disk.

Usage:
    python -m amia.pipeline.run_briefings            # all 5 tickers
    python -m amia.pipeline.run_briefings TSLA NVDA  # subset
"""
import os
import json
import sys
import time
import traceback
from datetime import datetime

from amia.config import TICKERS
from amia.observability.tracing import build_config, flush as flush_tracing


def run_one(ticker: str, run_date: str) -> dict:
    """Run the full graph for one ticker and return briefing plus quality metadata.

    All 5 tickers in one run share session amia-<run_date> so they group in Langfuse.
    """
    from amia.pipeline.graph import app

    config = build_config(ticker=ticker, run_date=run_date)
    result = app.invoke({"ticker": ticker, "retries": 0, "sources": []}, config=config)
    return {
        "briefing": result["briefing"],
        "quality_warnings": result.get("quality_warnings", []),
    }


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


def save_run_summary(run_date: str, summary: dict) -> str:
    """Write machine-readable run metadata for monitoring and CI."""
    out_dir = os.path.join("briefings", run_date)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "_run_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    return out_path


def main():
    # let user pass tickers on CLI, otherwise run all 5
    args = sys.argv[1:]
    strict = "--strict" in args or os.getenv("AMIA_STRICT", "").strip() == "1"
    tickers = [a.upper() for a in args if a != "--strict"] or list(TICKERS)
    invalid = [t for t in tickers if t not in TICKERS]
    if invalid:
        print(f"Invalid ticker(s): {', '.join(invalid)}")
        print(f"Allowed tickers: {', '.join(TICKERS)}")
        sys.exit(2)

    run_date = datetime.now().strftime("%Y-%m-%d")
    started_at = datetime.now().isoformat(timespec="seconds")

    print(f"Running briefings for {len(tickers)} tickers on {run_date}")
    print(f"Tickers: {', '.join(tickers)}")
    if strict:
        print("Strict mode: warnings or failures return a non-zero exit code")
    print("=" * 60)

    successes = []
    failures = []
    warnings = []
    start = time.time()

    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] {ticker} starting...")
        ticker_start = time.time()
        try:
            result = run_one(ticker, run_date)
            path = save_briefing(ticker, result["briefing"], run_date)
            elapsed = time.time() - ticker_start
            quality_warnings = result.get("quality_warnings", [])
            if quality_warnings:
                print(
                    f"[{i}/{len(tickers)}] {ticker} done with warnings "
                    f"in {elapsed:.1f}s -> {path}"
                )
                for warning in quality_warnings:
                    print(f"  warning: {warning[:160]}")
                warnings.append({"ticker": ticker, "warnings": quality_warnings})
            else:
                print(f"[{i}/{len(tickers)}] {ticker} done in {elapsed:.1f}s -> {path}")
            successes.append({
                "ticker": ticker,
                "path": path,
                "elapsed_seconds": round(elapsed, 1),
                "quality_warnings": quality_warnings,
            })
        except Exception as e:
            elapsed = time.time() - ticker_start
            print(f"[{i}/{len(tickers)}] {ticker} FAILED in {elapsed:.1f}s: {e}")
            # full trace to a log file so we can debug later without spamming stdout
            err_dir = os.path.join("briefings", run_date)
            os.makedirs(err_dir, exist_ok=True)
            with open(os.path.join(err_dir, f"{ticker}_ERROR.log"), "w") as f:
                f.write(f"Error for {ticker}:\n{e}\n\n")
                f.write(traceback.format_exc())
            failures.append({
                "ticker": ticker,
                "error": str(e),
                "elapsed_seconds": round(elapsed, 1),
            })

        # Pause between tickers to stay under Groq's free-tier ~30 RPM limit.
        if i < len(tickers):
            time.sleep(5)

    total = time.time() - start
    print("\n" + "=" * 60)
    print(f"Run complete in {total:.1f}s")
    print(f"Succeeded: {len(successes)}/{len(tickers)}")
    for success in successes:
        label = " (warnings)" if success["quality_warnings"] else ""
        print(f"  {success['ticker']} -> {success['path']}{label}")
    if failures:
        print(f"Failed: {len(failures)}")
        for failure in failures:
            print(f"  {failure['ticker']}: {failure['error'][:120]}")

    summary = {
        "run_date": run_date,
        "started_at": started_at,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(total, 1),
        "strict": strict,
        "requested_tickers": tickers,
        "successes": successes,
        "warnings": warnings,
        "failures": failures,
    }
    summary_path = save_run_summary(run_date, summary)
    print(f"Run summary -> {summary_path}")

    # Without this, buffered Langfuse traces can be lost when python tears down the SDK.
    flush_tracing()

    if failures or (strict and warnings):
        sys.exit(1)


if __name__ == "__main__":
    main()
