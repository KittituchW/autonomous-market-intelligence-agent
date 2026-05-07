"""One-command AMIA pipeline orchestrator.

Default run:
    python -m amia.main

Common dev run:
    python -m amia.main --skip-ingest --skip-reindex NVDA
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from amia.config import TICKERS as VALID_TICKERS


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_STEPS = ["ingest-news", "ingest-social", "reindex", "briefings"]
STEP_MODULES = {
    "ingest-news": "amia.ingest.news",
    "ingest-social": "amia.ingest.social",
    "reindex": "amia.retrieval.index",
    "briefings": "amia.pipeline.run_briefings",
}


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the AMIA ingestion, indexing, and briefing pipeline.",
    )
    parser.add_argument(
        "tickers",
        nargs="*",
        help="Optional ticker subset for the briefing step.",
    )
    parser.add_argument(
        "--steps",
        default=",".join(DEFAULT_STEPS),
        help=(
            "Comma-separated pipeline steps to run in order. "
            f"Allowed: {', '.join(DEFAULT_STEPS)}."
        ),
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip NewsAPI, StockTwits, and HackerNews ingestion.",
    )
    parser.add_argument(
        "--skip-reindex",
        action="store_true",
        help="Skip rebuilding the Qdrant vector index.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail the briefing step when quality warnings are emitted.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running later steps after a step fails.",
    )
    return parser.parse_args(argv)


def _normalise_tickers(raw_tickers: list[str]) -> list[str]:
    tickers = [ticker.upper() for ticker in raw_tickers]
    invalid = [ticker for ticker in tickers if ticker not in VALID_TICKERS]
    if invalid:
        allowed = ", ".join(VALID_TICKERS)
        raise ValueError(f"Invalid ticker(s): {', '.join(invalid)}. Allowed: {allowed}")
    return tickers


def _selected_steps(args: argparse.Namespace) -> list[str]:
    steps = [step.strip() for step in args.steps.split(",") if step.strip()]
    invalid = [step for step in steps if step not in DEFAULT_STEPS]
    if invalid:
        allowed = ", ".join(DEFAULT_STEPS)
        raise ValueError(f"Invalid step(s): {', '.join(invalid)}. Allowed: {allowed}")
    if args.skip_ingest:
        steps = [step for step in steps if step not in ("ingest-news", "ingest-social")]
    if args.skip_reindex:
        steps = [step for step in steps if step != "reindex"]
    return steps


def _command_for_step(step: str, tickers: list[str], strict: bool) -> list[str]:
    command = [sys.executable, "-m", STEP_MODULES[step]]
    if step == "briefings":
        if strict:
            command.append("--strict")
        command.extend(tickers)
    return command


def _write_summary(summary: dict) -> Path:
    run_date = summary["run_date"]
    out_dir = PROJECT_DIR / "briefings" / run_date
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "_orchestrator_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return out_path


def run(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    try:
        tickers = _normalise_tickers(args.tickers)
    except ValueError as exc:
        print(exc)
        return 2

    try:
        steps = _selected_steps(args)
    except ValueError as exc:
        print(exc)
        return 2
    run_date = datetime.now().strftime("%Y-%m-%d")
    started_at = datetime.now().isoformat(timespec="seconds")

    print(f"AMIA pipeline starting on {run_date}")
    print(f"Steps: {', '.join(steps) if steps else 'none'}")
    print(f"Briefing tickers: {', '.join(tickers) if tickers else 'all'}")
    print("=" * 60)

    results = []
    failed = False
    start = time.time()

    for index, step in enumerate(steps, 1):
        command = _command_for_step(step, tickers, args.strict)
        print(f"\n[{index}/{len(steps)}] {step} starting")
        print("Command:", " ".join(command))
        step_start = time.time()
        completed = subprocess.run(command, cwd=PROJECT_DIR, check=False)
        elapsed = time.time() - step_start
        result = {
            "step": step,
            "command": command,
            "returncode": completed.returncode,
            "elapsed_seconds": round(elapsed, 1),
        }
        results.append(result)

        if completed.returncode == 0:
            print(f"[{index}/{len(steps)}] {step} done in {elapsed:.1f}s")
            continue

        failed = True
        print(f"[{index}/{len(steps)}] {step} FAILED with exit code {completed.returncode}")
        if not args.continue_on_error:
            break

    total = time.time() - start
    summary = {
        "run_date": run_date,
        "started_at": started_at,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(total, 1),
        "requested_tickers": tickers or VALID_TICKERS,
        "steps": steps,
        "strict": args.strict,
        "continue_on_error": args.continue_on_error,
        "results": results,
        "status": "failed" if failed else "ok",
    }
    summary_path = _write_summary(summary)

    print("\n" + "=" * 60)
    print(f"AMIA pipeline {'failed' if failed else 'completed'} in {total:.1f}s")
    print(f"Orchestrator summary -> {summary_path}")
    return 1 if failed else 0


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
