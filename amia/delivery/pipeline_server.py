"""Tiny Flask server that exposes the Python pipeline steps over HTTP.

n8n (running in Docker) cannot call python scripts directly, so it hits
these endpoints instead. Run this on the host Mac with the venv activated:

    python -m amia.delivery.pipeline_server

Then n8n calls:
    POST http://host.docker.internal:8000/reindex
    POST http://host.docker.internal:8000/run-briefings
"""
import os
import sys
import subprocess
from pathlib import Path
from flask import Flask, jsonify, request, Response

from amia.delivery.digest import load_today, build_email_html

PROJECT_DIR = str(Path(__file__).resolve().parents[2])

app = Flask(__name__)


def _run_module(module: str, timeout_seconds: int):
    """Run an amia submodule in the project dir using this server's interpreter."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", module],
            cwd=PROJECT_DIR,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return jsonify({
            "status": "ok",
            "module": module,
            "stdout_tail": result.stdout[-2000:],
        })
    except subprocess.CalledProcessError as e:
        return jsonify({
            "status": "error",
            "module": module,
            "returncode": e.returncode,
            "stderr_tail": e.stderr[-2000:],
        }), 500
    except subprocess.TimeoutExpired:
        return jsonify({"status": "timeout", "module": module}), 504


@app.post("/ingest-news")
def ingest_news():
    """Pull fresh news for all 5 tickers via NewsAPI."""
    return _run_module("amia.ingest.news", timeout_seconds=300)


@app.post("/ingest-social")
def ingest_social():
    """Pull StockTwits + HackerNews for all 5 tickers."""
    return _run_module("amia.ingest.social", timeout_seconds=300)


@app.post("/reindex")
def reindex():
    """Rebuild the Qdrant index from data/news + data/social."""
    return _run_module("amia.retrieval.index", timeout_seconds=600)


@app.post("/run-briefings")
def run_briefings():
    """Run the LangGraph + Crew pipeline for all 5 tickers, save markdown to disk."""
    return _run_module("amia.pipeline.run_briefings", timeout_seconds=1800)


@app.get("/deliver")
def deliver():
    """Return today's briefings as structured JSON for n8n delivery workflows.

    Optional ?date=YYYY-MM-DD lets us replay a specific day, useful for
    testing without waiting for the morning cron to fire.
    """
    run_date = request.args.get("date")  # None -> today
    payload = load_today(run_date)
    return jsonify(payload)


@app.get("/deliver-html")
def deliver_html():
    """Same as /deliver but pre-rendered as a single HTML email body.

    n8n's Send Email node accepts HTML directly, so we hand it a finished string.
    Keeps the n8n workflow dumb: cron -> GET this URL -> pipe into email body.
    """
    run_date = request.args.get("date")
    payload = load_today(run_date)
    html = build_email_html(payload)
    subject = f"AMIA Daily Briefing - {payload['date']} ({payload['count']} tickers)"

    # return both subject and body so n8n can pluck them with Set node
    return jsonify({
        "subject": subject,
        "html": html,
        "date": payload["date"],
        "count": payload["count"],
    })


@app.get("/health")
def health():
    """n8n can hit this first to confirm the server is up before the real workflows fire."""
    return jsonify({"status": "ok"})


def main() -> None:
    # Bind to 0.0.0.0 so Docker n8n can reach the server via
    # host.docker.internal. Docker Desktop's host-gateway routes through the
    # host's network stack, so listening on all interfaces also exposes the
    # server to other devices on the LAN.
    app.run(host="0.0.0.0", port=8000, debug=False)


if __name__ == "__main__":
    main()
