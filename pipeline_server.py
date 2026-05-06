"""Day 7: tiny Flask server that exposes the Python pipeline steps over HTTP.

n8n (running in Docker) cannot call python scripts directly, so it hits these
endpoints instead. Run this on the host Mac with the venv activated:

    python pipeline_server.py

Then n8n calls:
    POST http://host.docker.internal:8000/reindex
    POST http://host.docker.internal:8000/run-briefings
"""
import os
import sys
import subprocess
from flask import Flask, jsonify, request, Response

# Day 8: digest helpers parse markdown briefings into structured JSON
# so n8n can render proper email and Notion blocks instead of raw blobs.
from digest import load_today, build_email_html

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)


def _run_script(script_name: str, timeout_seconds: int):
    """Run a python script in the project dir using the same interpreter as this server.
    sys.executable means we automatically use whichever python launched the server,
    which means the venv if Top activated it before starting.
    """
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=PROJECT_DIR,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        # tail of stdout so the n8n response is small but useful for debugging
        return jsonify({
            "status": "ok",
            "script": script_name,
            "stdout_tail": result.stdout[-2000:],
        })
    except subprocess.CalledProcessError as e:
        return jsonify({
            "status": "error",
            "script": script_name,
            "returncode": e.returncode,
            "stderr_tail": e.stderr[-2000:],
        }), 500
    except subprocess.TimeoutExpired:
        return jsonify({"status": "timeout", "script": script_name}), 504


@app.post("/ingest-news")
def ingest_news():
    """Pull fresh news for all 5 tickers via NewsAPI."""
    return _run_script("ingest.py", timeout_seconds=300)  # 5 min cap


@app.post("/ingest-social")
def ingest_social():
    """Pull StockTwits + HackerNews for all 5 tickers."""
    return _run_script("social_ingest.py", timeout_seconds=300)


@app.post("/reindex")
def reindex():
    """Rebuild the Qdrant index from data/news + data/social."""
    return _run_script("retrieval.py", timeout_seconds=600)  # 10 min cap


@app.post("/run-briefings")
def run_briefings():
    """Run the LangGraph + Crew pipeline for all 5 tickers, save markdown to disk."""
    return _run_script("run_briefings.py", timeout_seconds=1800)  # 30 min cap


@app.get("/deliver")
def deliver():
    """Day 8: return today's briefings as structured JSON for n8n delivery workflows.

    Optional ?date=YYYY-MM-DD lets us replay a specific day, useful for testing
    without waiting for the morning cron to fire.
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


if __name__ == "__main__":
    # bind to 0.0.0.0 so Docker n8n can reach the server via host.docker.internal.
    # Docker Desktop's host-gateway routes through the host's network stack,
    # so even though Flask listens on all interfaces, only Docker containers
    # on this host (and other devices on the LAN) can reach it.
    app.run(host="0.0.0.0", port=8000, debug=False)
