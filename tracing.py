"""Day 9: Langfuse v3 tracing for the whole pipeline.

We trace 3 layers from one place:

1. LangGraph state machine: pass the CallbackHandler in `config["callbacks"]`
   on every `app.invoke(...)` call. Captures plan, retrieve, write nodes plus
   the planner LLM call (LangChain ChatGroq).

2. LangChain LLM calls anywhere else: same handler propagates through chains.

3. CrewAI: CrewAI 0.51 uses LiteLLM under the hood. LiteLLM ships a built-in
   Langfuse hook turned on with litellm.success_callback = ["langfuse"].

Why one shared file: rest of the codebase calls `tracing.build_config(ticker)`
and never has to know which Langfuse SDK version is in use.

v3 vs v2 note: in v2 the CallbackHandler took session_id, tags, metadata as
constructor args. In v3 those moved to the LangChain run config under
metadata keys with the `langfuse_` prefix. We hide that detail in build_config.
"""
import os
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def _enabled() -> bool:
    """Return True only if all 3 Langfuse env vars are present.

    If any are missing, tracing silently no-ops so dev environments without
    keys do not blow up.
    """
    return all(
        os.getenv(k)
        for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST")
    )


# CrewAI -> LiteLLM -> Langfuse hook.
#
# Off by default. As of LiteLLM 1.83 + Langfuse 3.x the built-in "langfuse"
# callback string still passes a v2-only kwarg (sdk_integration) into the
# Langfuse client constructor and crashes during init. The error is non-blocking
# so the pipeline still runs, but it spams the log on every invoke.
#
# Opt in by setting LANGFUSE_LITELLM_CALLBACK in .env to one of:
#   langfuse        — original integration, currently broken with Langfuse v3.
#   langfuse_otel   — OpenTelemetry-based, may work with v3 once both sides ship the fix.
# Leave unset to skip wiring entirely. CrewAI agent traces will then show up
# in the CrewAI Plus ephemeral trace link printed at the end of every run.
_litellm_wired = False


def _wire_litellm():
    global _litellm_wired
    if _litellm_wired or not _enabled():
        return

    callback_name = os.getenv("LANGFUSE_LITELLM_CALLBACK", "").strip()
    if not callback_name:
        # default path: do not touch litellm
        _litellm_wired = True
        return

    try:
        import litellm

        if callback_name not in (litellm.success_callback or []):
            litellm.success_callback = (litellm.success_callback or []) + [callback_name]
        if callback_name not in (litellm.failure_callback or []):
            litellm.failure_callback = (litellm.failure_callback or []) + [callback_name]
        print(f"[tracing] litellm callback wired: {callback_name}")
        _litellm_wired = True
    except Exception as e:
        # do not let tracing kill the pipeline. log and move on.
        print(f"[tracing] litellm wiring skipped: {e}")


# Global Langfuse client. v3 needs a configured client somewhere in the process
# for the CallbackHandler to find creds.
_client = None


def _ensure_client():
    global _client
    if _client is not None or not _enabled():
        return _client
    from langfuse import Langfuse

    _client = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST"),
    )
    return _client


def get_handler():
    """Return a v3 CallbackHandler, or None if Langfuse is not configured.

    In v3 the handler itself is bare. Session id, tags, and user id ride on
    the LangChain run config under the langfuse_ metadata keys. Use
    build_config(ticker, run_date) to produce a ready-to-use config.
    """
    if not _enabled():
        print("[tracing] Langfuse keys missing, tracing disabled this run")
        return None

    _wire_litellm()
    _ensure_client()

    # late import so the project still runs when langfuse is not installed
    from langfuse.langchain import CallbackHandler

    return CallbackHandler()


def build_config(
    ticker: Optional[str] = None,
    run_date: Optional[str] = None,
    session_id: Optional[str] = None,
) -> dict:
    """Return a LangChain run config with callbacks + Langfuse metadata.

    Use:
        cfg = build_config(ticker="TSLA")
        result = app.invoke(state, config=cfg)

    Behavior:
        - 5 tickers run on the same day get session id `amia-YYYY-MM-DD`,
          so they show up grouped in the Langfuse Sessions tab.
        - Each invoke is tagged with `amia` and the lowercase ticker so you
          can filter on a single ticker across days.
        - When Langfuse is not configured returns an empty dict, which
          LangChain treats as no callbacks.
    """
    handler = get_handler()
    if handler is None:
        return {}

    if run_date is None:
        run_date = datetime.now().strftime("%Y-%m-%d")
    if session_id is None:
        session_id = f"amia-{run_date}"

    tags = ["amia"]
    if ticker:
        tags.append(ticker.lower())

    return {
        "callbacks": [handler],
        "metadata": {
            # v3 honors these specific keys on the run metadata
            "langfuse_session_id": session_id,
            "langfuse_tags": tags,
            "langfuse_user_id": "top",
            # plain extras you might want for filtering
            "ticker": ticker,
            "run_date": run_date,
        },
    }


def flush():
    """Force buffered traces to upload before the process exits.

    Langfuse batches HTTP posts. Without flush the last few traces can be
    lost when python tears down the SDK. Call at the end of run_briefings.py.
    """
    if not _enabled():
        return
    try:
        client = _ensure_client()
        if client:
            client.flush()
    except Exception as e:
        print(f"[tracing] flush failed: {e}")


if __name__ == "__main__":
    # quick sanity check
    print("Langfuse enabled:", _enabled())
    h = get_handler()
    print("Handler:", type(h).__name__ if h else None)
    cfg = build_config(ticker="TSLA")
    print("Config keys:", list(cfg.keys()))
    print("Metadata:", cfg.get("metadata"))
