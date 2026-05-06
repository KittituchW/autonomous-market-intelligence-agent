"""Single source of truth for AMIA model assignments.

Day 13 refactor: every model id used by the pipeline lives here so swapping
a model is a one-line change instead of a grep across graph.py + crew.py.

Model rationale per node:
    PLANNER         GPT-OSS-120B  - strongest reasoning for query decomposition
    RESEARCHER      Llama 4 Scout - long context + high throughput for many docs
    ANALYST         Qwen3-32B     - best numerical / quantitative reasoning
    CRITIC          Gemini Flash  - different family from generators (no echo chamber)
    WRITER          Gemma 3 (local) - cheap polished prose, no API spend
    FALLBACK        Gemini Flash  - covers Groq TPD/TPM rate limits

Switching the writer to gemma3:4b (from 1b) for noticeably better prose.
The 1b model kept producing stiff, repetitive briefings on Day 11 traces.
"""

# --- Groq models (LiteLLM-style ids: provider/model) ---
PLANNER_MODEL = "groq/openai/gpt-oss-120b"
RESEARCHER_MODEL = "groq/meta-llama/llama-4-scout-17b-16e-instruct"
ANALYST_MODEL = "groq/qwen/qwen3-32b"
# SubQuestion engine (LlamaIndex retrieve-side decomposition + light synthesis).
# Called as a tool from inside the Researcher agent, so latency matters more
# than depth -- the Researcher gets the final say on what to do with output.
RETRIEVE_MODEL_BARE = "llama-3.1-8b-instant"

# Bare ids (no provider prefix) for langchain_groq.ChatGroq, which only
# wants the model id without "groq/".
PLANNER_MODEL_BARE = "openai/gpt-oss-120b"

# --- Gemini ---
# Used for both the crew critic AND the planner fallback when Groq is rate
# limited. Top wants this model kept verbatim, do not change without asking.
CRITIC_MODEL = "gemini/gemini-3.1-flash-lite-preview"
GEMINI_FALLBACK_MODEL = "gemini/gemini-3.1-flash-lite-preview"
GEMINI_MODEL_BARE = "gemini-3.1-flash-lite-preview"  # for langchain_google_genai

# --- Local writer via Ollama ---
WRITER_MODEL = "ollama/gemma3:4b"
OLLAMA_BASE_URL = "http://localhost:11434"

# --- Writer fallback (when local Gemma is unavailable or too heavy) ---
# Used by crew._pick_writer_llm() in three cases:
#   1. AMIA_WRITER=cloud env var set explicitly
#   2. Ollama not running on OLLAMA_BASE_URL (pre-flight check fails)
#   3. gemma3:4b not in `ollama list` (model not pulled)
#   4. LiteLLM runtime fallback if Ollama times out / OOMs mid-run
# Aliased to the Gemini fallback so swapping providers is one line.
WRITER_FALLBACK_MODEL = GEMINI_FALLBACK_MODEL
