"""Single source of truth for AMIA model assignments.

Per-node rationale:
    PLANNER         GPT-OSS-120B  - strongest reasoning for query decomposition
    RESEARCHER      Llama 4 Scout - long context + high throughput for many docs
    ANALYST         Qwen3-32B     - best numerical / quantitative reasoning
    CRITIC          Gemini Flash  - different family from generators (no echo chamber)
    WRITER          Gemma 3 (local) - cheap polished prose, no API spend
    FALLBACK        Gemini Flash  - covers Groq TPD/TPM rate limits
"""

# --- Groq models (LiteLLM-style ids: provider/model) ---
PLANNER_MODEL = "groq/openai/gpt-oss-120b"
RESEARCHER_MODEL = "groq/meta-llama/llama-4-scout-17b-16e-instruct"
ANALYST_MODEL = "groq/qwen/qwen3-32b"
# SubQuestion engine is called as a tool from the Researcher, so latency
# matters more than depth.
RETRIEVE_MODEL_BARE = "llama-3.1-8b-instant"

# Bare ids (no provider prefix) for langchain_groq.ChatGroq, which only
# wants the model id without "groq/".
PLANNER_MODEL_BARE = "openai/gpt-oss-120b"

# --- Gemini ---
# Used for both the crew critic AND the planner fallback when Groq is rate limited.
CRITIC_MODEL = "gemini/gemini-3.1-flash-lite-preview"
GEMINI_FALLBACK_MODEL = "gemini/gemini-3.1-flash-lite-preview"
GEMINI_MODEL_BARE = "gemini-3.1-flash-lite-preview"  # for langchain_google_genai

# --- Local writer via Ollama ---
WRITER_MODEL = "ollama/gemma3:4b"
OLLAMA_BASE_URL = "http://localhost:11434"

# Used by crew._pick_writer_llm() when AMIA_WRITER=cloud, when Ollama is
# unreachable, when gemma3:4b is not pulled, or as a LiteLLM runtime fallback
# if the local writer crashes mid-run.
WRITER_FALLBACK_MODEL = GEMINI_FALLBACK_MODEL
