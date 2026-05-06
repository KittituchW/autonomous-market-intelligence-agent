# Day 9 Setup: Langfuse Tracing (Cloud Free Tier)

Goal: every plan, retrieve, write, and CrewAI agent call shows up as a span in the Langfuse UI so you can see exactly where the pipeline burns tokens or produces garbage. Set up takes ~10 min.

## What got built

1. `tracing.py` — `get_handler(ticker, run_date)` returns a LangChain CallbackHandler tagged with the ticker, plus auto-wires LiteLLM so CrewAI calls get traced too. `flush()` makes sure buffered traces upload before the process exits.
2. `graph.py` — smoke test now attaches the handler.
3. `run_briefings.py` — every ticker invoke ships with a handler, all 5 grouped under one session called `amia-YYYY-MM-DD` so you can see one run in one view.
4. `requirements.txt` — added `langfuse>=2.50.0,<3.0`.

## Step 1: Sign up + create a project

1. Go to https://cloud.langfuse.com
2. Sign up with Google or email. Free tier has 50k events/month, way more than you'll use.
3. Create a new organization called `Top` or whatever you like.
4. Create a project called `amia`.
5. Project Settings → API keys → "Create new API keys". Copy:
   - Public key (`pk-lf-...`)
   - Secret key (`sk-lf-...`)
   - Host: `https://cloud.langfuse.com`

## Step 2: Add keys to .env

Open your `.env` and append:

```
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LANGFUSE_HOST=https://cloud.langfuse.com
```

If any of those are missing, `tracing.py` no-ops and prints `Langfuse keys missing, tracing disabled this run`. Useful for dev work where you don't want noise in your project.

## Step 3: Install the lib

```bash
# in the project dir, with venv active
pip install -r requirements.txt
```

This pulls in `langfuse`. The LiteLLM hook is built into LiteLLM itself, no extra install.

## Step 4: Smoke test on one ticker

```bash
python graph.py
```

Watch the terminal for:
1. The plan, retrieve, and write phases scrolling past as before.
2. No `Langfuse keys missing` line (means tracing is on).
3. A final briefing printed at the bottom.

Now check the UI:
1. Open https://cloud.langfuse.com
2. Project `amia` → Traces tab.
3. You should see a trace with tag `amia` and `tsla`.
4. Click into it. You'll see a tree view:
   - Root span: the LangGraph invoke
   - Child: planner LLM call (ChatGroq llama-3.3-70b)
   - Child: retrieve_node (no LLM call, just a function span)
   - Child: write_node containing the CrewAI calls (researcher, analyst, critic, writer)

## Step 5: Run the full 5-ticker pipeline

```bash
python run_briefings.py
```

In Langfuse, sort traces by recent. You should see 5 traces grouped under session `amia-2026-05-02` (or whatever today is). Each trace tagged with its ticker.

## Step 6: Find one weak prompt or wasteful retrieval

This is the actual point of Day 9. Open one trace and look for:

1. **Big input token count, small output**
   The agent got dumped a wall of context but only used 3 sentences. Fix on Day 11 by tightening the prompt or filtering the retrieval.

2. **Same retrieval running twice**
   Probably from the retry edge if context came back empty. If it always retries on a specific ticker, the retrieval is broken for that one.

3. **Critic agent saying "looks good"**
   Check the critic spans. If it never pushes back, the prompt is too soft. Fix on Day 11.

4. **Writer (Gemma 3 4B) producing weak prose**
   Read the writer span output. If it's repetitive or shallow, the prompt or model is the bottleneck.

5. **One agent way slower than the others**
   Latency column on each span. If the analyst is taking 30s+, something in the prompt is expensive or causing it to retry tools.

Note one issue with a sentence in `notes/day9_findings.md` (just freeform notes for yourself). On Day 11 you'll fix it and re-run the eval.

## Common things that break

1. **No traces appear in the UI**
   - Check `python -c "from tracing import _enabled; print(_enabled())"`. If False, your env vars aren't loaded. Check `.env` and that you ran from the project dir.
   - The flush at the end of `run_briefings.py` is what guarantees upload. If you Ctrl+C mid-run the last trace can be lost.

2. **Traces show LangGraph + LangChain calls but no CrewAI calls**
   - LiteLLM hook didn't activate. Run `python -c "import litellm; print(litellm.success_callback)"`. Should include `'langfuse'`.
   - If empty, `tracing._wire_litellm()` failed. Check the warning printed at startup.

3. **Traces upload but ticker tag is wrong**
   - You called `app.invoke` somewhere without going through `run_one(ticker)`. Make sure all entry points pass `config={"callbacks": [handler]}`.

4. **Token counts on Ollama spans show 0**
   - LiteLLM doesn't always parse Ollama responses for token usage. Real cost still tracked from cloud calls (Groq, Gemini). Expected.

## Day 9 EOD checklist

- [ ] Langfuse Cloud project `amia` created, keys in `.env`
- [ ] `pip install -r requirements.txt` ran clean
- [ ] `python graph.py` smoke test produces a trace in the UI
- [ ] `python run_briefings.py` produces 5 traces grouped under one session
- [ ] All three frameworks visible in trace tree (LangGraph nodes, LangChain LLM, CrewAI agents)
- [ ] One specific weakness noted in `notes/day9_findings.md`

## What's next (Day 10 preview)

Eval set: 10 questions with known answers, scored on retrieval (right source found) and reasoning (right conclusion). Saved as `evals/eval_set_v1.json`. Day 9 traces tell you *what* is weak; Day 10 tells you *how weak* with a number you can track.
