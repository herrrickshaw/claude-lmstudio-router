# Claude + LM Studio Router

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/herrrickshaw/claude-lmstudio-router/blob/main/notebooks/colab_test.ipynb)

This repo has **two tools**:

1. **`router.py`** — a one-shot CLI. Sends a single prompt to the **Claude API
   first**, and falls back to a **local model in LM Studio** if Claude is
   rate-limited, overloaded, or unreachable. (Documented below.)

2. **`claude_code_proxy.py`** — a translation **proxy** that makes the whole
   **Claude Code agent** run on a local LM Studio model, fully offline. Use this
   on local RAG machines where you want Claude Code itself (its tools, file
   edits, multi-turn loop) driven by a local model instead of the cloud.
   See [Running Claude Code on LM Studio](#running-claude-code-on-lm-studio).

Both run **on your own machine** — they talk to LM Studio's local server, which
only exists on your laptop, so they can't run inside a hosted chat session.

---

## One-time setup

**1. Install the Anthropic Python SDK**
```
pip install anthropic --break-system-packages
```
(Drop `--break-system-packages` if you're using a virtual environment instead.)

**2. Get an Anthropic API key**
- This is **separate from your claude.ai login** — different account system, different billing.
- Go to console.anthropic.com (also reachable via platform.claude.com) → Settings → API Keys → Create Key.
- Requires a payment method on file; usage is billed per token, no ongoing free tier.
- The key is shown only once — copy it immediately.

**3. Set the key as an environment variable**
- Windows (PowerShell): `setx ANTHROPIC_API_KEY "sk-ant-...."` (then close/reopen your terminal)
- Mac/Linux: add `export ANTHROPIC_API_KEY="sk-ant-...."` to your `~/.zshrc` or `~/.bashrc`, then `source` it.
- Never hardcode the key inside `router.py`.

**4. Start LM Studio's local server**
- Open LM Studio → load whichever model you want as your local fallback.
- Go to the Developer tab → "Start Server" (default address `http://localhost:1234`).
- Leave LM Studio running in the background whenever you want fallback to work.

---

## Usage

**Simple prompt, default behavior (Claude first, local fallback):**
```
python router.py --prompt "Summarize the key risks in this paragraph: ..."
```

**Feed it a file (e.g. a chunk of trade data or a report) plus a task instruction:**
```
python router.py --task "Extract HSN codes and values into a clean table" --input data.txt
```

**Save the result to a file instead of printing it:**
```
python router.py --prompt "Draft an executive summary..." --output summary.txt
```

**Force local only (e.g. you deliberately want to save Claude quota today):**
```
python router.py --prompt "..." --force-local
```

**Force Claude only (skip the local fallback entirely):**
```
python router.py --prompt "..." --force-claude
```

**If LM Studio is using a non-default port or you loaded multiple models:**
```
python router.py --prompt "..." --local-base-url "http://localhost:1234/v1" --local-model "qwen2.5-14b-instruct"
```

---

## How the fallback decision works

1. Tries Claude API (`claude-sonnet-4-6` by default — edit `CLAUDE_MODEL` in `router.py` to change).
2. Retries up to 2 times on rate limits (429) or server overload (529) with short backoff.
3. If Claude still fails (or the key is missing/invalid), automatically calls your local
   LM Studio model with the *same* prompt and system instruction.
4. Every call is logged to `router_log.jsonl` (one JSON line per call) so you can see
   which engine actually answered, and when Claude was unavailable.

---

## Notes and limits

- **Quality difference is real.** The local model will generally be weaker than Claude,
  especially for nuanced finishing work (e.g. exec-summary polish, sensitivity-table
  framing). Treat local output as a draft pass when it's used as a fallback, and consider
  re-running through Claude later to upgrade it once your quota resets.
- **This script doesn't manage LM Studio for you** — you still load the model and click
  "Start Server" yourself before running it.
- **No data leaves your machine except to Anthropic's API** when Claude is used. The
  local fallback path is fully offline.
- Edit the `CONFIG` block at the top of `router.py` to change default model names,
  retry counts, or the log file path.

---

# Running Claude Code on LM Studio

`claude_code_proxy.py` is a small local server that lets the **Claude Code agent
CLI** run against a model served by **LM Studio** — fully offline. Ideal for a
local RAG machine where you don't want code or data leaving the box.

### How it works

Claude Code speaks the **Anthropic Messages API**. LM Studio speaks the
**OpenAI Chat Completions API**. The proxy sits between them:

```
Claude Code  --(Anthropic /v1/messages)-->  proxy  --(OpenAI /v1/chat)-->  LM Studio
             <--    Anthropic SSE        --        <--   OpenAI SSE     --
```

It translates messages, tools, streaming, and tool calls in both directions, so
to Claude Code the proxy looks exactly like `api.anthropic.com`.

### Setup

**1. Install deps (once):**
```
pip install fastapi uvicorn httpx --break-system-packages
```
(drop `--break-system-packages` inside a venv)

**2. Load a TOOL-CAPABLE model in LM Studio and start its server.**
This is the single most important step. Claude Code lives on tool calls
(Read, Edit, Bash, etc.), so the local model **must support function/tool
calling**, or Claude Code can't edit files or run commands. Good choices that
fit modest RAM:
- `qwen2.5-coder-7b-instruct` / `qwen2.5-7b-instruct`
- `llama-3.1-8b-instruct`
- larger if you have the memory (14B–32B noticeably better at agentic work)

Pure "reasoning" models (DeepSeek-R1, some Gemma/Nemotron "thinking" builds)
often **don't** emit tool calls in LM Studio — avoid them as your Claude Code
backend even though they're fine for plain chat.

Then: LM Studio → Developer tab → **Start Server** (default `http://localhost:1234`).

**3. Start the proxy** (in its own terminal):
```
python claude_code_proxy.py
# listens on http://127.0.0.1:8787, forwards to LM Studio
# set LMSTUDIO_MODEL if you have several models loaded:
LMSTUDIO_MODEL="qwen2.5-coder-7b-instruct" python claude_code_proxy.py
```
Check it: `curl http://127.0.0.1:8787/health` should report `"lmstudio":"ok"`
and list your loaded models.

**4. Point Claude Code at the proxy** and run it (in another terminal):
```
export ANTHROPIC_BASE_URL="http://127.0.0.1:8787"
export ANTHROPIC_API_KEY="local"     # any non-empty value; the proxy ignores it
claude
```
Or just use the bundled helper, which sets those vars and sanity-checks the
proxy first:
```
./run_local_claude_code.sh
```

### Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `LMSTUDIO_BASE_URL` | `http://localhost:1234/v1` | LM Studio endpoint |
| `LMSTUDIO_MODEL` | *(empty = LM Studio's loaded model)* | Force a specific model id |
| `LMSTUDIO_SMALL_MODEL` | = `LMSTUDIO_MODEL` | Model for Claude Code's small "haiku" calls (titles/summaries) |
| `PROXY_HOST` / `PROXY_PORT` | `127.0.0.1` / `8787` | Where the proxy listens |
| `PROXY_REASONING` | `drop` | Reasoning-model chain-of-thought handling: `drop` (hide), `think` (show as thinking blocks), `text` (inline into answer) |
| `PROXY_LOG` | *(off)* | `1` to log each translated request to stderr |
| `LMSTUDIO_TIMEOUT` | `600` | Per-request timeout (seconds) |

### Notes and limits

- **Tool-calling support is mandatory** for real agentic use (see step 2).
  Without it Claude Code can chat but can't edit files or run commands.
- **Quality and speed.** A local 7–8B model is much weaker and slower than the
  cloud Claude models, and will make more mistakes on multi-step tasks. Keep
  changes small and review them.
- **No vision.** Image blocks are dropped (replaced with a placeholder); most
  local LM Studio setups aren't multimodal through this path.
- **Fully offline.** When using the proxy, nothing leaves the machine — every
  request goes only to your local LM Studio. (Unlike `router.py`, there is no
  Claude/cloud path here at all.)
- **The proxy doesn't manage LM Studio for you** — you still load the model and
  click "Start Server" yourself.
