#!/usr/bin/env python3
"""
claude_lmstudio_router.py
--------------------------
Routes a single prompt to the Claude API first. If Claude is unavailable
(rate-limited, overloaded, no/invalid key, or you force local mode),
falls back to a local model served by LM Studio.

USAGE (after editing the CONFIG block below, or via CLI flags):

    python router.py --task "Summarize this trade dataset" --input mydata.txt
    python router.py --prompt "Draft an executive summary of..." --force-local
    python router.py --prompt "..." --output result.txt

Run `python router.py --help` for all options.

SETUP CHECKLIST (do this once):
  1. pip install anthropic --break-system-packages   (or in a venv: pip install anthropic)
  2. Get an Anthropic API key at https://console.anthropic.com (Settings > API Keys).
     This is SEPARATE from your claude.ai login and requires a payment method on file.
  3. Set it as an environment variable (don't hardcode it in this file):
       Windows (PowerShell):  setx ANTHROPIC_API_KEY "sk-ant-...."
       Mac/Linux:             export ANTHROPIC_API_KEY="sk-ant-...."
     Close and reopen your terminal after setting it on Windows.
  4. Open LM Studio, load a model, and start the local server
     (LM Studio > Developer tab > "Start Server" — default port 1234).
  5. Note the model name LM Studio shows you (e.g. "qwen2.5-14b-instruct") and
     put it in LOCAL_MODEL below, or pass --local-model on the command line.
"""

import argparse
import os
import sys
import time
import json
import urllib.request
import urllib.error

# ============================================================
# CONFIG -- edit these defaults, or override via CLI flags
# ============================================================

CLAUDE_MODEL = "claude-sonnet-4-6"          # Claude model to call
CLAUDE_MAX_TOKENS = 4096

LOCAL_BASE_URL = "http://localhost:1234/v1"  # LM Studio's default OpenAI-compatible endpoint
LOCAL_MODEL = "local-model"                  # LM Studio ignores this if only one model is loaded,
                                              # but set it to match what LM Studio reports if you
                                              # load multiple models at once.

MAX_RETRIES_CLAUDE = 2          # quick retries before giving up on Claude for this call
RETRY_BACKOFF_SECONDS = 3       # base backoff between Claude retries

LOG_FILE = "router_log.jsonl"   # appends one JSON line per call: which engine answered, when, etc.

# ============================================================


def log_call(engine: str, prompt_preview: str, success: bool, note: str = ""):
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "engine": engine,
        "success": success,
        "note": note,
        "prompt_preview": prompt_preview[:120],
    }
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass  # logging should never crash the actual task


def call_claude(prompt: str, system: str = None) -> str:
    """Calls the Claude API. Raises on failure so the router can decide to fall back."""
    try:
        import anthropic
    except ImportError:
        raise RuntimeError(
            "The 'anthropic' package isn't installed. Run: "
            "pip install anthropic --break-system-packages"
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Get a key at console.anthropic.com and set it as an env var."
        )

    client = anthropic.Anthropic(api_key=api_key)

    last_error = None
    for attempt in range(1, MAX_RETRIES_CLAUDE + 1):
        try:
            kwargs = {
                "model": CLAUDE_MODEL,
                "max_tokens": CLAUDE_MAX_TOKENS,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system:
                kwargs["system"] = system
            response = client.messages.create(**kwargs)
            text_parts = [block.text for block in response.content if block.type == "text"]
            return "\n".join(text_parts)
        except anthropic.RateLimitError as e:
            last_error = e
            print(f"  [Claude] Rate limited (attempt {attempt}/{MAX_RETRIES_CLAUDE}). "
                  f"Backing off {RETRY_BACKOFF_SECONDS * attempt}s...", file=sys.stderr)
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
        except anthropic.APIStatusError as e:
            # Covers 529 overloaded, 500s, etc. Treat as retryable once, then bail to local.
            last_error = e
            print(f"  [Claude] API error ({e.status_code}). Attempt {attempt}/{MAX_RETRIES_CLAUDE}.",
                  file=sys.stderr)
            time.sleep(RETRY_BACKOFF_SECONDS)
        except anthropic.APIConnectionError as e:
            last_error = e
            print(f"  [Claude] Connection error: {e}", file=sys.stderr)
            time.sleep(RETRY_BACKOFF_SECONDS)

    raise RuntimeError(f"Claude API failed after {MAX_RETRIES_CLAUDE} attempts: {last_error}")


def call_local(prompt: str, system: str = None, base_url: str = None, model: str = None) -> str:
    """Calls LM Studio's local OpenAI-compatible chat completions endpoint."""
    base_url = base_url or LOCAL_BASE_URL
    model = model or LOCAL_MODEL

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0.7,
    }).encode("utf-8")

    url = f"{base_url.rstrip('/')}/chat/completions"
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not reach LM Studio at {url}. Is LM Studio running with its local "
            f"server started (Developer tab > Start Server)? Underlying error: {e}"
        )
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        raise RuntimeError(f"LM Studio returned an unexpected response shape: {e}")


def route(prompt: str, system: str = None, force_local: bool = False,
          force_claude: bool = False, local_base_url: str = None,
          local_model: str = None) -> tuple:
    """
    Returns (result_text, engine_used).
    Default behavior: try Claude, fall back to local on any failure.
    """
    preview = prompt[:120].replace("\n", " ")

    if force_local:
        print("Forcing local model (LM Studio)...")
        text = call_local(prompt, system, local_base_url, local_model)
        log_call("local", preview, True, "forced")
        return text, "local"

    if force_claude:
        print(f"Forcing Claude API ({CLAUDE_MODEL})...")
        text = call_claude(prompt, system)
        log_call("claude", preview, True, "forced")
        return text, "claude"

    # Default: Claude first, local fallback
    try:
        print(f"Trying Claude API ({CLAUDE_MODEL})...")
        text = call_claude(prompt, system)
        log_call("claude", preview, True)
        return text, "claude"
    except RuntimeError as e:
        print(f"Claude unavailable: {e}", file=sys.stderr)
        print("Falling back to local LM Studio model...")
        try:
            text = call_local(prompt, system, local_base_url, local_model)
            log_call("local", preview, True, f"fallback_after: {e}")
            return text, "local"
        except RuntimeError as local_e:
            log_call("none", preview, False, f"both failed: claude={e} | local={local_e}")
            raise RuntimeError(
                f"Both Claude and local LM Studio failed.\n"
                f"  Claude error: {e}\n"
                f"  LM Studio error: {local_e}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Route a prompt to Claude API, falling back to a local LM Studio model."
    )
    parser.add_argument("--prompt", type=str, help="The prompt text directly.")
    parser.add_argument("--task", type=str,
                         help="A short task instruction, combined with --input file content if given.")
    parser.add_argument("--input", type=str, help="Path to a text file whose contents get appended to the prompt.")
    parser.add_argument("--system", type=str, default=None, help="Optional system prompt.")
    parser.add_argument("--output", type=str, default=None, help="Path to write the result to. Prints to stdout if omitted.")
    parser.add_argument("--force-local", action="store_true", help="Skip Claude entirely, use LM Studio only.")
    parser.add_argument("--force-claude", action="store_true", help="Skip local fallback, use Claude only.")
    parser.add_argument("--local-base-url", type=str, default=None, help=f"Override LM Studio URL (default: {LOCAL_BASE_URL}).")
    parser.add_argument("--local-model", type=str, default=None, help=f"Override LM Studio model name (default: {LOCAL_MODEL}).")

    args = parser.parse_args()

    if not args.prompt and not args.task:
        parser.error("Provide either --prompt or --task (optionally with --input).")

    if args.prompt:
        prompt = args.prompt
    else:
        prompt = args.task
        if args.input:
            try:
                with open(args.input, "r", encoding="utf-8") as f:
                    file_content = f.read()
                prompt = f"{args.task}\n\n---\n\n{file_content}"
            except OSError as e:
                print(f"Could not read --input file: {e}", file=sys.stderr)
                sys.exit(1)

    try:
        result, engine = route(
            prompt,
            system=args.system,
            force_local=args.force_local,
            force_claude=args.force_claude,
            local_base_url=args.local_base_url,
            local_model=args.local_model,
        )
    except RuntimeError as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n--- Answered by: {engine.upper()} ---\n", file=sys.stderr)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"Result written to {args.output} (engine: {engine})")
    else:
        print(result)


if __name__ == "__main__":
    main()
