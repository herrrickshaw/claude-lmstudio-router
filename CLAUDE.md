# Project context for Claude Code

This repo holds two LM-Studio-adjacent tools:
1. `router.py` — a one-shot CLI that sends a prompt to the Claude API first and
   falls back to a local LM Studio model if Claude is rate-limited, overloaded,
   or unavailable.
2. `claude_code_proxy.py` — a long-running Anthropic-API-compatible proxy that
   lets the whole Claude Code agent run against a local LM Studio model, fully
   offline (for local RAG machines). It translates the Anthropic Messages API
   (what Claude Code emits) to/from LM Studio's OpenAI Chat Completions API,
   including system prompts, tools, tool calls, streaming SSE, and reasoning.

## Key files
- `router.py` — the one-shot CLI fallback tool.
- `claude_code_proxy.py` — the Claude-Code-to-LM-Studio translation proxy (FastAPI).
- `run_local_claude_code.sh` — helper that sets ANTHROPIC_BASE_URL/KEY and launches `claude`.
- `README.md` — setup and usage instructions for a human running these.
- `router_log.jsonl` — runtime log (git-ignored, created on first run, not in repo).

## Design intent
- Default behavior: try Claude (`claude-sonnet-4-6`), retry briefly on
  429/529, then fall back to LM Studio's local OpenAI-compatible endpoint
  (`http://localhost:1234/v1`) with the same prompt.
- `--force-local` / `--force-claude` bypass the fallback logic entirely.
- No API keys are ever read from files or CLI args — only from the
  `ANTHROPIC_API_KEY` environment variable. Do not add code that accepts
  a key as a parameter or writes one to disk.
- Local fallback calls are fully offline; only the Claude path sends data
  off-machine.

## Design intent — claude_code_proxy.py
- The proxy is the "clear second concern" that justifies a second file: it's a
  server, not a CLI, and exists so Claude Code itself runs on a local model.
- It is **fully offline** — there is no Claude/cloud path in this file at all.
  All requests go only to LM Studio. Don't add a cloud fallback here.
- Config is via environment variables only (LMSTUDIO_*, PROXY_*), never CLI
  args or files — consistent with the no-secrets-on-disk rule below.
- Claude Code is tool-driven, so the proxy must faithfully translate tools and
  tool calls both directions; the local model must support function calling.
- `PROXY_REASONING` controls how a reasoning model's chain-of-thought is mapped
  (drop / think / text); default `drop` keeps Claude Code output clean.

## Conventions when extending this
- `router.py` stays a single-file CLI; `claude_code_proxy.py` stays a single-file
  server. Add a third file only for another clearly distinct concern.
- Any new `router.py` CLI flag should have a sensible default matching current
  Claude-first behavior, so existing invocations don't silently change.
- Don't log full prompt text — `router.py` logs 120-char previews to
  `router_log.jsonl`; the proxy logs only request metadata (counts, model) under
  `PROXY_LOG=1`, never prompt content. This file may contain sensitive business
  content from the user's consulting work.
