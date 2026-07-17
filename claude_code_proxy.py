#!/usr/bin/env python3
"""
claude_code_proxy.py
--------------------
An Anthropic-API-compatible proxy that lets **Claude Code** (the agent CLI)
run against a **local model served by LM Studio**.

Claude Code talks the Anthropic Messages API. LM Studio talks the OpenAI
Chat Completions API. This proxy sits in the middle: it accepts Anthropic
`/v1/messages` requests from Claude Code, translates them to OpenAI chat
completions, forwards them to LM Studio, and translates the (streaming)
response back into Anthropic SSE events.

This is different from `router.py` in this repo:
  - router.py      = one-shot CLI: send a prompt to Claude, fall back to local.
  - this file      = a long-running server so the whole Claude Code agent
                     (tools, streaming, multi-turn) runs on your local model,
                     fully offline. Ideal for local RAG machines.

----------------------------------------------------------------------
QUICK START
----------------------------------------------------------------------
1. Install deps (once):
       pip install fastapi uvicorn httpx --break-system-packages
   (drop --break-system-packages inside a venv)

2. Start LM Studio, load a tool-capable model (e.g. Qwen2.5-Coder,
   Llama-3.1-8B-Instruct, etc.), and start its server
   (Developer tab > Start Server, default http://localhost:1234).
   IMPORTANT: pick a model that supports tool/function calling, or
   Claude Code's file-editing and shell tools won't work.

3. Start this proxy:
       python claude_code_proxy.py
   It listens on http://localhost:8787 by default.

4. In a NEW terminal, point Claude Code at the proxy and run it:
       export ANTHROPIC_BASE_URL="http://localhost:8787"
       export ANTHROPIC_API_KEY="local"      # any non-empty value
       claude

   (Or use the bundled helper: ./run_local_claude_code.sh)

Everything Claude Code does now runs on your local LM Studio model.
No data leaves the machine.

----------------------------------------------------------------------
CONFIG via environment variables (all optional)
----------------------------------------------------------------------
  LMSTUDIO_BASE_URL   default http://localhost:1234/v1
  LMSTUDIO_MODEL      default "" -> let LM Studio use its loaded model.
                      Set to the exact model id LM Studio reports if you
                      have several loaded.
  LMSTUDIO_SMALL_MODEL  optional: model id to use for Claude Code's small
                      "fast" calls (titles, summaries — the haiku-class
                      requests). Defaults to LMSTUDIO_MODEL.
  PROXY_HOST          default 127.0.0.1
  PROXY_PORT          default 8787
  PROXY_LOG           "1" to log each translated request to stderr.
  LMSTUDIO_TIMEOUT    per-request timeout seconds, default 600.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# ============================================================
# CONFIG
# ============================================================
LMSTUDIO_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1").rstrip("/")
LMSTUDIO_MODEL = os.environ.get("LMSTUDIO_MODEL", "").strip()
LMSTUDIO_SMALL_MODEL = os.environ.get("LMSTUDIO_SMALL_MODEL", "").strip() or LMSTUDIO_MODEL
PROXY_HOST = os.environ.get("PROXY_HOST", "127.0.0.1")
PROXY_PORT = int(os.environ.get("PROXY_PORT", "8787"))
PROXY_LOG = os.environ.get("PROXY_LOG", "") == "1"
LMSTUDIO_TIMEOUT = float(os.environ.get("LMSTUDIO_TIMEOUT", "600"))

# How to handle a reasoning model's chain-of-thought (LM Studio streams it in a
# separate "reasoning_content" field). Many local models (DeepSeek-R1, Qwen3,
# Gemma "thinking", Nemotron) are reasoning models.
#   "drop"  -> hide it; Claude Code only sees the final answer. (default, safest)
#   "think" -> surface it as Anthropic `thinking` content blocks.
#   "text"  -> inline it into the visible answer text.
PROXY_REASONING = os.environ.get("PROXY_REASONING", "drop").strip().lower()

app = FastAPI(title="claude-code-lmstudio-proxy")


def log(*a: Any) -> None:
    if PROXY_LOG:
        print("[proxy]", *a, file=sys.stderr, flush=True)


# ============================================================
# Anthropic -> OpenAI request translation
# ============================================================

def _text_from_blocks(blocks: Any) -> str:
    """Flatten an Anthropic content value (str or list of blocks) to plain text."""
    if isinstance(blocks, str):
        return blocks
    if not isinstance(blocks, list):
        return ""
    out: List[str] = []
    for b in blocks:
        if not isinstance(b, dict):
            out.append(str(b))
            continue
        t = b.get("type")
        if t == "text":
            out.append(b.get("text", ""))
        elif t == "image":
            out.append("[image omitted — local model has no vision in this proxy]")
        else:
            # tool_use / tool_result handled elsewhere; ignore here
            pass
    return "\n".join(p for p in out if p)


def _system_to_text(system: Any) -> str:
    if not system:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return _text_from_blocks(system)
    return str(system)


def anthropic_to_openai_messages(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert an Anthropic Messages request body into OpenAI chat messages."""
    messages: List[Dict[str, Any]] = []

    system_text = _system_to_text(body.get("system"))
    if system_text:
        messages.append({"role": "system", "content": system_text})

    for msg in body.get("messages", []):
        role = msg.get("role")
        content = msg.get("content")

        # Simple string content
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            continue

        if role == "user":
            # A user turn may contain text and/or tool_result blocks.
            text_parts: List[str] = []
            tool_results: List[Dict[str, Any]] = []
            for b in content:
                if not isinstance(b, dict):
                    continue
                bt = b.get("type")
                if bt == "text":
                    text_parts.append(b.get("text", ""))
                elif bt == "image":
                    text_parts.append("[image omitted]")
                elif bt == "tool_result":
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": b.get("tool_use_id", ""),
                        "content": _tool_result_to_text(b.get("content")),
                    })
            # OpenAI wants tool results as their own messages, before any
            # follow-up user text.
            messages.extend(tool_results)
            if text_parts:
                messages.append({"role": "user", "content": "\n".join(text_parts)})

        elif role == "assistant":
            text_parts2: List[str] = []
            tool_calls: List[Dict[str, Any]] = []
            for b in content:
                if not isinstance(b, dict):
                    continue
                bt = b.get("type")
                if bt == "text":
                    text_parts2.append(b.get("text", ""))
                elif bt == "thinking":
                    # drop thinking blocks; not all locals understand them
                    pass
                elif bt == "tool_use":
                    tool_calls.append({
                        "id": b.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": b.get("name", ""),
                            "arguments": json.dumps(b.get("input", {})),
                        },
                    })
            asst: Dict[str, Any] = {"role": "assistant"}
            asst["content"] = "\n".join(text_parts2) if text_parts2 else None
            if tool_calls:
                asst["tool_calls"] = tool_calls
            messages.append(asst)
        else:
            messages.append({"role": role, "content": _text_from_blocks(content)})

    return messages


def _tool_result_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _text_from_blocks(content)
    if content is None:
        return ""
    return json.dumps(content)


def anthropic_tools_to_openai(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return None
    out: List[Dict[str, Any]] = []
    for t in tools:
        if not isinstance(t, dict) or "name" not in t:
            continue
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return out or None


def pick_model(body: Dict[str, Any]) -> str:
    """Map the requested Anthropic model onto a local LM Studio model id."""
    requested = (body.get("model") or "").lower()
    if "haiku" in requested and LMSTUDIO_SMALL_MODEL:
        return LMSTUDIO_SMALL_MODEL
    if LMSTUDIO_MODEL:
        return LMSTUDIO_MODEL
    # Empty -> LM Studio uses whatever single model is loaded.
    return body.get("model", "local-model")


def build_openai_payload(body: Dict[str, Any], stream: bool) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": pick_model(body),
        "messages": anthropic_to_openai_messages(body),
        "stream": stream,
    }
    if body.get("max_tokens") is not None:
        payload["max_tokens"] = body["max_tokens"]
    if body.get("temperature") is not None:
        payload["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        payload["top_p"] = body["top_p"]
    tools = anthropic_tools_to_openai(body.get("tools"))
    if tools:
        payload["tools"] = tools
        # Pass through tool_choice when Claude forces a tool.
        tc = body.get("tool_choice")
        if isinstance(tc, dict):
            if tc.get("type") == "auto":
                payload["tool_choice"] = "auto"
            elif tc.get("type") == "any":
                payload["tool_choice"] = "required"
            elif tc.get("type") == "tool" and tc.get("name"):
                payload["tool_choice"] = {"type": "function", "function": {"name": tc["name"]}}
    if stream:
        payload["stream_options"] = {"include_usage": True}
    return payload


def _stop_reason(finish: Optional[str]) -> str:
    return {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
        "content_filter": "end_turn",
    }.get(finish or "stop", "end_turn")


def _approx_tokens(text: str) -> int:
    # Rough heuristic when the backend gives no usage numbers.
    return max(1, len(text) // 4)


# ============================================================
# OpenAI -> Anthropic response translation (non-streaming)
# ============================================================

def openai_to_anthropic_response(data: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message", {}) or {}
    blocks: List[Dict[str, Any]] = []

    reasoning = message.get("reasoning_content") or message.get("reasoning")
    text = message.get("content")

    if reasoning:
        if PROXY_REASONING == "think":
            blocks.append({"type": "thinking", "thinking": reasoning,
                           "signature": "local-proxy"})
        elif PROXY_REASONING == "text":
            text = (reasoning + "\n\n" + text) if text else reasoning

    if text:
        blocks.append({"type": "text", "text": text})

    for tc in message.get("tool_calls") or []:
        fn = tc.get("function", {}) or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (json.JSONDecodeError, TypeError):
            args = {}
        blocks.append({
            "type": "tool_use",
            "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
            "name": fn.get("name", ""),
            "input": args,
        })

    if not blocks:
        blocks.append({"type": "text", "text": ""})

    usage = data.get("usage") or {}
    return {
        "id": data.get("id") or f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model_name,
        "content": blocks,
        "stop_reason": _stop_reason(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


# ============================================================
# OpenAI SSE -> Anthropic SSE translation (streaming)
# ============================================================

def _sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def stream_anthropic(openai_payload: Dict[str, Any], model_name: str):
    """Async generator yielding Anthropic-format SSE bytes."""
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    # message_start
    yield _sse("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id, "type": "message", "role": "assistant",
            "model": model_name, "content": [],
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })
    yield _sse("ping", {"type": "ping"})

    # Block bookkeeping. Anthropic content blocks are emitted in stream order
    # with a running index. `cur` tracks the currently-open non-tool block
    # (a "thinking" or "text" block); tool_use blocks get their own indices.
    next_index = 0
    cur_kind: Optional[str] = None   # None | "thinking" | "text"
    cur_index: Optional[int] = None
    tool_slots: Dict[int, Dict[str, Any]] = {}   # OpenAI tool index -> our block
    finish_reason: Optional[str] = None
    completion_tokens = 0
    out_text_len = 0

    def close_cur() -> List[str]:
        """Emit the stop events for whatever non-tool block is currently open."""
        nonlocal cur_kind, cur_index
        evs: List[str] = []
        if cur_kind == "thinking":
            evs.append(_sse("content_block_delta", {
                "type": "content_block_delta", "index": cur_index,
                "delta": {"type": "signature_delta", "signature": "local-proxy"}}))
            evs.append(_sse("content_block_stop",
                            {"type": "content_block_stop", "index": cur_index}))
        elif cur_kind == "text":
            evs.append(_sse("content_block_stop",
                            {"type": "content_block_stop", "index": cur_index}))
        cur_kind, cur_index = None, None
        return evs

    url = f"{LMSTUDIO_BASE_URL}/chat/completions"

    try:
        async with httpx.AsyncClient(timeout=LMSTUDIO_TIMEOUT) as client:
            async with client.stream("POST", url, json=openai_payload,
                                     headers={"Content-Type": "application/json"}) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    raise RuntimeError(f"LM Studio HTTP {resp.status_code}: {body[:500]}")

                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[len("data:"):].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    if chunk.get("usage"):
                        u = chunk["usage"]
                        prompt_tokens = u.get("prompt_tokens", prompt_tokens)
                        completion_tokens = u.get("completion_tokens", completion_tokens)

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta", {}) or {}
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]

                    # ---- reasoning (chain-of-thought) delta ----
                    dreason = delta.get("reasoning_content") or delta.get("reasoning")
                    dtext = delta.get("content")
                    if dreason and PROXY_REASONING == "text":
                        # fold reasoning into the visible answer
                        dtext = dreason + (dtext or "")
                        dreason = None
                    if dreason and PROXY_REASONING == "think":
                        if cur_kind != "thinking":
                            for e in close_cur():
                                yield e
                            cur_index = next_index
                            next_index += 1
                            cur_kind = "thinking"
                            yield _sse("content_block_start", {
                                "type": "content_block_start", "index": cur_index,
                                "content_block": {"type": "thinking", "thinking": ""}})
                        yield _sse("content_block_delta", {
                            "type": "content_block_delta", "index": cur_index,
                            "delta": {"type": "thinking_delta", "thinking": dreason}})
                    # ("drop" mode simply ignores dreason)

                    # ---- text delta ----
                    if dtext:
                        if cur_kind != "text":
                            for e in close_cur():
                                yield e
                            cur_index = next_index
                            next_index += 1
                            cur_kind = "text"
                            yield _sse("content_block_start", {
                                "type": "content_block_start", "index": cur_index,
                                "content_block": {"type": "text", "text": ""}})
                        out_text_len += len(dtext)
                        yield _sse("content_block_delta", {
                            "type": "content_block_delta", "index": cur_index,
                            "delta": {"type": "text_delta", "text": dtext},
                        })

                    # ---- tool_call deltas ----
                    for tcd in delta.get("tool_calls") or []:
                        oi = tcd.get("index", 0)
                        if oi not in tool_slots:
                            # close any open thinking/text block before a tool block
                            for e in close_cur():
                                yield e
                            my_index = next_index
                            next_index += 1
                            fn = tcd.get("function", {}) or {}
                            tool_slots[oi] = {
                                "index": my_index,
                                "id": tcd.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                                "name": fn.get("name", ""),
                                "started": False,
                            }
                        slot = tool_slots[oi]
                        fn = tcd.get("function", {}) or {}
                        if fn.get("name") and not slot["name"]:
                            slot["name"] = fn["name"]
                        if not slot["started"]:
                            yield _sse("content_block_start", {
                                "type": "content_block_start", "index": slot["index"],
                                "content_block": {
                                    "type": "tool_use", "id": slot["id"],
                                    "name": slot["name"], "input": {},
                                },
                            })
                            slot["started"] = True
                        args_piece = fn.get("arguments")
                        if args_piece:
                            yield _sse("content_block_delta", {
                                "type": "content_block_delta", "index": slot["index"],
                                "delta": {"type": "input_json_delta", "partial_json": args_piece},
                            })

        # close any open blocks
        for e in close_cur():
            yield e
        for slot in tool_slots.values():
            if slot["started"]:
                yield _sse("content_block_stop", {"type": "content_block_stop", "index": slot["index"]})

        stop = _stop_reason(finish_reason if finish_reason else ("tool_calls" if tool_slots else "stop"))
        if not completion_tokens:
            completion_tokens = _approx_tokens("x" * out_text_len)
        yield _sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop, "stop_sequence": None},
            "usage": {"output_tokens": completion_tokens},
        })
        yield _sse("message_stop", {"type": "message_stop"})

    except Exception as e:  # surface backend errors as an Anthropic error event
        log("stream error:", repr(e))
        if cur_kind is None and not tool_slots:
            yield _sse("content_block_start", {
                "type": "content_block_start", "index": 0,
                "content_block": {"type": "text", "text": ""}})
            yield _sse("content_block_delta", {
                "type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta",
                          "text": f"\n[proxy error contacting LM Studio: {e}]"}})
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
        yield _sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 0}})
        yield _sse("message_stop", {"type": "message_stop"})


# ============================================================
# Routes
# ============================================================

@app.post("/v1/messages")
async def messages(request: Request):
    body = await request.json()
    stream = bool(body.get("stream"))
    model_name = body.get("model", "claude-local")
    payload = build_openai_payload(body, stream)
    log(f"{'stream' if stream else 'sync'} -> model={payload['model']} "
        f"msgs={len(payload['messages'])} tools={len(payload.get('tools', []))}")

    if stream:
        return StreamingResponse(
            stream_anthropic(payload, model_name),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    url = f"{LMSTUDIO_BASE_URL}/chat/completions"
    try:
        async with httpx.AsyncClient(timeout=LMSTUDIO_TIMEOUT) as client:
            r = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPStatusError as e:
        return JSONResponse(status_code=502, content={
            "type": "error",
            "error": {"type": "api_error",
                      "message": f"LM Studio HTTP {e.response.status_code}: {e.response.text[:500]}"}})
    except Exception as e:
        return JSONResponse(status_code=502, content={
            "type": "error",
            "error": {"type": "api_error", "message": f"Could not reach LM Studio at {url}: {e}"}})

    return JSONResponse(content=openai_to_anthropic_response(data, model_name))


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    """Claude Code calls this to size context. We give a cheap estimate."""
    body = await request.json()
    text = _system_to_text(body.get("system"))
    for m in body.get("messages", []):
        text += "\n" + _text_from_blocks(m.get("content"))
    return JSONResponse(content={"input_tokens": _approx_tokens(text)})


@app.get("/health")
async def health():
    """Report whether LM Studio is reachable behind the proxy."""
    url = f"{LMSTUDIO_BASE_URL}/models"
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(url)
            models = [m.get("id") for m in (r.json().get("data") or [])]
        return {"proxy": "ok", "lmstudio": "ok", "lmstudio_url": LMSTUDIO_BASE_URL,
                "models_loaded": models}
    except Exception as e:
        return JSONResponse(status_code=503, content={
            "proxy": "ok", "lmstudio": "unreachable",
            "lmstudio_url": LMSTUDIO_BASE_URL, "error": str(e)})


def main():
    try:
        import uvicorn
    except ImportError:
        print("Missing deps. Run: pip install fastapi uvicorn httpx --break-system-packages",
              file=sys.stderr)
        sys.exit(1)

    print(f"claude-code -> LM Studio proxy", file=sys.stderr)
    print(f"  listening on  http://{PROXY_HOST}:{PROXY_PORT}", file=sys.stderr)
    print(f"  forwarding to {LMSTUDIO_BASE_URL}", file=sys.stderr)
    print(f"  local model   {LMSTUDIO_MODEL or '(LM Studio default)'}", file=sys.stderr)
    print(f"\nPoint Claude Code at it:", file=sys.stderr)
    print(f'  export ANTHROPIC_BASE_URL="http://{PROXY_HOST}:{PROXY_PORT}"', file=sys.stderr)
    print(f'  export ANTHROPIC_API_KEY="local"', file=sys.stderr)
    print(f"  claude\n", file=sys.stderr)
    uvicorn.run(app, host=PROXY_HOST, port=PROXY_PORT, log_level="warning")


if __name__ == "__main__":
    main()
