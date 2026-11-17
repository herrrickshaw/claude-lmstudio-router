"""Tests for claude_code_proxy.py. LM Studio is replaced by an httpx.MockTransport;
nothing listens on or connects to a real port."""

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import claude_code_proxy as proxy  # noqa: E402


# ---- helpers ---------------------------------------------------------------

def _install_transport(monkeypatch, handler):
    real = httpx.AsyncClient

    def factory(*a, **kw):
        kw["transport"] = httpx.MockTransport(handler)
        return real(*a, **kw)

    monkeypatch.setattr(proxy.httpx, "AsyncClient", factory)


def _sse_body(chunks, done=True):
    lines = [f"data: {json.dumps(c)}\n\n" for c in chunks]
    if done:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def _collect(gen):
    async def run():
        return [e async for e in gen]
    return asyncio.run(run())


def _events(raw):
    out = []
    for e in raw:
        ev, data = e.strip().split("\n", 1)
        out.append((ev[len("event: "):], json.loads(data[len("data: "):])))
    return out


def _chunk(delta=None, finish=None, usage=None):
    c = {"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
    if usage is not None:
        c = {"choices": [], "usage": usage}
    return c


def _assert_blocks_balanced(events):
    started = [d["index"] for e, d in events if e == "content_block_start"]
    stopped = [d["index"] for e, d in events if e == "content_block_stop"]
    assert sorted(started) == sorted(stopped), (started, stopped)


# ---- request translation ---------------------------------------------------

def test_messages_translation_tool_round_trip():
    body = {
        "system": [{"type": "text", "text": "be brief"}],
        "messages": [
            {"role": "user", "content": "list files"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "hmm"},
                {"type": "text", "text": "ok"},
                {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"cmd": "ls"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1",
                 "content": [{"type": "text", "text": "a.txt"}]},
                {"type": "text", "text": "thanks"},
            ]},
        ],
    }
    msgs = proxy.anthropic_to_openai_messages(body)
    assert msgs[0] == {"role": "system", "content": "be brief"}
    assert msgs[1] == {"role": "user", "content": "list files"}
    assert msgs[2]["content"] == "ok"
    assert msgs[2]["tool_calls"][0]["function"] == {"name": "Bash", "arguments": '{"cmd": "ls"}'}
    assert msgs[3] == {"role": "tool", "tool_call_id": "t1", "content": "a.txt"}
    assert msgs[4] == {"role": "user", "content": "thanks"}


def test_assistant_tool_only_turn_has_null_content():
    body = {"messages": [{"role": "assistant", "content": [
        {"type": "tool_use", "id": "t", "name": "X", "input": {}}]}]}
    assert proxy.anthropic_to_openai_messages(body)[0]["content"] is None


def test_tools_and_tool_choice_mapping():
    body = {"model": "claude-x", "max_tokens": 10, "temperature": 0,
            "tools": [{"name": "Read", "description": "d", "input_schema": {"type": "object"}},
                      {"description": "no name -> skipped"}],
            "messages": []}
    for tc, expected in [({"type": "auto"}, "auto"), ({"type": "any"}, "required"),
                         ({"type": "tool", "name": "Read"},
                          {"type": "function", "function": {"name": "Read"}})]:
        p = proxy.build_openai_payload({**body, "tool_choice": tc}, stream=False)
        assert p["tool_choice"] == expected
    assert len(p["tools"]) == 1 and p["max_tokens"] == 10 and p["temperature"] == 0
    assert "stream_options" not in p
    assert proxy.build_openai_payload(body, stream=True)["stream_options"] == {"include_usage": True}


def test_pick_model_routes_haiku_to_small(monkeypatch):
    monkeypatch.setattr(proxy, "LMSTUDIO_MODEL", "big")
    monkeypatch.setattr(proxy, "LMSTUDIO_SMALL_MODEL", "small")
    assert proxy.pick_model({"model": "claude-haiku-4"}) == "small"
    assert proxy.pick_model({"model": "claude-sonnet"}) == "big"
    monkeypatch.setattr(proxy, "LMSTUDIO_MODEL", "")
    monkeypatch.setattr(proxy, "LMSTUDIO_SMALL_MODEL", "")
    assert proxy.pick_model({"model": "claude-sonnet"}) == "claude-sonnet"


@pytest.mark.parametrize("finish,stop", [("stop", "end_turn"), ("length", "max_tokens"),
                                          ("tool_calls", "tool_use"), (None, "end_turn"),
                                          ("weird", "end_turn")])
def test_stop_reason(finish, stop):
    assert proxy._stop_reason(finish) == stop


# ---- non-streaming response translation ------------------------------------

def test_openai_to_anthropic_response_text_and_tools(monkeypatch):
    monkeypatch.setattr(proxy, "PROXY_REASONING", "drop")
    data = {"id": "x", "choices": [{"finish_reason": "tool_calls", "message": {
        "content": "running", "reasoning_content": "secret",
        "tool_calls": [{"id": "c1", "function": {"name": "Bash", "arguments": '{"cmd":"ls"}'}},
                       {"id": "c2", "function": {"name": "Bad", "arguments": "{not json"}}]}}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3}}
    r = proxy.openai_to_anthropic_response(data, "m")
    assert [b["type"] for b in r["content"]] == ["text", "tool_use", "tool_use"]
    assert r["content"][1]["input"] == {"cmd": "ls"}
    assert r["content"][2]["input"] == {}
    assert r["stop_reason"] == "tool_use"
    assert r["usage"] == {"input_tokens": 7, "output_tokens": 3}


def test_reasoning_modes(monkeypatch):
    data = {"choices": [{"message": {"content": "ans", "reasoning_content": "why"}}]}
    monkeypatch.setattr(proxy, "PROXY_REASONING", "think")
    assert [b["type"] for b in proxy.openai_to_anthropic_response(data, "m")["content"]] == ["thinking", "text"]
    monkeypatch.setattr(proxy, "PROXY_REASONING", "text")
    assert proxy.openai_to_anthropic_response(data, "m")["content"] == [{"type": "text", "text": "why\n\nans"}]


def test_empty_response_yields_empty_text_block():
    r = proxy.openai_to_anthropic_response({"choices": []}, "m")
    assert r["content"] == [{"type": "text", "text": ""}]


# ---- streaming ---------------------------------------------------------------

def test_stream_text_with_trailing_usage_chunk(monkeypatch):
    """LM Studio honours stream_options.include_usage (the proxy asks for it) and sends
    a final usage-only chunk. That must not break the stream."""
    chunks = [_chunk({"content": "Hel"}), _chunk({"content": "lo"}, finish="stop"),
              _chunk(usage={"prompt_tokens": 11, "completion_tokens": 2})]
    _install_transport(monkeypatch, lambda req: httpx.Response(200, content=_sse_body(chunks)))
    ev = _events(_collect(proxy.stream_anthropic({"model": "m"}, "m")))
    text = "".join(d["delta"]["text"] for e, d in ev if e == "content_block_delta")
    assert text == "Hello"
    assert "proxy error" not in text
    _assert_blocks_balanced(ev)
    md = [d for e, d in ev if e == "message_delta"][0]
    assert md["delta"]["stop_reason"] == "end_turn"
    assert md["usage"]["output_tokens"] == 2


def test_stream_tool_call_with_usage_chunk_reports_tool_use(monkeypatch):
    chunks = [
        _chunk({"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "Bash", "arguments": ""}}]}),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"cmd":'}}]}),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"ls"}'}}]}, finish="tool_calls"),
        _chunk(usage={"prompt_tokens": 5, "completion_tokens": 9}),
    ]
    _install_transport(monkeypatch, lambda req: httpx.Response(200, content=_sse_body(chunks)))
    ev = _events(_collect(proxy.stream_anthropic({"model": "m"}, "m")))
    start = [d for e, d in ev if e == "content_block_start"][0]["content_block"]
    assert start["type"] == "tool_use" and start["name"] == "Bash" and start["id"] == "c1"
    parts = "".join(d["delta"]["partial_json"] for e, d in ev
                    if e == "content_block_delta" and d["delta"]["type"] == "input_json_delta")
    assert json.loads(parts) == {"cmd": "ls"}
    _assert_blocks_balanced(ev)
    assert [d for e, d in ev if e == "message_delta"][0]["delta"]["stop_reason"] == "tool_use"


def test_stream_text_then_tool_closes_text_block_first(monkeypatch):
    chunks = [_chunk({"content": "Let me look."}),
              _chunk({"tool_calls": [{"index": 0, "id": "c", "function": {"name": "Read", "arguments": "{}"}}]},
                     finish="tool_calls")]
    _install_transport(monkeypatch, lambda req: httpx.Response(200, content=_sse_body(chunks)))
    ev = _events(_collect(proxy.stream_anthropic({"model": "m"}, "m")))
    kinds = [e for e, _ in ev]
    assert kinds.index("content_block_stop") < [i for i, (e, d) in enumerate(ev)
                                                if e == "content_block_start"][1]
    _assert_blocks_balanced(ev)


def test_stream_backend_http_error_becomes_text_error(monkeypatch):
    _install_transport(monkeypatch, lambda req: httpx.Response(500, content=b"model not loaded"))
    ev = _events(_collect(proxy.stream_anthropic({"model": "m"}, "m")))
    text = "".join(d["delta"].get("text", "") for e, d in ev if e == "content_block_delta")
    assert "LM Studio HTTP 500" in text and "model not loaded" in text
    _assert_blocks_balanced(ev)
    assert ev[-1][0] == "message_stop"


class _BreakingStream(httpx.AsyncByteStream):
    def __init__(self, first):
        self.first = first

    async def __aiter__(self):
        yield self.first
        raise httpx.ReadError("connection dropped")


def test_stream_error_after_text_still_closes_open_block(monkeypatch):
    """If LM Studio drops the connection mid-answer, the Anthropic event stream must
    still be well-formed: every content_block_start needs its content_block_stop."""
    first = _sse_body([_chunk({"content": "partial"})], done=False)
    _install_transport(monkeypatch, lambda req: httpx.Response(200, stream=_BreakingStream(first)))
    ev = _events(_collect(proxy.stream_anthropic({"model": "m"}, "m")))
    _assert_blocks_balanced(ev)
    assert ev[-1][0] == "message_stop"


# ---- routes -------------------------------------------------------------------

def test_count_tokens_and_sync_route(monkeypatch):
    from fastapi.testclient import TestClient

    def handler(req):
        sent = json.loads(req.content)
        assert sent["stream"] is False
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop",
                                                      "message": {"content": "hi"}}]})

    _install_transport(monkeypatch, handler)
    c = TestClient(proxy.app)
    r = c.post("/v1/messages", json={"model": "m", "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 200 and r.json()["content"] == [{"type": "text", "text": "hi"}]
    r = c.post("/v1/messages/count_tokens", json={"system": "a" * 40, "messages": []})
    assert r.json() == {"input_tokens": 10}


def test_sync_route_backend_error_is_502(monkeypatch):
    from fastapi.testclient import TestClient
    _install_transport(monkeypatch, lambda req: httpx.Response(503, text="busy"))
    r = TestClient(proxy.app).post("/v1/messages", json={"model": "m", "messages": []})
    assert r.status_code == 502 and "503" in r.json()["error"]["message"]
