"""claude_code_proxy.py: message/tool translation branches, payload building, streaming modes, routes, main()."""
import asyncio
import json
import sys
import types

import httpx
import pytest
from fastapi.testclient import TestClient

import claude_code_proxy as proxy


def _install(monkeypatch, handler):
    real = httpx.AsyncClient
    monkeypatch.setattr(proxy.httpx, "AsyncClient",
                        lambda *a, **kw: real(*a, **{**kw, "transport": httpx.MockTransport(handler)}))


def _sse_body(chunks, done=True):
    s = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    return (s + ("data: [DONE]\n\n" if done else "")).encode()


def _stream(monkeypatch, chunks, raw=None, done=True):
    body = raw if raw is not None else _sse_body(chunks, done)
    _install(monkeypatch, lambda req: httpx.Response(200, content=body))

    async def run():
        return [e async for e in proxy.stream_anthropic({"model": "m"}, "m")]
    out = []
    for e in asyncio.run(run()):
        ev, data = e.strip().split("\n", 1)
        out.append((ev[7:], json.loads(data[6:])))
    return out


def _c(delta, finish=None):
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


# ---------------------------- request translation ----------------------------

def test_text_from_blocks_variants():
    assert proxy._text_from_blocks("s") == "s"
    assert proxy._text_from_blocks(5) == ""
    assert proxy._text_from_blocks(None) == ""
    blocks = [{"type": "text", "text": "a"}, "raw", {"type": "image"}, {"type": "tool_use"}, {"type": "text", "text": ""},
              {"type": "text", "text": "b"}]
    assert proxy._text_from_blocks(blocks) == "a\nraw\n[image omitted — local model has no vision in this proxy]\nb"


@pytest.mark.parametrize("system,want", [(None, ""), ("", ""), ("s", "s"), (["x"], "x"),
                                         ([{"type": "text", "text": "p"}, {"type": "text", "text": "q"}], "p\nq"),
                                         (7, "7")])
def test_system_to_text(system, want):
    assert proxy._system_to_text(system) == want


def test_tool_result_to_text_variants():
    assert proxy._tool_result_to_text("s") == "s"
    assert proxy._tool_result_to_text([{"type": "text", "text": "t"}]) == "t"
    assert proxy._tool_result_to_text(None) == ""
    assert proxy._tool_result_to_text({"a": 1}) == '{"a": 1}'


def test_messages_system_blocks_and_odd_content():
    body = {"system": [{"type": "text", "text": "SYS"}], "messages": [
        {"role": "user", "content": 42},                                         # not str/list: skipped
        {"role": "user", "content": ["junk", {"type": "image"}, {"type": "text", "text": "hi"}]},
        {"role": "system", "content": [{"type": "text", "text": "mid"}]},        # other role -> flattened
    ]}
    assert proxy.anthropic_to_openai_messages(body) == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "[image omitted]\nhi"},
        {"role": "system", "content": "mid"},
    ]


def test_user_tool_results_precede_followup_text_and_defaults():
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "after"},
        {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "out"}]},
        {"type": "tool_result", "content": None}]}]}
    assert proxy.anthropic_to_openai_messages(body) == [
        {"role": "tool", "tool_call_id": "t1", "content": "out"},
        {"role": "tool", "tool_call_id": "", "content": ""},
        {"role": "user", "content": "after"}]


def test_assistant_text_thinking_and_tool_use_blocks():
    body = {"messages": [{"role": "assistant", "content": [
        "junk", {"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "a"}, {"type": "text", "text": "b"},
        {"type": "tool_use", "id": "i", "name": "Bash", "input": {"c": 1}}, {"type": "tool_use"}]}]}
    (m,) = proxy.anthropic_to_openai_messages(body)
    assert m["content"] == "a\nb"
    assert m["tool_calls"][0] == {"id": "i", "type": "function", "function": {"name": "Bash", "arguments": '{"c": 1}'}}
    assert m["tool_calls"][1]["function"] == {"name": "", "arguments": "{}"}
    (n,) = proxy.anthropic_to_openai_messages({"messages": [{"role": "assistant", "content": [{"type": "text", "text": "x"}]}]})
    assert n == {"role": "assistant", "content": "x"}                      # no tool_calls key


def test_tools_conversion_skips_invalid_and_defaults_schema():
    assert proxy.anthropic_tools_to_openai(None) is None
    assert proxy.anthropic_tools_to_openai([]) is None
    assert proxy.anthropic_tools_to_openai(["x", {"no": "name"}]) is None
    (t,) = proxy.anthropic_tools_to_openai([{"name": "n"}, 5])
    assert t == {"type": "function", "function": {"name": "n", "description": "",
                                                  "parameters": {"type": "object", "properties": {}}}}


def test_payload_optional_fields_and_stream_options():
    p = proxy.build_openai_payload({"model": "claude-x", "messages": [], "max_tokens": 0, "temperature": 0, "top_p": 0.5}, True)
    assert (p["max_tokens"], p["temperature"], p["top_p"]) == (0, 0, 0.5)          # zero values are kept
    assert p["stream"] is True and p["stream_options"] == {"include_usage": True}
    q = proxy.build_openai_payload({"model": "claude-x", "messages": []}, False)
    assert not ({"max_tokens", "temperature", "top_p", "tools", "stream_options"} & set(q)) and q["stream"] is False


@pytest.mark.parametrize("tc,want", [({"type": "auto"}, "auto"), ({"type": "any"}, "required"),
                                     ({"type": "tool", "name": "Bash"}, {"type": "function", "function": {"name": "Bash"}}),
                                     ({"type": "tool"}, None), ({"type": "none"}, None), ("auto", None)])
def test_tool_choice_mapping(tc, want):
    p = proxy.build_openai_payload({"model": "m", "messages": [], "tools": [{"name": "Bash"}], "tool_choice": tc}, False)
    assert p.get("tool_choice") == want


def test_tool_choice_ignored_without_tools():
    assert "tool_choice" not in proxy.build_openai_payload({"model": "m", "messages": [], "tool_choice": {"type": "any"}}, False)


def test_pick_model_precedence(monkeypatch):
    monkeypatch.setattr(proxy, "LMSTUDIO_MODEL", "")
    monkeypatch.setattr(proxy, "LMSTUDIO_SMALL_MODEL", "")
    assert proxy.pick_model({"model": "claude-sonnet"}) == "claude-sonnet"
    assert proxy.pick_model({}) == "local-model"
    monkeypatch.setattr(proxy, "LMSTUDIO_MODEL", "big")
    assert proxy.pick_model({"model": "claude-HAIKU-4"}) == "big"                  # no small model configured
    monkeypatch.setattr(proxy, "LMSTUDIO_SMALL_MODEL", "small")
    assert proxy.pick_model({"model": "claude-HAIKU-4"}) == "small" and proxy.pick_model({"model": "opus"}) == "big"


def test_approx_tokens_floor():
    assert proxy._approx_tokens("") == 1 and proxy._approx_tokens("abcdefgh") == 2


# ---------------------------- non-streaming response ----------------------------

def test_response_invalid_tool_arguments_and_generated_ids(monkeypatch):
    monkeypatch.setattr(proxy, "PROXY_REASONING", "drop")
    data = {"choices": [{"message": {"content": None, "reasoning_content": "secret", "tool_calls": [
        {"function": {"name": "A", "arguments": "{bad"}}, {"id": "k", "function": {"name": "B", "arguments": None}},
        {"id": "z", "function": {"name": "C", "arguments": '{"x":1}'}}]}, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4}}
    r = proxy.openai_to_anthropic_response(data, "mm")
    a, b, c = r["content"]
    assert a["input"] == {} and a["id"].startswith("toolu_") and len(a["id"]) == 30
    assert b["id"] == "k" and b["input"] == {} and c["input"] == {"x": 1}
    assert r["id"].startswith("msg_") and r["model"] == "mm" and r["stop_reason"] == "tool_use"
    assert r["usage"] == {"input_tokens": 3, "output_tokens": 4} and all(x["type"] == "tool_use" for x in r["content"])


def test_response_reasoning_modes_with_and_without_text(monkeypatch):
    d = {"choices": [{"message": {"content": "ans", "reasoning": "why"}}]}
    monkeypatch.setattr(proxy, "PROXY_REASONING", "text")
    assert proxy.openai_to_anthropic_response(d, "m")["content"] == [{"type": "text", "text": "why\n\nans"}]
    d2 = {"choices": [{"message": {"content": "", "reasoning_content": "why"}}]}
    assert proxy.openai_to_anthropic_response(d2, "m")["content"] == [{"type": "text", "text": "why"}]
    monkeypatch.setattr(proxy, "PROXY_REASONING", "think")
    b = proxy.openai_to_anthropic_response(d, "m")["content"]
    assert b[0] == {"type": "thinking", "thinking": "why", "signature": "local-proxy"} and b[1]["text"] == "ans"
    assert proxy.openai_to_anthropic_response({"id": "X", "choices": []}, "m")["id"] == "X"
    assert proxy.openai_to_anthropic_response({}, "m")["content"] == [{"type": "text", "text": ""}]


# ---------------------------- streaming ----------------------------

def test_stream_think_mode_emits_thinking_then_text(monkeypatch):
    monkeypatch.setattr(proxy, "PROXY_REASONING", "think")
    ev = _stream(monkeypatch, [_c({"reasoning_content": "a"}), _c({"reasoning": "b"}), _c({"content": "T"}, "stop")])
    kinds = [(e, d.get("index"), d.get("delta", {}).get("type") or d.get("content_block", {}).get("type")) for e, d in ev]
    assert kinds[2:] == [("content_block_start", 0, "thinking"), ("content_block_delta", 0, "thinking_delta"),
                         ("content_block_delta", 0, "thinking_delta"), ("content_block_delta", 0, "signature_delta"),
                         ("content_block_stop", 0, None), ("content_block_start", 1, "text"),
                         ("content_block_delta", 1, "text_delta"), ("content_block_stop", 1, None),
                         ("message_delta", None, None), ("message_stop", None, None)]
    assert [d["delta"]["thinking"] for e, d in ev if e == "content_block_delta" and d["delta"]["type"] == "thinking_delta"] == ["a", "b"]


def test_stream_text_mode_inlines_reasoning_and_drop_mode_hides_it(monkeypatch):
    monkeypatch.setattr(proxy, "PROXY_REASONING", "text")
    ev = _stream(monkeypatch, [_c({"reasoning_content": "R", "content": "X"}), _c({"reasoning_content": "S"}, "stop")])
    assert [d["delta"]["text"] for e, d in ev if e == "content_block_delta"] == ["RX", "S"]
    monkeypatch.setattr(proxy, "PROXY_REASONING", "drop")
    ev = _stream(monkeypatch, [_c({"reasoning_content": "R"}), _c({"content": "X"}, "stop")])
    assert [d["delta"]["text"] for e, d in ev if e == "content_block_delta"] == ["X"]
    assert [d["index"] for e, d in ev if e == "content_block_start"] == [0]


def test_stream_skips_noise_lines_and_stops_at_done(monkeypatch):
    raw = (b": comment\n\nevent: x\n\ndata: {not json}\n\n" + b"data: " + json.dumps({"choices": []}).encode() + b"\n\n"
           + b"data: " + json.dumps(_c({"content": "ok"})).encode() + b"\n\ndata: [DONE]\n\n"
           + b"data: " + json.dumps(_c({"content": "AFTER"})).encode() + b"\n\n")
    ev = _stream(monkeypatch, None, raw=raw)
    assert [d["delta"]["text"] for e, d in ev if e == "content_block_delta"] == ["ok"]


def test_stream_without_usage_estimates_tokens_and_default_stop(monkeypatch):
    ev = _stream(monkeypatch, [_c({"content": "x" * 40})], done=False)
    md = [d for e, d in ev if e == "message_delta"][0]
    assert md["usage"] == {"output_tokens": 10} and md["delta"]["stop_reason"] == "end_turn"
    assert ev[0][0] == "message_start" and ev[1][0] == "ping" and ev[-1][0] == "message_stop"


def test_stream_tool_call_without_finish_reason_reports_tool_use_and_late_name(monkeypatch):
    ev = _stream(monkeypatch, [_c({"tool_calls": [{"function": {"arguments": ""}}]}),
                               _c({"tool_calls": [{"function": {"name": "Late", "arguments": "{}"}}]})])
    starts = [d for e, d in ev if e == "content_block_start"]
    assert len(starts) == 1 and starts[0]["content_block"]["id"].startswith("toolu_")
    md = [d for e, d in ev if e == "message_delta"][0]
    assert md["delta"]["stop_reason"] == "tool_use"


def test_stream_parallel_tool_calls_get_separate_indices_and_stops(monkeypatch):
    ev = _stream(monkeypatch, [_c({"tool_calls": [{"index": 0, "id": "a", "function": {"name": "A", "arguments": "1"}},
                                                  {"index": 1, "id": "b", "function": {"name": "B", "arguments": "2"}}]}, "tool_calls")])
    assert [d["index"] for e, d in ev if e == "content_block_start"] == [0, 1]
    assert sorted(d["index"] for e, d in ev if e == "content_block_stop") == [0, 1]


def test_stream_http_error_without_open_blocks_yields_error_text(monkeypatch):
    _install(monkeypatch, lambda req: httpx.Response(500, text="boom"))

    async def run():
        return [e async for e in proxy.stream_anthropic({}, "m")]
    out = "".join(asyncio.run(run()))
    assert "proxy error contacting LM Studio: LM Studio HTTP 500: boom" in out and out.count("event: content_block_stop") == 1
    assert out.rstrip().endswith('{"type": "message_stop"}')


# ---------------------------- routes ----------------------------

def test_messages_route_streams_with_sse_headers(monkeypatch):
    _install(monkeypatch, lambda req: httpx.Response(200, content=_sse_body([_c({"content": "hey"}, "stop")])))
    r = TestClient(proxy.app).post("/v1/messages", json={"model": "m", "stream": True, "messages": []})
    assert r.headers["content-type"].startswith("text/event-stream") and r.headers["cache-control"] == "no-cache"
    assert "event: message_start" in r.text and "hey" in r.text


def test_sync_route_unreachable_backend_is_502(monkeypatch):
    def h(req):
        raise httpx.ConnectError("refused")
    _install(monkeypatch, h)
    r = TestClient(proxy.app).post("/v1/messages", json={"model": "m", "messages": []})
    assert r.status_code == 502 and "Could not reach LM Studio at" in r.json()["error"]["message"]


def test_count_tokens_includes_messages(monkeypatch):
    r = TestClient(proxy.app).post("/v1/messages/count_tokens", json={
        "system": "", "messages": [{"role": "user", "content": "a" * 36}, {"role": "user", "content": [{"type": "text", "text": "b" * 40}]}]})
    assert r.json() == {"input_tokens": (1 + 36 + 1 + 40) // 4}


def test_health_ok_and_unreachable(monkeypatch):
    state = {"down": False}

    def h(req):
        if state["down"]:
            raise httpx.ConnectError("nope")
        return httpx.Response(200, json={"data": [{"id": "m1"}, {"id": "m2"}]})
    _install(monkeypatch, h)
    r = TestClient(proxy.app).get("/health")
    assert r.status_code == 200 and r.json()["models_loaded"] == ["m1", "m2"] and r.json()["lmstudio"] == "ok"
    state["down"] = True
    r = TestClient(proxy.app).get("/health")
    assert r.status_code == 503 and r.json()["lmstudio"] == "unreachable" and r.json()["proxy"] == "ok"


def test_health_with_no_data_key(monkeypatch):
    _install(monkeypatch, lambda req: httpx.Response(200, json={}))
    assert TestClient(proxy.app).get("/health").json()["models_loaded"] == []


def test_log_only_when_enabled(monkeypatch, capsys):
    monkeypatch.setattr(proxy, "PROXY_LOG", False)
    proxy.log("a")
    assert capsys.readouterr().err == ""
    monkeypatch.setattr(proxy, "PROXY_LOG", True)
    proxy.log("a", 1)
    assert capsys.readouterr().err == "[proxy] a 1\n"


# ---------------------------- main ----------------------------

def test_main_runs_uvicorn_with_config(monkeypatch, capsys):
    seen = {}
    fake = types.ModuleType("uvicorn")
    fake.run = lambda app, **kw: seen.update(app=app, **kw)
    monkeypatch.setitem(sys.modules, "uvicorn", fake)
    monkeypatch.setattr(proxy, "PROXY_HOST", "1.2.3.4")
    monkeypatch.setattr(proxy, "PROXY_PORT", 9999)
    monkeypatch.setattr(proxy, "LMSTUDIO_MODEL", "")
    proxy.main()
    assert seen == {"app": proxy.app, "host": "1.2.3.4", "port": 9999, "log_level": "warning"}
    err = capsys.readouterr().err
    assert "http://1.2.3.4:9999" in err and "(LM Studio default)" in err and 'ANTHROPIC_BASE_URL="http://1.2.3.4:9999"' in err


def test_main_without_uvicorn_exits_1(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    with pytest.raises(SystemExit) as e:
        proxy.main()
    assert e.value.code == 1 and "Missing deps" in capsys.readouterr().err
