"""Tests for router.py. Anthropic and LM Studio are both mocked; no network."""

import io
import json
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import router  # noqa: E402


@pytest.fixture(autouse=True)
def isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(router, "LOG_FILE", str(tmp_path / "log.jsonl"))
    monkeypatch.setattr(router, "RETRY_BACKOFF_SECONDS", 0)
    monkeypatch.setattr(router.time, "sleep", lambda s: None)
    return tmp_path


def _log(tmp_path):
    p = tmp_path / "log.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


class FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_urlopen(payload=None, exc=None, seen=None):
    def urlopen(req, timeout=None):
        if seen is not None:
            seen.append((req.full_url, json.loads(req.data)))
        if exc:
            raise exc
        return FakeResp(json.dumps(payload).encode())
    return urlopen


# ---- call_local --------------------------------------------------------------

def test_call_local_builds_openai_request(monkeypatch):
    seen = []
    monkeypatch.setattr(router.urllib.request, "urlopen",
                        _fake_urlopen({"choices": [{"message": {"content": "ok"}}]}, seen=seen))
    assert router.call_local("hi", system="sys", base_url="http://h:1/v1/", model="m") == "ok"
    url, body = seen[0]
    assert url == "http://h:1/v1/chat/completions"
    assert body["model"] == "m"
    assert body["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]


def test_call_local_unreachable_is_runtime_error(monkeypatch):
    monkeypatch.setattr(router.urllib.request, "urlopen",
                        _fake_urlopen(exc=urllib.error.URLError("refused")))
    with pytest.raises(RuntimeError, match="Could not reach LM Studio"):
        router.call_local("hi")


def test_call_local_bad_shape_is_runtime_error(monkeypatch):
    monkeypatch.setattr(router.urllib.request, "urlopen", _fake_urlopen({"oops": 1}))
    with pytest.raises(RuntimeError, match="unexpected response shape"):
        router.call_local("hi")


def test_call_local_read_timeout_is_runtime_error(monkeypatch):
    """A slow local model that exceeds the 300 s timeout raises TimeoutError from
    getresponse()/read(), which urllib does NOT wrap in URLError. The router's
    fallback/error handling only catches RuntimeError, so this must be converted."""
    monkeypatch.setattr(router.urllib.request, "urlopen", _fake_urlopen(exc=TimeoutError("timed out")))
    with pytest.raises(RuntimeError):
        router.call_local("hi")


# ---- call_claude -------------------------------------------------------------

def test_call_claude_without_key_raises_runtime_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        router.call_claude("hi")


def test_call_claude_joins_text_blocks(monkeypatch):
    import anthropic
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    calls = []

    class FakeClient:
        def __init__(self, api_key):
            self.messages = SimpleNamespace(create=self.create)

        def create(self, **kw):
            calls.append(kw)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="a"),
                                            SimpleNamespace(type="tool_use"),
                                            SimpleNamespace(type="text", text="b")])

    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    assert router.call_claude("hi", system="s") == "a\nb"
    assert calls[0]["system"] == "s" and calls[0]["model"] == router.CLAUDE_MODEL


def test_call_claude_retries_connection_errors_then_raises(monkeypatch):
    import anthropic
    import httpx
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    n = []

    class FakeClient:
        def __init__(self, api_key):
            self.messages = SimpleNamespace(create=self.create)

        def create(self, **kw):
            n.append(1)
            raise anthropic.APIConnectionError(request=httpx.Request("POST", "http://x"))

    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    with pytest.raises(RuntimeError, match="failed after 2 attempts"):
        router.call_claude("hi")
    assert len(n) == router.MAX_RETRIES_CLAUDE


# ---- route -------------------------------------------------------------------

def test_route_prefers_claude(monkeypatch, isolate):
    monkeypatch.setattr(router, "call_claude", lambda p, s=None: "from claude")
    monkeypatch.setattr(router, "call_local", lambda *a: pytest.fail("local should not be called"))
    assert router.route("hi") == ("from claude", "claude")
    assert _log(isolate)[0]["engine"] == "claude"


def test_route_falls_back_to_local(monkeypatch, isolate):
    def boom(p, s=None):
        raise RuntimeError("429")
    monkeypatch.setattr(router, "call_claude", boom)
    monkeypatch.setattr(router, "call_local", lambda *a: "from local")
    assert router.route("hi") == ("from local", "local")
    entry = _log(isolate)[0]
    assert entry["engine"] == "local" and "429" in entry["note"]


def test_route_both_fail_logs_and_raises(monkeypatch, isolate):
    def boom(*a, **k):
        raise RuntimeError("down")
    monkeypatch.setattr(router, "call_claude", boom)
    monkeypatch.setattr(router, "call_local", boom)
    with pytest.raises(RuntimeError, match="Both Claude and local"):
        router.route("hi")
    assert _log(isolate)[0]["engine"] == "none"


def test_route_local_timeout_after_claude_failure_is_reported(monkeypatch, isolate):
    def boom(p, s=None):
        raise RuntimeError("no key")
    monkeypatch.setattr(router, "call_claude", boom)
    monkeypatch.setattr(router.urllib.request, "urlopen", _fake_urlopen(exc=TimeoutError("timed out")))
    with pytest.raises(RuntimeError, match="Both Claude and local"):
        router.route("hi")
    assert _log(isolate)[0]["engine"] == "none"


def test_force_local_skips_claude(monkeypatch):
    monkeypatch.setattr(router, "call_claude", lambda *a: pytest.fail("claude called"))
    monkeypatch.setattr(router, "call_local", lambda *a: "L")
    assert router.route("hi", force_local=True) == ("L", "local")


def test_prompt_preview_truncated_in_log(monkeypatch, isolate):
    monkeypatch.setattr(router, "call_claude", lambda p, s=None: "x")
    router.route("y" * 500)
    assert len(_log(isolate)[0]["prompt_preview"]) == 120
