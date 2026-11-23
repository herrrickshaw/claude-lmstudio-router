"""router.py: Claude retry/back-off paths (fake anthropic module), logging, force flags and the CLI (main)."""
import json
import sys
import types
from types import SimpleNamespace

import pytest

import router


@pytest.fixture(autouse=True)
def iso(monkeypatch, tmp_path):
    monkeypatch.setattr(router, "LOG_FILE", str(tmp_path / "log.jsonl"))
    sleeps = []
    monkeypatch.setattr(router.time, "sleep", sleeps.append)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    return sleeps


def _fake_anthropic(monkeypatch, script):
    """script: list of results/exceptions returned by successive messages.create() calls."""
    mod = types.ModuleType("anthropic")

    class RateLimitError(Exception):
        pass

    class APIStatusError(Exception):
        def __init__(self, code=529):
            super().__init__(code)
            self.status_code = code

    class APIConnectionError(Exception):
        pass
    calls = []

    class Anthropic:
        def __init__(self, api_key):
            calls.append(("key", api_key))
            self.messages = SimpleNamespace(create=self.create)

        def create(self, **kw):
            calls.append(kw)
            item = script.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
    mod.RateLimitError, mod.APIStatusError, mod.APIConnectionError, mod.Anthropic = (
        RateLimitError, APIStatusError, APIConnectionError, Anthropic)
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    return mod, calls


def _ok(text="hi"):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


def test_log_call_swallows_oserror(monkeypatch, tmp_path):
    monkeypatch.setattr(router, "LOG_FILE", str(tmp_path / "no_such_dir" / "x.jsonl"))
    router.log_call("claude", "p", True)          # must not raise


def test_log_call_entry_shape(tmp_path):
    router.log_call("local", "x" * 300, False, "why")
    e = json.loads((tmp_path / "log.jsonl").read_text())
    assert (e["engine"], e["success"], e["note"]) == ("local", False, "why")
    assert len(e["prompt_preview"]) == 120 and len(e["timestamp"]) == 19


def test_call_claude_missing_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)       # import raises ImportError
    with pytest.raises(RuntimeError, match="isn't installed"):
        router.call_claude("hi")


def test_call_claude_sends_model_tokens_and_optional_system(monkeypatch):
    _, calls = _fake_anthropic(monkeypatch, [_ok("a"), _ok("b")])
    assert router.call_claude("q") == "a"
    assert router.call_claude("q", system="sys") == "b"
    assert calls[0] == ("key", "k")
    first, second = calls[1], calls[3]
    assert first == {"model": router.CLAUDE_MODEL, "max_tokens": router.CLAUDE_MAX_TOKENS,
                     "messages": [{"role": "user", "content": "q"}]}
    assert second["system"] == "sys"


def test_rate_limit_backoff_grows_with_attempt_then_succeeds(monkeypatch, iso):
    script = []
    mod, _ = _fake_anthropic(monkeypatch, script)
    script += [mod.RateLimitError(), _ok("done")]
    monkeypatch.setattr(router, "RETRY_BACKOFF_SECONDS", 3)
    assert router.call_claude("q") == "done"
    assert iso == [3]                                          # BACKOFF * attempt(1)


def test_rate_limit_every_attempt_exhausts_with_scaled_sleeps(monkeypatch, iso, capsys):
    script = []
    mod, _ = _fake_anthropic(monkeypatch, script)
    script += [mod.RateLimitError("rl"), mod.RateLimitError("rl")]
    monkeypatch.setattr(router, "RETRY_BACKOFF_SECONDS", 5)
    with pytest.raises(RuntimeError, match=r"Claude API failed after 2 attempts: rl"):
        router.call_claude("q")
    assert iso == [5, 10]
    assert "Rate limited (attempt 2/2). Backing off 10s" in capsys.readouterr().err


def test_api_status_error_retries_with_flat_backoff(monkeypatch, iso, capsys):
    script = []
    mod, _ = _fake_anthropic(monkeypatch, script)
    script += [mod.APIStatusError(529), _ok("ok")]
    monkeypatch.setattr(router, "RETRY_BACKOFF_SECONDS", 4)
    assert router.call_claude("q") == "ok"
    assert iso == [4]
    assert "API error (529). Attempt 1/2." in capsys.readouterr().err


def test_connection_error_retries_flat(monkeypatch, iso):
    script = []
    mod, _ = _fake_anthropic(monkeypatch, script)
    script += [mod.APIConnectionError("down"), mod.APIConnectionError("down")]
    monkeypatch.setattr(router, "RETRY_BACKOFF_SECONDS", 2)
    with pytest.raises(RuntimeError, match="down"):
        router.call_claude("q")
    assert iso == [2, 2]


def test_force_claude_logs_and_never_touches_local(monkeypatch, tmp_path):
    monkeypatch.setattr(router, "call_claude", lambda p, s=None: "C")
    monkeypatch.setattr(router, "call_local", lambda *a, **k: pytest.fail("local must not run"))
    assert router.route("p", force_claude=True) == ("C", "claude")
    e = json.loads((tmp_path / "log.jsonl").read_text())
    assert (e["engine"], e["note"]) == ("claude", "forced")


def test_force_claude_propagates_failure(monkeypatch):
    def boom(p, s=None):
        raise RuntimeError("nope")
    monkeypatch.setattr(router, "call_claude", boom)
    with pytest.raises(RuntimeError, match="nope"):
        router.route("p", force_claude=True)


def test_force_local_passes_overrides(monkeypatch):
    seen = {}
    monkeypatch.setattr(router, "call_local", lambda p, s, b, m: seen.update(a=(p, s, b, m)) or "L")
    assert router.route("p", system="s", force_local=True, local_base_url="u", local_model="m") == ("L", "local")
    assert seen["a"] == ("p", "s", "u", "m")


def test_call_local_defaults_and_strips_trailing_slash(monkeypatch):
    seen = {}

    class R:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "x"}}]}).encode()

    def fake(req, timeout):
        seen.update(url=req.full_url, body=json.loads(req.data), timeout=timeout, method=req.get_method())
        return R()
    monkeypatch.setattr(router.urllib.request, "urlopen", fake)
    assert router.call_local("p", system="s", base_url="http://h:1/v1/", model="m") == "x"
    assert seen["url"] == "http://h:1/v1/chat/completions" and seen["timeout"] == 300 and seen["method"] == "POST"
    assert seen["body"] == {"model": "m", "temperature": 0.7,
                            "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "p"}]}
    router.call_local("p")
    assert seen["url"] == router.LOCAL_BASE_URL + "/chat/completions"
    assert seen["body"]["model"] == router.LOCAL_MODEL and len(seen["body"]["messages"]) == 1


# ---------------------------------- CLI ----------------------------------

def _cli(monkeypatch, argv, route=None):
    monkeypatch.setattr(sys, "argv", ["router.py", *argv])
    if route:
        monkeypatch.setattr(router, "route", route)
    return router.main


def test_cli_requires_prompt_or_task(monkeypatch, capsys):
    with pytest.raises(SystemExit) as e:
        _cli(monkeypatch, [])()
    assert e.value.code == 2 and "Provide either --prompt or --task" in capsys.readouterr().err


def test_cli_prompt_goes_straight_through_and_prints(monkeypatch, capsys):
    seen = {}

    def fake_route(prompt, **kw):
        seen.update(prompt=prompt, **kw)
        return "ANSWER", "claude"
    _cli(monkeypatch, ["--prompt", "P", "--task", "ignored", "--system", "S", "--force-local",
                       "--local-base-url", "U", "--local-model", "M"], fake_route)()
    out = capsys.readouterr()
    assert seen == {"prompt": "P", "system": "S", "force_local": True, "force_claude": False,
                    "local_base_url": "U", "local_model": "M"}
    assert out.out == "ANSWER\n" and "--- Answered by: CLAUDE ---" in out.err


def test_cli_task_with_input_file_combines_with_separator(monkeypatch, tmp_path, capsys):
    f = tmp_path / "in.txt"
    f.write_text("DATA", encoding="utf-8")
    seen = {}
    _cli(monkeypatch, ["--task", "T", "--input", str(f)],
         lambda prompt, **kw: seen.update(p=prompt) or ("r", "local"))()
    assert seen["p"] == "T\n\n---\n\nDATA"


def test_cli_task_without_input_uses_task_only(monkeypatch):
    seen = {}
    _cli(monkeypatch, ["--task", "T"], lambda prompt, **kw: seen.update(p=prompt) or ("r", "local"))()
    assert seen["p"] == "T"


def test_cli_unreadable_input_exits_1(monkeypatch, tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        _cli(monkeypatch, ["--task", "T", "--input", str(tmp_path / "missing.txt")],
             lambda *a, **k: pytest.fail("route must not run"))()
    assert e.value.code == 1 and "Could not read --input file" in capsys.readouterr().err


def test_cli_route_failure_exits_1(monkeypatch, capsys):
    def boom(prompt, **kw):
        raise RuntimeError("both down")
    with pytest.raises(SystemExit) as e:
        _cli(monkeypatch, ["--prompt", "P"], boom)()
    assert e.value.code == 1 and "FAILED: both down" in capsys.readouterr().err


def test_cli_output_file_written_not_printed(monkeypatch, tmp_path, capsys):
    out = tmp_path / "o.txt"
    _cli(monkeypatch, ["--prompt", "P", "--output", str(out)], lambda p, **k: ("RESULT", "local"))()
    assert out.read_text(encoding="utf-8") == "RESULT"
    assert capsys.readouterr().out == f"Result written to {out} (engine: local)\n"
