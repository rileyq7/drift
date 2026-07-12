"""Tests for §10 imports + stdlib stubs."""
import asyncio

import pytest

from drift import ast_nodes as ast


class TestImportParse:
    def test_single_name(self, parse_ast):
        p = parse_ast('import GrantChecker from "./agents/grants.drift"')
        d = p.declarations[0]
        assert isinstance(d, ast.ImportDecl)
        assert d.names == ["GrantChecker"]
        assert d.source_path == "./agents/grants.drift"

    def test_multiple_names_comma(self, parse_ast):
        p = parse_ast('import GrantChecker, Notifier from "./agents.drift"')
        d = p.declarations[0]
        assert d.names == ["GrantChecker", "Notifier"]

    def test_brace_form(self, parse_ast):
        p = parse_ast('import { fetch_url, load_pdf } from "drift/io"')
        d = p.declarations[0]
        assert set(d.names) == {"fetch_url", "load_pdf"}
        assert d.source_path == "drift/io"

    def test_mixed_case_names_ok(self, parse_ast):
        p = parse_ast('import { Notifier, email } from "drift/notify"')
        d = p.declarations[0]
        assert "Notifier" in d.names
        assert "email" in d.names


class TestImportCodegen:
    def test_sibling_drift_file(self, transpile):
        out = transpile('import GrantChecker from "./agents/grants.drift"')
        assert "from grants import GrantChecker" in out

    def test_stdlib_path(self, transpile):
        out = transpile('import { fetch_url, load_pdf } from "drift/io"')
        assert "from drift.io import fetch_url, load_pdf" in out

    def test_multiple_imports(self, transpile):
        out = transpile(
            'import GrantChecker from "./grants.drift"\n'
            'import { email } from "drift/notify"'
        )
        assert "from grants import GrantChecker" in out
        assert "from drift.notify import email" in out


class TestAsyncStdlibCallsAreAwaited:
    """Regression: a bare call to an imported stdlib function
    (`fetch_url(url)`, matching the actual `from drift.io import
    fetch_url` codegen — there is no `io.` prefix at the call site) went
    through gen_fn_call's no-target fallback, which never emitted `await`.
    For the 3 stdlib functions that are actually `async def`
    (fetch_url, webhook, wait), this silently returned an unawaited
    coroutine — the real HTTP request / webhook POST / sleep never
    happened — with no error, only a RuntimeWarning at GC time. Every
    other stdlib function is plain sync and must NOT get `await` (that
    would be a TypeError, worse than the original bug)."""

    def _step_body(self, transpile, import_line: str, call_line: str) -> str:
        src = (
            f'{import_line}\n'
            f'agent A {{ model: "claude-haiku" '
            f'  step f(url: string) -> string {{ {call_line} return "x" }} }}'
        )
        return transpile(src)

    def test_fetch_url_call_is_awaited(self, transpile):
        out = self._step_body(
            transpile,
            'import { fetch_url } from "drift/io"',
            'let content = fetch_url(url)',
        )
        assert "content = await fetch_url(url)" in out

    def test_wait_call_is_awaited(self, transpile):
        out = self._step_body(
            transpile,
            'import { wait } from "drift/time"',
            'wait(1.0)',
        )
        assert "await wait(" in out

    def test_webhook_call_is_awaited(self, transpile):
        out = self._step_body(
            transpile,
            'import { webhook } from "drift/notify"',
            'webhook(url, payload)',
        )
        assert "await webhook(url, payload)" in out

    def test_sync_stdlib_call_is_not_awaited(self, transpile):
        # read/redact_pii/etc. are plain sync — must NOT get `await`,
        # which would be a TypeError against a non-awaitable return value.
        out = self._step_body(
            transpile,
            'import { read } from "drift/io"',
            'let content = read(url)',
        )
        assert "content = read(url)" in out
        assert "await read(url)" not in out

    @pytest.mark.asyncio
    async def test_fetch_url_call_actually_executes(self, transpile, tmp_path, monkeypatch):
        # End-to-end proof the real function body runs (not just an
        # unawaited coroutine object silently discarded).
        from drift.runtime import run_agent

        py = transpile(
            'import { fetch_url } from "drift/io"\n'
            'agent A { model: "claude-haiku" '
            '  step f(url: string) -> string { '
            '    let content = fetch_url(url) '
            '    return content '
            '  } }'
        )
        mod_path = tmp_path / "stdlib_call_under_test.py"
        mod_path.write_text(py)
        monkeypatch.syspath_prepend(str(tmp_path))

        import importlib, sys
        if "stdlib_call_under_test" in sys.modules:
            del sys.modules["stdlib_call_under_test"]
        mod = importlib.import_module("stdlib_call_under_test")

        # Monkeypatch fetch_url in the generated module's namespace to
        # prove it's actually called (and awaited) rather than hitting
        # the network in a test.
        called = {}
        async def fake_fetch_url(url):
            called["url"] = url
            return "fetched-content"
        mod.fetch_url = fake_fetch_url

        result = await run_agent(mod.A, inputs={"url": "https://example.com"})
        assert result == "fetched-content"
        assert called["url"] == "https://example.com"


class TestStdlibIO:
    def test_read_write_round_trip(self, tmp_path):
        from drift.io import read, write
        p = tmp_path / "x.txt"
        write(str(p), "hello\nworld\n")
        assert read(str(p)) == "hello\nworld\n"

    def test_load_csv(self, tmp_path):
        from drift.io import load_csv
        p = tmp_path / "x.csv"
        p.write_text("name,age\nalice,30\nbob,25\n")
        rows = load_csv(str(p))
        assert rows == [{"name": "alice", "age": "30"}, {"name": "bob", "age": "25"}]


class TestStdlibNotify:
    def test_email_stub_does_not_raise(self, capsys):
        from drift.notify import email
        email(to="x@y.com", subject="hi", body="hello")
        captured = capsys.readouterr()
        assert "x@y.com" in captured.out

    def test_slack_stub(self, capsys):
        from drift.notify import slack
        slack("general", "hi team")
        assert "general" in capsys.readouterr().out

    def test_push_stub(self, capsys):
        from drift.notify import push
        push("title", "body")
        assert "title" in capsys.readouterr().out


class TestStdlibSafety:
    def test_redact_pii(self):
        from drift.safety import redact_pii
        out = redact_pii("Contact alice@example.com or call 555-123-4567 about SSN 123-45-6789")
        assert "[EMAIL]" in out
        assert "[PHONE]" in out
        assert "[SSN]" in out
        assert "alice@example.com" not in out

    def test_check_content_allows_clean_text(self):
        from drift.safety import check_content
        assert check_content("hello world", banned_patterns=["forbidden"])

    def test_check_content_blocks_banned(self):
        from drift.safety import check_content
        assert not check_content("this is forbidden", banned_patterns=["forbidden"])

    def test_sanitize_trims_control_chars(self):
        from drift.safety import sanitize
        out = sanitize("hello\x01world", max_length=100)
        assert "\x01" not in out

    def test_rate_limit_caps_calls(self):
        from drift.safety import rate_limit
        key = "test_unique_42"
        # Allow 3 calls per 60s
        assert rate_limit(key, max_per_window=3, window_seconds=60.0)
        assert rate_limit(key, max_per_window=3, window_seconds=60.0)
        assert rate_limit(key, max_per_window=3, window_seconds=60.0)
        # 4th should be blocked
        assert not rate_limit(key, max_per_window=3, window_seconds=60.0)


class TestStdlibData:
    def test_filter(self):
        from drift.data import filter as df
        assert df([1, 2, 3, 4, 5], lambda x: x > 2) == [3, 4, 5]

    def test_sort_descending(self):
        from drift.data import sort
        assert sort([3, 1, 4, 1, 5], descending=True) == [5, 4, 3, 1, 1]

    def test_group_by(self):
        from drift.data import group_by
        groups = group_by([1, 2, 3, 4], key=lambda x: x % 2)
        assert groups == {1: [1, 3], 0: [2, 4]}

    def test_deduplicate_preserves_order(self):
        from drift.data import deduplicate
        assert deduplicate([3, 1, 2, 1, 3, 4]) == [3, 1, 2, 4]

    def test_paginate(self):
        from drift.data import paginate
        items = list(range(10))
        assert paginate(items, page=1, page_size=3) == [0, 1, 2]
        assert paginate(items, page=2, page_size=3) == [3, 4, 5]


class TestStdlibText:
    def test_chunk(self):
        from drift.text import chunk
        out = chunk("a" * 100, max_chars=30, overlap=5)
        assert all(len(c) <= 30 for c in out)

    def test_similarity_identity(self):
        from drift.text import similarity
        assert similarity("hello world", "hello world") == 1.0

    def test_similarity_no_overlap(self):
        from drift.text import similarity
        assert similarity("apple", "banana") == 0.0

    def test_embed_stub_raises(self):
        from drift.text import embed
        with pytest.raises(NotImplementedError):
            embed("text")


class TestStdlibObserve:
    def test_log_outputs_json(self, capsys):
        from drift.observe import log
        log("event", key="value")
        out = capsys.readouterr().out
        import json
        record = json.loads(out.strip().split("\n")[0])
        assert record["message"] == "event"
        assert record["key"] == "value"

    def test_metric(self, capsys):
        from drift.observe import metric
        metric("latency_ms", 42.0, route="/x")
        out = capsys.readouterr().out
        import json
        record = json.loads(out.strip().split("\n")[0])
        assert record["metric"] == "latency_ms"
        assert record["value"] == 42.0


class TestStdlibTime:
    @pytest.mark.asyncio
    async def test_wait(self):
        from drift.time import wait
        import time as _time
        start = _time.time()
        await wait(0.05)
        assert _time.time() - start >= 0.04

    def test_deadline_is_future(self):
        from drift.time import deadline
        import time as _time
        d = deadline(10)
        assert d > _time.time()

    def test_schedule_appends(self):
        from drift.time import schedule, SCHEDULES
        before = len(SCHEDULES)
        schedule("every Monday at 9am")
        assert SCHEDULES[-1] == "every Monday at 9am"
        # Cleanup so other tests don't see this
        SCHEDULES.pop()
