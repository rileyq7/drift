"""Tests for the `drift mcp` server helpers (drift/mcp_server.py).

The key regression: `drift_run` must work when invoked from an already-running
event loop (the MCP server's), which previously crashed with
"asyncio.run() cannot be called from a running event loop".
"""
import os

import pytest

from drift.mcp_server import _run, _check, _transpile


HELLO = (
    'agent Hi { step greet(name: string) -> string { '
    'respond "Hi {name}" return "Hi {name}" } }'
)


class TestDriftRun:
    @pytest.mark.asyncio
    async def test_run_inside_event_loop(self, monkeypatch):
        # We are inside pytest-asyncio's running loop — this is exactly the
        # condition that used to raise RuntimeError.
        monkeypatch.setenv("DRIFT_USE_MOCK", "1")
        result = await _run(HELLO, '{"name": "Riley"}')
        assert result["ok"] is True
        assert result["result"] == "Hi Riley"

    @pytest.mark.asyncio
    async def test_run_reports_structured_cost_on_success(self, monkeypatch):
        # drift_run's docstring promises a `cost` field, not just a
        # human-readable `banner` — a calling agent shouldn't have to
        # parse the printed cost report to know what a run spent.
        monkeypatch.setenv("DRIFT_USE_MOCK", "1")
        result = await _run(HELLO, '{"name": "Riley"}')
        assert result["ok"] is True
        assert "cost" in result
        assert "total_cost" in result["cost"]
        assert "calls" in result["cost"]

    @pytest.mark.asyncio
    async def test_run_reports_kind_and_cost_on_step_failure(self, monkeypatch):
        monkeypatch.setenv("DRIFT_USE_MOCK", "1")
        # A step with no `as <Type>` clause and a schema mismatch is hard to
        # force generically here, so instead exercise a step that always
        # fails via `fail "..."` — this raises StepFailed, a business-logic
        # outcome distinct from an infra/codegen bug.
        source = (
            'agent Hi { step greet(name: string) -> string { '
            'fail "not implemented" } }'
        )
        result = await _run(source, '{"name": "Riley"}')
        assert result["ok"] is False
        assert result["stage"] == "run"
        assert result["kind"] == "business-logic"
        assert "cost" in result

    @pytest.mark.asyncio
    async def test_run_reports_lex_error_cleanly(self, monkeypatch):
        monkeypatch.setenv("DRIFT_USE_MOCK", "1")
        result = await _run('agent Hi { step f() { let x = "unterminated } }')
        assert result["ok"] is False
        assert result["stage"] in ("lex", "parse")

    @pytest.mark.asyncio
    async def test_run_restores_module_slot(self, monkeypatch):
        import sys
        monkeypatch.setenv("DRIFT_USE_MOCK", "1")
        had = "drift_mcp_program" in sys.modules
        await _run(HELLO, '{"name": "x"}')
        # The transient program module must not linger in sys.modules.
        assert ("drift_mcp_program" in sys.modules) == had


class TestCheckAndTranspile:
    def test_check_ok(self):
        assert _check(HELLO) == {"ok": True}

    def test_transpile_emits_python(self):
        py = _transpile(HELLO)
        assert "class Hi" in py
