"""Tests for the `drift mcp` server helpers (drift/mcp_server.py).

The key regression: `drift_run` must work when invoked from an already-running
event loop (the MCP server's), which previously crashed with
"asyncio.run() cannot be called from a running event loop".
"""
import os

import pytest

from drift.mcp_server import _run, _check, _transpile, _transpile_result


HELLO = (
    'agent Hi { step greet(name: string) -> string { '
    'respond "Hi {name}" return "Hi {name}" } }'
)

SCHEDULE_PIPELINE = (
    'pipeline Triage {\n'
    '  schedule: "every morning"\n'
    '  input_email -> Classifier.tag\n'
    '}\n'
)


ORDER_SRC = (
    'agent Zeta { step greet() -> string { return "Hello from Zeta" } } '
    'agent Alpha { step greet() -> string { return "Hello from Alpha" } }'
)


class TestDriftRun:
    @pytest.mark.asyncio
    async def test_run_selects_first_declared_agent_not_alphabetical(
        self, monkeypatch
    ):
        # Regression: agents were collected via dir(module) — alphabetical
        # order — then `next(iter(agents.values()))` silently ran
        # whichever agent's class name sorted first, contradicting
        # LLM.md's documented "runs the first agent's first step".
        monkeypatch.setenv("DRIFT_USE_MOCK", "1")
        result = await _run(ORDER_SRC, "{}")
        assert result["ok"] is True
        assert result["result"] == "Hello from Zeta"

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
    async def test_cross_file_import_gets_actionable_hint(self, monkeypatch):
        # drift_run takes raw source TEXT, not a file path — a cross-file
        # `import { X } from "./other.drift"` has no directory to resolve
        # against and always fails with a bare ModuleNotFoundError. Without
        # a hint, a calling agent would have no way to know this is a
        # structural limitation (source text vs. file path) rather than a
        # bug in their program — `drift run <file>` via the CLI resolves
        # this exact pattern correctly, so the error needs to say why this
        # specific tool can't.
        monkeypatch.setenv("DRIFT_USE_MOCK", "1")
        # Use a distinctive module name — a real .drift dependency named
        # exactly this is exceedingly unlikely to exist anywhere on
        # sys.path, avoiding any risk of colliding with an unrelated
        # test's own transpiled dependency of the same name still cached
        # in sys.modules from earlier in the same pytest process (e.g.
        # test_cli.py's cross-file-import tests, which legitimately leave
        # their own dependency modules cached after passing — clearing
        # sys.modules on every _run_once call is what THAT bug fix is
        # about; this test exercises a different function, mcp_server's
        # _run(), which never touches sys.modules at all).
        src = (
            'import { Shared } from "./mcp_server_test_nonexistent_dep_xyz.drift"\n'
            'agent A { model: "claude-haiku" '
            '  step f(item: string) -> string { return item } }'
        )
        result = await _run(src)
        assert result["ok"] is False
        assert result["stage"] == "import"
        assert "ModuleNotFoundError" in result["error"]
        assert "drift_run can't resolve cross-file" in result["error"]

    @pytest.mark.asyncio
    async def test_unrelated_module_not_found_has_no_misleading_hint(self, monkeypatch):
        # A `tool ... from python "module:fn"` referencing a genuinely
        # missing Python module raises the same ModuleNotFoundError type
        # as the cross-file-import case, but for an unrelated reason — the
        # cross-file-import hint must not fire when the source has no
        # `.drift`-suffixed import at all.
        monkeypatch.setenv("DRIFT_USE_MOCK", "1")
        src = (
            'tool calc from python "nonexistent_module_xyz:some_fn"\n'
            'agent A { model: "claude-haiku" '
            '  step f() -> string { let x = calc() return x } }'
        )
        result = await _run(src)
        assert result["ok"] is False
        assert result["stage"] == "import"
        assert "ModuleNotFoundError" in result["error"]
        assert "drift_run can't resolve cross-file" not in result["error"]

    @pytest.mark.asyncio
    async def test_run_restores_module_slot(self, monkeypatch):
        import sys
        monkeypatch.setenv("DRIFT_USE_MOCK", "1")
        had = "drift_mcp_program" in sys.modules
        await _run(HELLO, '{"name": "x"}')
        # The transient program module must not linger in sys.modules.
        assert ("drift_mcp_program" in sys.modules) == had

    @pytest.mark.asyncio
    async def test_run_returns_outputs_not_raw_banner(self, monkeypatch):
        # Ergonomics: `respond` output is genuinely useful info (distinct
        # from the step's return value) but used to be buried unstructured
        # inside a `banner` field that was otherwise pure duplicate text
        # (box-drawing header, re-printed cost report) paid on every call.
        # It should now be its own structured field, and `banner` gone.
        monkeypatch.setenv("DRIFT_USE_MOCK", "1")
        result = await _run(HELLO, '{"name": "Riley"}')
        assert result["outputs"] == ["Hi Riley"]
        assert "banner" not in result

    @pytest.mark.asyncio
    async def test_run_reports_outputs_on_partial_failure(self, monkeypatch):
        # respond output produced before a mid-step failure shouldn't be
        # lost — it's exactly the kind of partial-progress signal a caller
        # debugging a StepFailed wants.
        monkeypatch.setenv("DRIFT_USE_MOCK", "1")
        source = (
            'agent Hi { step greet(name: string) -> string { '
            'respond "trying..." fail "not implemented" } }'
        )
        result = await _run(source, '{"name": "Riley"}')
        assert result["ok"] is False
        assert result["outputs"] == ["trying..."]
        assert "banner" not in result


PIPELINE_SRC = (
    'agent Triager { step classify(x: string) -> string { return "urgent" } } '
    'pipeline P { '
    '  use Triager '
    '  step tickets(batch: list<string>) -> list<string> { return batch } '
    '  tickets -> Triager.classify '
    '}'
)

PIPELINE_ONLY_SRC = (
    'pipeline OnlyPipeline { '
    '  step tickets(batch: list<string>) -> list<string> { return batch } '
    '  step process(batch: list<string>) -> string { return "done" } '
    '  tickets -> process '
    '}'
)


class TestDriftRunPipelines:
    """Regression: drift_run never looked for pipelines at all — a source
    declaring a `pipeline` (with or without agents) either silently ran an
    agent step instead of the pipeline, or crashed with a misleading
    `kind: "bug"` (AttributeError from feeding pipeline-shaped input into
    the dict-oriented agent-input coercion path) when pipeline-shaped
    input was passed. LLM.md's MCP section documents only ONE limitation
    for drift_run (cross-file imports) — this gap was undocumented.
    """

    @pytest.mark.asyncio
    async def test_pipeline_arg_runs_the_named_pipeline(self, monkeypatch):
        monkeypatch.setenv("DRIFT_USE_MOCK", "1")
        result = await _run(PIPELINE_SRC, '["a", "b"]', pipeline="P")
        assert result == {"ok": True, "result": "urgent"}

    @pytest.mark.asyncio
    async def test_without_pipeline_arg_agent_still_runs_by_default(
        self, monkeypatch
    ):
        # Backward compatible: a source with BOTH an agent and a pipeline,
        # called with no `pipeline` arg, still runs the agent — existing
        # callers that never knew about pipelines see no behavior change.
        monkeypatch.setenv("DRIFT_USE_MOCK", "1")
        result = await _run(PIPELINE_SRC, '{"x": "hi"}')
        assert result["ok"] is True
        assert result["result"] == "urgent"

    @pytest.mark.asyncio
    async def test_pipeline_only_source_without_pipeline_arg_gives_actionable_error(
        self, monkeypatch
    ):
        # Previously: silently ran nothing sensible or crashed opaquely.
        # Now: a clear, actionable {ok: false, ...} naming the available
        # pipeline, not a raw AttributeError misattributed as kind: "bug".
        monkeypatch.setenv("DRIFT_USE_MOCK", "1")
        result = await _run(PIPELINE_ONLY_SRC, '["a"]')
        assert result["ok"] is False
        assert "OnlyPipeline" in result["error"]

    @pytest.mark.asyncio
    async def test_pipeline_only_source_with_pipeline_arg_runs(self, monkeypatch):
        monkeypatch.setenv("DRIFT_USE_MOCK", "1")
        result = await _run(PIPELINE_ONLY_SRC, '["a"]', pipeline="OnlyPipeline")
        assert result == {"ok": True, "result": "done"}

    @pytest.mark.asyncio
    async def test_unknown_pipeline_name_gives_actionable_error(self, monkeypatch):
        monkeypatch.setenv("DRIFT_USE_MOCK", "1")
        result = await _run(PIPELINE_SRC, '[]', pipeline="DoesNotExist")
        assert result["ok"] is False
        assert "not found" in result["error"]
        assert "P" in result["error"]


class TestDriftRunMalformedInput:
    @pytest.mark.asyncio
    async def test_malformed_input_json_returns_ok_false_not_raises(
        self, monkeypatch
    ):
        # Regression: json.loads(input_json) was called before the
        # try/except that wraps the rest of _run, so malformed JSON raised
        # straight out of the function instead of the documented
        # {ok: false, ...} envelope every other failure mode here uses.
        monkeypatch.setenv("DRIFT_USE_MOCK", "1")
        result = await _run(HELLO, "not valid json")
        assert result["ok"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_malformed_input_json_with_pipeline_also_returns_ok_false(
        self, monkeypatch
    ):
        monkeypatch.setenv("DRIFT_USE_MOCK", "1")
        result = await _run(PIPELINE_SRC, "{not json", pipeline="P")
        assert result["ok"] is False
        assert "error" in result


class TestCheckAndTranspile:
    def test_check_ok(self):
        assert _check(HELLO) == {"ok": True}

    def test_transpile_emits_python(self):
        py = _transpile(HELLO)
        assert "class Hi" in py

    def test_check_failure_uses_error_field_not_message(self):
        # Regression: _check used to return "message" for failure text
        # while drift_run/drift_transpile use "error" — a caller writing
        # one generic `if not ok: report(result["error"])` handler across
        # all three tools would KeyError on a _check failure.
        result = _check('agent Hi { step f() { let x = "unterminated } }')
        assert result["ok"] is False
        assert "error" in result
        assert "message" not in result

    def test_transpile_result_catches_codegen_error(self):
        # Regression: drift_transpile's MCP handler only caught
        # (LexError, ParseError), so a construct that parses but can't
        # compile (e.g. an unimplemented `schedule:` pipeline modifier)
        # would raise CodegenError uncaught out of call_tool instead of
        # the same clean {ok: false, ...} shape drift_check/drift_run give
        # it for the identical input.
        result = _transpile_result(SCHEDULE_PIPELINE)
        assert result["ok"] is False
        assert "schedule" in result["error"]
        assert "not implemented" in result["error"]

    def test_check_and_transpile_result_agree_on_codegen_failure(self):
        # Same source, same failure class — both tools should report it
        # the same way (same field names, same message), not diverge.
        check_result = _check(SCHEDULE_PIPELINE)
        transpile_result = _transpile_result(SCHEDULE_PIPELINE)
        assert check_result["ok"] is False
        assert transpile_result["ok"] is False
        assert check_result["error"] == transpile_result["error"]
