"""Tests for `for each ... parallel` and cross-agent calls.

The parser flag and codegen template existed before Phase 3, but two
things didn't work:
  - checkpoint.save inside a parallel step saved under the inner _task
    function's name instead of the step name
  - cross-agent invocations (`OtherAgent.step(args)`) emitted unbound
    method calls (`OtherAgent.step(args)`) which crash at runtime

Both are fixed; these tests pin the fixed behavior down."""
import asyncio
import sys
from pathlib import Path

import pytest

from drift import ast_nodes as ast


# ── Parser ─────────────────────────────────────────────────────────────


class TestParallelParse:
    def test_parallel_flag_set(self, parse_ast):
        d = parse_ast(
            'agent A { '
            'step f(xs: list<string>) { '
            '  for each x in xs parallel { respond x } '
            '} }'
        ).declarations[0]
        for_each = d.steps[0].body[0]
        assert isinstance(for_each, ast.ForEachStmt)
        assert for_each.parallel is True

    def test_sequential_default(self, parse_ast):
        d = parse_ast(
            'agent A { '
            'step f(xs: list<string>) { '
            '  for each x in xs { respond x } '
            '} }'
        ).declarations[0]
        assert d.steps[0].body[0].parallel is False


# ── Codegen ────────────────────────────────────────────────────────────


class TestParallelCodegen:
    def test_emits_asyncio_gather(self, transpile):
        py = transpile(
            'agent A { '
            'step f(xs: list<string>) -> string { '
            '  for each x in xs parallel { respond x } '
            '  return "done" '
            '} }'
        )
        assert "asyncio.gather" in py
        assert "async def _task(x):" in py

    def test_checkpoint_uses_step_name_not_task(self, transpile):
        """Regression: the heuristic used to walk back through emitted
        lines to find the most recent `async def`, which was the
        parallel block's _task wrapper. Now it uses the explicit step
        name stack."""
        py = transpile(
            'agent A { '
            'step process(xs: list<string>) -> string { '
            '  for each x in xs parallel { respond x } '
            '  return "ok" '
            '} }'
        )
        assert "self.checkpoint.save('process', _result)" in py
        assert "self.checkpoint.save('_task'" not in py


# ── Cross-agent calls ──────────────────────────────────────────────────


class TestCrossAgentCall:
    def test_emits_instantiation_and_await(self, transpile):
        py = transpile(
            'schema FitScore { score: number } '
            'agent GrantChecker { '
            '  step evaluate(c: string) -> FitScore { '
            '    return FitScore { score: 80 } '
            '  } '
            '} '
            'agent Pipeline { '
            '  step process(cs: list<string>) -> string { '
            '    for each c in cs parallel { '
            '      let s = GrantChecker.evaluate(c) '
            '    } '
            '    return "done" '
            '  } '
            '}'
        )
        # The fix: PascalCase call target resolved as an agent name →
        # `await GrantChecker().evaluate(c)` (instantiation + await).
        assert "await GrantChecker().evaluate(c)" in py
        # Negative — the broken old form should be gone.
        assert "GrantChecker.evaluate(c)" not in py.replace(
            "await GrantChecker().evaluate(c)", "",  # mask the good one
        )

    def test_unknown_type_target_not_treated_as_agent(self, transpile):
        """A non-agent PascalCase target (e.g. an enum or external
        symbol) must keep the literal call shape — we only special-case
        names we know are agents in this program."""
        py = transpile(
            'agent A { '
            '  step f() -> string { '
            '    let r = NotAnAgent.method() '
            '    return "x" '
            '  } '
            '}'
        )
        # No `await NotAnAgent()` instantiation — the call passes through.
        assert "await NotAnAgent()" not in py
        assert "NotAnAgent.method()" in py


# ── End-to-end execution ───────────────────────────────────────────────


class TestParallelRuntime:
    """Verify the generated code actually runs. Uses the mock provider
    via the autouse conftest fixture so no API keys are needed."""

    @pytest.mark.asyncio
    async def test_parallel_step_runs_and_returns(self, transpile, tmp_path,
                                                   monkeypatch):
        py = transpile(
            'agent A { '
            '  model: "claude-haiku" '
            '  step process(xs: list<string>) -> list<string> { '
            '    let results = [] '
            '    for each x in xs parallel { '
            '      let r = classify x as string '
            '      results.add(r) '
            '    } '
            '    return results '
            '  } '
            '}'
        )
        # Write the generated module to disk and import it.
        mod_path = tmp_path / "agent_under_test.py"
        mod_path.write_text(py)
        monkeypatch.syspath_prepend(str(tmp_path))

        import importlib
        # Ensure a fresh import even if the test file name was used before.
        if "agent_under_test" in sys.modules:
            del sys.modules["agent_under_test"]
        mod = importlib.import_module("agent_under_test")

        agent = mod.A()
        result = await agent.process(["a", "b", "c"])
        # MockProvider returns plausible strings for classify; we only
        # care that the parallel block completed and accumulated 3 items.
        assert isinstance(result, list)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_one_failed_item_loses_the_whole_batch_without_attempt(
        self, transpile, tmp_path, monkeypatch
    ):
        # Documents real, current behavior (LLM.md's "Parallel triage"
        # pattern, as written with no attempt/recover): `parallel`
        # compiles to a plain asyncio.gather with no return_exceptions —
        # one item's intent call ultimately failing propagates out of the
        # WHOLE for-each, discarding every already-collected result along
        # with it, not just the failed item's slot. This isn't a bug to
        # fix (gather's default semantics are standard and arguably
        # correct to fail loud by default) but is exactly the kind of
        # silent-until-it-bites-you gap the language docs need to call
        # out explicitly, since "parallel batch processing" reads as if
        # it should be resilient by default.
        py = transpile(
            'agent A { '
            '  model: "claude-haiku" '
            '  step process(xs: list<string>) -> list<string> { '
            '    let results = [] '
            '    for each x in xs parallel { '
            '      let r = classify x as string '
            '      results.add(r) '
            '    } '
            '    return results '
            '  } '
            '}'
        )
        mod_path = tmp_path / "agent_under_test_fail.py"
        mod_path.write_text(py)
        monkeypatch.syspath_prepend(str(tmp_path))
        import importlib, sys as _sys
        if "agent_under_test_fail" in _sys.modules:
            del _sys.modules["agent_under_test_fail"]
        mod = importlib.import_module("agent_under_test_fail")

        agent = mod.A()

        async def flaky_intent(verb, input_data=None, output_schema=None, **kwargs):
            if input_data == "bad":
                raise ValueError("simulated LLM failure")
            return f"ok:{input_data}"
        agent.intent = flaky_intent

        with pytest.raises(ValueError):
            await agent.process(["a", "bad", "c"])
        # The whole call raised — there is no partial "results" to
        # inspect from the caller's side, confirming the batch-loss.

    @pytest.mark.asyncio
    async def test_attempt_recover_inside_parallel_isolates_failures(
        self, transpile, tmp_path, monkeypatch
    ):
        # The documented fix (LLM.md's updated Parallel triage note):
        # wrapping the per-item work in attempt/recover INSIDE the
        # parallel body makes each item's failure independent — a caught
        # item is simply missing from results instead of sinking the
        # whole batch.
        py = transpile(
            'agent A { '
            '  model: "claude-haiku" '
            '  step process(xs: list<string>) -> list<string> { '
            '    let results = [] '
            '    for each x in xs parallel { '
            '      attempt { '
            '        let r = classify x as string '
            '        results.add(r) '
            '      } recover from { '
            '        any error -> respond "skipping" '
            '      } '
            '    } '
            '    return results '
            '  } '
            '}'
        )
        mod_path = tmp_path / "agent_under_test_resilient.py"
        mod_path.write_text(py)
        monkeypatch.syspath_prepend(str(tmp_path))
        import importlib, sys as _sys
        if "agent_under_test_resilient" in _sys.modules:
            del _sys.modules["agent_under_test_resilient"]
        mod = importlib.import_module("agent_under_test_resilient")

        agent = mod.A()

        async def flaky_intent(verb, input_data=None, output_schema=None, **kwargs):
            if input_data == "bad":
                raise ValueError("simulated LLM failure")
            return f"ok:{input_data}"
        agent.intent = flaky_intent

        result = await agent.process(["a", "bad", "c"])
        # The two good items survive; the bad one is simply absent —
        # no exception, no lost batch.
        assert sorted(result) == ["ok:a", "ok:c"]
