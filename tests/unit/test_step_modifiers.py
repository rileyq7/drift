"""Tests for step modifiers: cached, silent, manual, parallel.

cached/silent/manual are real runtime behavior (see step_decorator and
run_agent in drift/runtime/core.py). `parallel step` has no well-defined
semantics of its own — it's rejected at codegen instead of silently doing
nothing (see TestParallelModifierRejected).
"""
import importlib.util
import sys

import pytest

from drift.codegen import CodegenError


def _load(transpile, src, mod_name, tmp_path):
    py = transpile(src).replace("Source: <drift_file>", "Source: inline")
    path = tmp_path / f"{mod_name}.py"
    path.write_text(py)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestParallelModifierRejected:
    def test_parallel_step_is_codegen_error(self, transpile):
        with pytest.raises(CodegenError, match="parallel step"):
            transpile("agent A { parallel step f() -> string { return \"x\" } }")


class TestCachedStep:
    @pytest.mark.asyncio
    async def test_second_call_with_same_args_is_memoized(self, transpile, tmp_path):
        # Each real invocation appends one output via `respond`; if caching
        # works, the second call with identical args must not invoke the
        # underlying function again, so _outputs must not grow.
        src = (
            "agent A { "
            "  cached step f(x: string) -> string { "
            "    respond \"ran\" "
            "    return x "
            "  } "
            "}"
        )
        mod = _load(transpile, src, "cached_mod", tmp_path)
        agent = mod.A()
        r1 = await agent.f("hi")
        r2 = await agent.f("hi")
        assert r1 == r2 == "hi"
        assert agent._outputs == ["ran"]

    @pytest.mark.asyncio
    async def test_different_args_are_not_conflated(self, transpile, tmp_path):
        src = (
            "agent A { "
            "  cached step f(x: string) -> string { return x } "
            "}"
        )
        mod = _load(transpile, src, "cached_mod2", tmp_path)
        agent = mod.A()
        assert await agent.f("a") == "a"
        assert await agent.f("b") == "b"


class TestSilentStep:
    @pytest.mark.asyncio
    async def test_respond_is_suppressed_during_silent_step(self, transpile, tmp_path):
        src = (
            "agent A { "
            "  silent step f() -> string { "
            "    respond \"should not appear\" "
            "    return \"done\" "
            "  } "
            "}"
        )
        mod = _load(transpile, src, "silent_mod", tmp_path)
        agent = mod.A()
        result = await agent.f()
        assert result == "done"
        assert agent._outputs == []

    @pytest.mark.asyncio
    async def test_respond_still_works_in_non_silent_step(self, transpile, tmp_path):
        src = (
            "agent A { "
            "  step f() -> string { "
            "    respond \"visible\" "
            "    return \"done\" "
            "  } "
            "}"
        )
        mod = _load(transpile, src, "silent_mod2", tmp_path)
        agent = mod.A()
        await agent.f()
        assert agent._outputs == ["visible"]

    @pytest.mark.asyncio
    async def test_nested_non_silent_step_called_from_silent_step_is_not_suppressed(
        self, transpile, tmp_path
    ):
        # Regression: LLM.md documents silent as scoped to "that step
        # only; nested non-silent steps called from within it are
        # unaffected once they return." But `_drift_silent` is a single
        # flag on the AGENT INSTANCE, not scoped per call frame — the
        # old code only ever set it to True (never explicitly False), so
        # a silent step calling a non-silent step internally left the
        # flag set to True for the whole nested call, silently
        # suppressing the inner step's own respond output too.
        src = (
            "agent A { "
            "  silent step outer() -> string { "
            "    respond \"outer message\" "
            "    let x = inner() "
            "    return x "
            "  } "
            "  step inner() -> string { "
            "    respond \"inner message\" "
            "    return \"done\" "
            "  } "
            "}"
        )
        mod = _load(transpile, src, "silent_nested_mod", tmp_path)
        agent = mod.A()
        result = await agent.outer()
        assert result == "done"
        # "outer message" correctly suppressed (outer IS silent);
        # "inner message" must NOT be suppressed (inner is not silent).
        assert agent._outputs == ["inner message"]

    @pytest.mark.asyncio
    async def test_silence_fully_restored_after_silent_step_returns(
        self, transpile, tmp_path
    ):
        # A silent step's silence must not leak past its own return —
        # a later, unrelated non-silent step call on the same agent
        # instance must show its output normally.
        src = (
            "agent A { "
            "  silent step outer() -> string { "
            "    respond \"outer message\" "
            "    return \"done\" "
            "  } "
            "  step after_outer() -> string { "
            "    respond \"after outer\" "
            "    return \"ok\" "
            "  } "
            "}"
        )
        mod = _load(transpile, src, "silent_restore_mod", tmp_path)
        agent = mod.A()
        await agent.outer()
        assert agent._outputs == []
        await agent.after_outer()
        assert agent._outputs == ["after outer"]


class TestManualStep:
    @pytest.mark.asyncio
    async def test_manual_step_is_not_auto_selected_as_entry_point(self, transpile, tmp_path):
        # `run_agent` with no --step picks the first-DECLARED step. A manual
        # step must be skipped by that selection even if declared first.
        from drift.runtime.core import run_agent

        src = (
            "agent A { "
            "  manual step admin() -> string { return \"admin\" } "
            "  step normal() -> string { return \"normal\" } "
            "}"
        )
        mod = _load(transpile, src, "manual_mod", tmp_path)
        result = await run_agent(mod.A)
        assert result == "normal"

    @pytest.mark.asyncio
    async def test_manual_step_still_runs_via_explicit_step_name(self, transpile, tmp_path):
        from drift.runtime.core import run_agent

        src = (
            "agent A { "
            "  manual step admin() -> string { return \"admin\" } "
            "  step normal() -> string { return \"normal\" } "
            "}"
        )
        mod = _load(transpile, src, "manual_mod2", tmp_path)
        result = await run_agent(mod.A, step_name="admin")
        assert result == "admin"
