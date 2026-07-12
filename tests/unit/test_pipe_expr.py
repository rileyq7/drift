"""Tests for the expression-level `|>` pipe operator (function composition
inside an expression — distinct from a pipeline declaration's `|>` edge,
which is unimplemented/rejected at codegen, see TestPipelineCodegen in
test_pipeline.py).
"""
import pytest


class TestPipeAwaitsAsyncStages:
    """Regression: gen_pipe hand-built each stage's call text directly
    instead of routing through gen_fn_call, so it silently skipped the
    same await-detection gen_fn_call does for async stdlib functions
    (fetch_url/wait/webhook), internal step calls, and MCP/REST tool
    methods. `url |> fetch_url` compiled to a bare `fetch_url(url)` — an
    unawaited coroutine object as the "result", not the fetched content.
    No error, no warning, the real call never fires.
    """

    def test_async_stdlib_stage_is_awaited(self, transpile):
        out = transpile(
            'import { fetch_url } from "drift/io" '
            "agent A { step run(url: string) -> string { "
            "  let result = url |> fetch_url "
            "  return result "
            "} }"
        )
        assert "await fetch_url(url)" in out

    def test_sync_stdlib_stage_is_not_awaited(self, transpile):
        # Negative case: a sync stdlib function must NOT get a spurious
        # await — only the 3 real async stdlib fns should.
        out = transpile(
            'import { redact } from "drift/safety" '
            "agent A { step run(text: string) -> string { "
            "  let result = text |> redact "
            "  return result "
            "} }"
        )
        assert "await redact(text)" not in out
        assert "redact(text)" in out

    def test_mcp_tool_method_stage_is_awaited(self, transpile):
        out = transpile(
            'tool weather from mcp "https://x.com" '
            "agent A { step run(city: string) -> string { "
            "  let result = city |> weather.get_forecast "
            "  return result "
            "} }"
        )
        assert "await weather.get_forecast(city)" in out

    def test_internal_step_stage_is_awaited(self, transpile):
        out = transpile(
            "agent A { "
            "  step run(x: string) -> string { "
            "    let result = x |> helper "
            "    return result "
            "  } "
            "  step helper(x: string) -> string { return x } "
            "}"
        )
        assert "await self.helper(x)" in out

    @pytest.mark.asyncio
    async def test_async_stdlib_stage_actually_runs_not_just_a_coroutine(
        self, transpile, tmp_path, monkeypatch
    ):
        # End-to-end: without the fix, `result` would be an unawaited
        # coroutine object, not the string fetch_url resolves to.
        src = (
            'import { fetch_url } from "drift/io" '
            "agent A { step run(url: string) -> string { "
            "  let result = url |> fetch_url "
            "  return result "
            "} }"
        )
        py = transpile(src).replace("Source: <drift_file>", "Source: inline")
        path = tmp_path / "gen_pipe_await.py"
        path.write_text(py)
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("gen_pipe_await", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["gen_pipe_await"] = mod
        spec.loader.exec_module(mod)

        async def fake_fetch_url(url):
            return f"fetched:{url}"

        monkeypatch.setattr(mod, "fetch_url", fake_fetch_url)

        agent = mod.A()
        result = await agent.run("https://example.com")
        assert result == "fetched:https://example.com"
