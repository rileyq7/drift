"""Tests for §2.5 pipeline declarations."""
import pytest

from drift import ast_nodes as ast
from drift.codegen import CodegenError


class TestPipelineParse:
    def test_minimal_chain(self, parse_ast):
        p = parse_ast("pipeline P { a -> b }")
        d = p.declarations[0]
        assert isinstance(d, ast.PipelineDecl)
        assert d.name == "P"
        assert len(d.edges) == 1
        assert d.edges[0].op == "->"

    def test_chained_arrows_become_multiple_edges(self, parse_ast):
        p = parse_ast("pipeline P { a -> b -> c }")
        edges = p.declarations[0].edges
        assert len(edges) == 2
        assert (edges[0].from_node, edges[0].to_node) == ("a", "b")
        assert (edges[1].from_node, edges[1].to_node) == ("b", "c")

    def test_qualified_node_ref(self, parse_ast):
        p = parse_ast("pipeline P { GrantScout.discover -> FitChecker.evaluate }")
        e = p.declarations[0].edges[0]
        assert e.from_node == "GrantScout.discover"
        assert e.to_node == "FitChecker.evaluate"

    def test_parallel_fanout_op(self, parse_ast):
        p = parse_ast("pipeline P { a => b }")
        assert p.declarations[0].edges[0].op == "=>"

    def test_conditional_op(self, parse_ast):
        p = parse_ast("pipeline P { a ~> b }")
        assert p.declarations[0].edges[0].op == "~>"

    def test_stream_op(self, parse_ast):
        p = parse_ast("pipeline P { a |> b }")
        assert p.declarations[0].edges[0].op == "|>"

    def test_use_agents(self, parse_ast):
        p = parse_ast("pipeline P { use Foo use Bar a -> b }")
        d = p.declarations[0]
        assert d.use_agents == ["Foo", "Bar"]

    def test_budget_and_timeout(self, parse_ast):
        p = parse_ast(
            "pipeline P { "
            "  budget: £20 per run "
            "  timeout: 10m "
            "  a -> b "
            "}"
        )
        d = p.declarations[0]
        assert d.budget_config.value == 20.0
        assert d.timeout_seconds == 600.0

    def test_schedule(self, parse_ast):
        p = parse_ast('pipeline P { schedule: "every Monday at 9am" a -> b }')
        assert p.declarations[0].schedule == "every Monday at 9am"

    def test_failure_handler(self, parse_ast):
        p = parse_ast(
            "pipeline P { a -> b on failure in b: skip and continue }"
        )
        d = p.declarations[0]
        assert "b" in d.failure_handlers
        assert d.failure_handlers["b"].startswith("skip")

    def test_budget_handler(self, parse_ast):
        p = parse_ast(
            "pipeline P { a -> b on budget exceeded: finish current item then stop }"
        )
        d = p.declarations[0]
        assert d.budget_handler.startswith("finish")


class TestPipelineCodegen:
    def test_emits_class(self, transpile):
        out = transpile(
            "agent A { step f(x: string) -> string { return x } } "
            "pipeline P { use A A.f -> A.f }"
        )
        assert "class P:" in out
        assert "async def run(self" in out
        assert "self.A = A()" in out

    def test_fanout_uses_gather(self, transpile):
        out = transpile(
            "agent A { step f(x: string) -> string { return x } "
            "  step g(x: string) -> string { return x } } "
            "pipeline P { use A A.f => A.g }"
        )
        assert "asyncio.gather" in out

    def test_failure_handler_emits_try_except(self, transpile):
        out = transpile(
            "agent A { step f(x: string) -> string { return x } } "
            "pipeline P { use A A.f -> A.f on failure in f: skip and continue }"
        )
        assert "try:" in out
        assert "skipping f" in out

    def test_conditional_edge_is_rejected_at_codegen(self, transpile):
        # `~>` parses (see TestPipelineParse.test_conditional_op) but has no
        # runtime semantics — codegen must refuse to silently run it as `->`.
        with pytest.raises(CodegenError, match="~>"):
            transpile(
                "agent A { step f(x: string) -> string { return x } } "
                "pipeline P { use A A.f ~> A.f }"
            )

    def test_stream_edge_is_rejected_at_codegen(self, transpile):
        with pytest.raises(CodegenError, match=r"\|>"):
            transpile(
                "agent A { step f(x: string) -> string { return x } } "
                "pipeline P { use A A.f |> A.f }"
            )

    def test_schedule_is_rejected_at_codegen(self, transpile):
        # `schedule:` parses (nothing drives it — no daemon/cron loop exists)
        # so codegen must refuse rather than silently ignore it.
        with pytest.raises(CodegenError, match="schedule"):
            transpile(
                "agent A { step f(x: string) -> string { return x } } "
                'pipeline P { schedule: "every Monday at 9am" use A A.f -> A.f }'
            )

    def test_unrecognized_failure_handler_is_rejected_at_codegen(self, transpile):
        # Only a "skip..." prefix is implemented; any other phrase parses
        # but would silently do nothing, so it must be a compile error too.
        with pytest.raises(CodegenError, match="on failure"):
            transpile(
                "agent A { step f(x: string) -> string { return x } } "
                "pipeline P { use A A.f -> A.f on failure in f: retry twice then fail }"
            )

    def test_timeout_emits_wait_for(self, transpile):
        out = transpile(
            "agent A { step f(x: string) -> string { return x } } "
            "pipeline P { timeout: 5s use A A.f -> A.f }"
        )
        assert "asyncio.wait_for(self._orchestrate(initial_input), timeout=5.0)" in out

    def test_budget_handler_emits_except_budget_exceeded(self, transpile):
        out = transpile(
            "agent A { step f(x: string) -> string { return x } } "
            "pipeline P { use A A.f -> A.f "
            "on budget exceeded: finish current item then stop }"
        )
        assert "except BudgetExceeded as _e:" in out


class TestPipelineEndToEnd:
    @pytest.mark.asyncio
    async def test_simple_chain_runs(self, transpile, tmp_path):
        src = (
            "agent Producer { step make() -> string { return \"hello\" } } "
            "agent Consumer { step take(x: string) -> string { return x } } "
            "pipeline P { "
            "  use Producer use Consumer "
            "  Producer.make -> Consumer.take "
            "}"
        )
        py = transpile(src).replace("Source: <drift_file>", "Source: inline")
        path = tmp_path / "gen.py"
        path.write_text(py)
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("gen_pipe", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["gen_pipe"] = mod
        spec.loader.exec_module(mod)
        pipe = mod.P()
        result = await pipe.run()
        # Producer returns "hello" → Consumer.take echoes it back
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_parallel_fanout_runs_per_item(self, transpile, tmp_path):
        src = (
            "agent Gen { step items() -> string { return \"x\" } } "
            "agent Squared { step go(item: string) -> string { return item } } "
            "pipeline P { use Gen use Squared Gen.items => Squared.go }"
        )
        py = transpile(src).replace("Source: <drift_file>", "Source: inline")
        path = tmp_path / "gen.py"
        path.write_text(py)
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("gen_fan", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["gen_fan"] = mod
        spec.loader.exec_module(mod)
        pipe = mod.P()
        # The Gen.items step returns a string "x"; fanout iterates over it
        # char by char (one element: "x"). The point is no crash.
        result = await pipe.run(initial_input=["a", "b", "c"])
        # Verify gather actually returned a list
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_timeout_actually_fires(self, transpile, tmp_path):
        import asyncio
        src = (
            "agent Slow { step wait() -> string { return \"done\" } } "
            "pipeline P { timeout: 1s use Slow Slow.wait -> Slow.wait }"
        )
        py = transpile(src).replace("Source: <drift_file>", "Source: inline")
        path = tmp_path / "gen.py"
        path.write_text(py)
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("gen_timeout", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["gen_timeout"] = mod
        spec.loader.exec_module(mod)
        pipe = mod.P()
        # Patch _orchestrate to hang past the 1s timeout instead of editing
        # generated source — proves run() actually enforces the wait_for.
        async def _hang(initial_input=None):
            await asyncio.sleep(10)
        pipe._orchestrate = _hang
        pipe.timeout_seconds = 0.05
        with pytest.raises(asyncio.TimeoutError):
            await pipe.run()

    @pytest.mark.asyncio
    async def test_budget_handler_logs_and_reraises(self, transpile, tmp_path, capsys):
        from drift.runtime import BudgetExceeded
        src = (
            "agent A { step f() -> string { return \"x\" } } "
            "pipeline P { use A A.f -> A.f "
            "on budget exceeded: finish current item then stop }"
        )
        py = transpile(src).replace("Source: <drift_file>", "Source: inline")
        path = tmp_path / "gen.py"
        path.write_text(py)
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("gen_budget", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["gen_budget"] = mod
        spec.loader.exec_module(mod)
        pipe = mod.P()

        async def _blow_budget(initial_input=None):
            raise BudgetExceeded("over limit")
        pipe._orchestrate = _blow_budget

        with pytest.raises(BudgetExceeded):
            await pipe.run()
        assert "finish current item then stop" in capsys.readouterr().out
