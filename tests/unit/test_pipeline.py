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
        # gather_or_cancel wraps asyncio.gather with cancel-on-failure
        # cleanup for still-in-flight siblings (drift/runtime/core.py).
        assert "gather_or_cancel" in out

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


class TestPipelineNodeResolution:
    """A bare (undotted) pipeline node — `process`, not `Agent.process` —
    used to resolve via `use_agents[0]`, i.e. whichever agent happened to
    be LISTED FIRST in the `use` clause, regardless of which agent
    actually declares that step. That silently broke as soon as `use`
    order didn't match declaration order, and — worse — if two `use`d
    agents both declared a same-named step, the second agent's step was
    permanently, silently unreachable with no error at all. Fixed by
    resolving bare nodes against the actual declaring agent (first
    DECLARED, not first `use`d, on a tie), and raising CodegenError for
    genuine ambiguity/missing-`use` cases instead of guessing."""

    def test_bare_node_resolves_to_declaring_agent_not_use_order(self, transpile):
        # `process` is declared on A but `use` lists B first — must still
        # resolve to A, the agent that actually owns the step.
        out = transpile(
            "agent A { step process(x: string) -> string { return \"A\" } } "
            "agent B { step other(x: string) -> string { return \"B\" } } "
            "pipeline P { use B use A process -> B.other }"
        )
        assert "self.A.process" in out

    def test_same_named_step_on_two_used_agents_resolves_to_first_declared(
        self, transpile
    ):
        out = transpile(
            "agent A { step process(x: string) -> string { return \"A\" } } "
            "agent B { step process(x: string) -> string { return \"B\" } } "
            "agent C { step other(x: string) -> string { return x } } "
            "pipeline P { use A use B use C C.other -> process }"
        )
        assert "self.A.process" in out

    def test_dotted_node_referencing_a_not_used_agent_is_rejected(self, transpile):
        # This is LLM.md §12's own flagship example shape: agents named in
        # pipeline edges but never `use`d. Used to pass drift check/transpile
        # cleanly and only fail at `drift run` with an AttributeError.
        with pytest.raises(CodegenError, match="use Classifier"):
            transpile(
                "agent Intake { step input_email(x: string) -> string { return x } } "
                "agent Classifier { step tag(x: string) -> string { return \"s\" } } "
                "pipeline Triage { input_email -> Classifier.tag }"
            )

    def test_bare_node_owned_by_no_used_agent_is_rejected(self, transpile):
        with pytest.raises(CodegenError, match="none of its"):
            transpile(
                "agent A { step process(x: string) -> string { return x } } "
                "pipeline P { use A nonexistent -> A.process }"
            )

    def test_ambiguous_failure_handler_across_same_named_steps_is_rejected(
        self, transpile
    ):
        # _read_handler_phrase reads to the next NEWLINE (a plain natural-
        # language phrase, unterminated by any keyword) — the edge must be
        # on its own line or it gets swallowed into the handler text.
        src = (
            "agent A { step process(x: string) -> string { return \"A\" } } "
            "agent B { step process(x: string) -> string { return \"B\" } } "
            "agent C { step other(x: string) -> string { return x } } "
            "pipeline P {\n"
            "  use A use B use C\n"
            "  on failure in process: skip and continue\n"
            "  C.other -> process\n"
            "}"
        )
        with pytest.raises(CodegenError, match="ambiguous"):
            transpile(src)

    def test_unambiguous_failure_handler_still_works(self, transpile):
        # Sanity check: the new ambiguity check must not false-positive
        # when only one used agent owns the named step.
        src = (
            "agent A { step entry(x: string) -> string { return x } "
            "  step process(x: string) -> string { return x } } "
            "pipeline P {\n"
            "  use A\n"
            "  A.entry -> process\n"
            "  on failure in process: skip and continue\n"
            "}"
        )
        out = transpile(src)
        assert "try:" in out


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
    async def test_inline_entry_step_return_does_not_crash(self, transpile, tmp_path):
        # Regression: LLM.md's own documented workaround for "no syntax to
        # start a pipeline with raw data" is to declare a real inline step
        # as the entry node (`step tickets(batch: T) -> T { return batch }`).
        # gen_return unconditionally emitted `self.checkpoint.save(...)`,
        # but inline pipeline steps are plain methods on the pipeline class
        # (gen_inline_step) — not Agent subclasses — so they have no
        # `self.checkpoint`, crashing with AttributeError on the exact
        # pattern the docs tell users to write.
        src = (
            "agent Triager { step classify(x: string) -> string { return \"urgent\" } } "
            "pipeline P { "
            "  use Triager "
            "  step tickets(batch: list<string>) -> list<string> { return batch } "
            "  tickets -> Triager.classify "
            "}"
        )
        py = transpile(src).replace("Source: <drift_file>", "Source: inline")
        path = tmp_path / "gen_inline_entry.py"
        path.write_text(py)
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("gen_inline_entry", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["gen_inline_entry"] = mod
        spec.loader.exec_module(mod)
        pipe = mod.P()
        result = await pipe.run(["a", "b"])
        assert result == "urgent"

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
    async def test_fanout_item_coerced_to_schema_dataclass(self, transpile, tmp_path):
        # Regression: a `=>` fan-out target with a schema-typed parameter
        # used to receive the raw JSON-decoded dict/list item verbatim
        # (--input is never mapped by keyword for pipelines — it's passed
        # as the entry node's single positional value), so attribute
        # access on the item crashed with AttributeError. coerce_arg()
        # must run before each fan-out call, using the target's own type
        # hint, exactly the way a --pipeline --input '[{"ticket_id": ...}]'
        # run needs it to.
        src = (
            "schema Ticket { ticket_id: string ticket_text: string } "
            "agent Intake { step tickets(batch: list<Ticket>) -> list<Ticket> "
            "  { return batch } } "
            "agent Router { step route(ticket: Ticket) -> string "
            "  { return ticket.ticket_id } } "
            "pipeline P { use Intake use Router tickets => Router.route }"
        )
        py = transpile(src).replace("Source: <drift_file>", "Source: inline")
        path = tmp_path / "gen_coerce.py"
        path.write_text(py)
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("gen_coerce", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["gen_coerce"] = mod
        spec.loader.exec_module(mod)
        pipe = mod.P()
        result = await pipe.run(initial_input=[
            {"ticket_id": "T-1", "ticket_text": "x"},
            {"ticket_id": "T-2", "ticket_text": "y"},
        ])
        assert result == ["T-1", "T-2"]

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
