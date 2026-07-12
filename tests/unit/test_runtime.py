"""Runtime unit tests — behavior of router, budget, validation, errors."""
import asyncio
from dataclasses import dataclass

import pytest

from drift.runtime import (
    Agent, Budget, CostTracker, ModelRouter,
    BudgetExceeded, ModelUnavailable, RateLimited, AuthError,
    SchemaViolation, StepFailed, step_decorator, run_agent, first_declared,
    gather_or_cancel,
)
from drift.runtime.core import parse_llm_response, MockProvider


# ─── ModelRouter ────────────────────────────────────────────────────

class TestModelRouter:
    def test_default_only(self):
        r = ModelRouter(default="claude-sonnet")
        assert r.select() == "claude-sonnet"

    def test_prefer_overrides_default(self):
        r = ModelRouter(default="claude-haiku", prefer="claude-sonnet")
        assert r.select() == "claude-sonnet"

    def test_fallback_used_when_prefer_unavailable(self):
        r = ModelRouter(default="claude-sonnet", fallback=["gpt-4o"])
        r.mark_unavailable("claude-sonnet")
        assert r.select() == "gpt-4o"

    def test_select_raises_when_all_exhausted(self):
        r = ModelRouter(default="claude-sonnet", fallback=["gpt-4o"])
        r.mark_unavailable("claude-sonnet")
        r.mark_unavailable("gpt-4o")
        with pytest.raises(ModelUnavailable):
            r.select()

    def test_reset_clears_unavailability(self):
        r = ModelRouter(default="claude-sonnet", fallback=["gpt-4o"])
        r.mark_unavailable("claude-sonnet")
        r.reset_availability()
        assert r.select() == "claude-sonnet"

    def test_never_list_blocks_model(self):
        r = ModelRouter(default="gpt-3.5-turbo", fallback=["claude-sonnet"], never=["gpt-3.5-turbo"])
        assert r.select() == "claude-sonnet"

    def test_mark_unavailable_with_none_is_safe(self):
        # The decorator can call this with model=None if the exception didn't
        # carry one — it should not crash.
        r = ModelRouter(default="claude-sonnet")
        r.mark_unavailable(None)
        assert r.select() == "claude-sonnet"

    def test_candidates_deduplicates(self):
        # If default == prefer, only one entry in candidates
        r = ModelRouter(default="claude-sonnet", prefer="claude-sonnet", fallback=["claude-sonnet"])
        assert r.candidates() == ["claude-sonnet"]

    def test_tight_budget_picks_cheaper(self):
        r = ModelRouter(default="claude-opus", fallback=["claude-haiku"])
        # remaining < 0.10 → cheapest first
        assert r.select(budget_remaining=0.05) == "claude-haiku"

    def test_api_model_id_translates_logical_name(self):
        r = ModelRouter()
        assert r.api_model_id("claude-opus").startswith("claude-opus")
        # Unknown name passes through
        assert r.api_model_id("custom-model") == "custom-model"


# ─── Budget / CostTracker ───────────────────────────────────────────

class TestBudget:
    def test_pre_check_passes_under_budget(self):
        ct = CostTracker(Budget(max_per_run=1.0))
        ct.pre_check(estimated_cost=0.5)  # fine

    def test_pre_check_raises_over_budget(self):
        ct = CostTracker(Budget(max_per_run=1.0))
        ct.record(0.95, "claude-sonnet", 1000, 1000)
        with pytest.raises(BudgetExceeded):
            ct.pre_check(estimated_cost=0.1)

    def test_remaining_is_clamped_at_zero(self):
        ct = CostTracker(Budget(max_per_run=1.0))
        ct.record(2.0, "model", 0, 0)  # overshoot
        assert ct.remaining == 0

    def test_call_log_records_entries(self):
        ct = CostTracker(Budget())
        ct.record(0.1, "m", 100, 50)
        ct.record(0.2, "m", 200, 100)
        assert len(ct.call_log) == 2
        assert ct.total_cost == pytest.approx(0.3)

    def test_currency_symbol_lookup(self):
        assert Budget(currency="GBP").symbol == "£"
        assert Budget(currency="USD").symbol == "$"
        assert Budget(currency="EUR").symbol == "€"
        # Unknown currency defaults
        assert Budget(currency="JPY").symbol == "$"


class TestCostReservation:
    def test_reservations_count_against_budget(self):
        # Two concurrent reservations that together exceed the cap: the second
        # must be rejected even though nothing has been *spent* yet. This is
        # what stops parallel fan-out from overspending.
        ct = CostTracker(Budget(max_per_run=0.10))
        ct.reserve(0.06)
        with pytest.raises(BudgetExceeded):
            ct.reserve(0.06)

    def test_release_frees_reservation(self):
        ct = CostTracker(Budget(max_per_run=0.10))
        r = ct.reserve(0.08)
        ct.release(r)
        # After release the budget is free again.
        assert ct.reserve(0.08) == 0.08

    def test_settle_replaces_reservation_with_actual(self):
        ct = CostTracker(Budget(max_per_run=1.0))
        r = ct.reserve(0.50)          # worst-case hold
        ct.settle(r, 0.05, "m", 10, 5)  # actual came in much lower
        assert ct.total_cost == pytest.approx(0.05)
        assert ct.reserved == pytest.approx(0.0)
        # Freed reservation is available again.
        assert ct.remaining == pytest.approx(0.95)

    def test_concurrent_reservations_never_exceed_budget(self):
        # Simulate N parallel tasks each reserving before "awaiting": the number
        # that succeed must never let committed spend exceed the cap.
        ct = CostTracker(Budget(max_per_run=0.10))
        succeeded = 0
        for _ in range(50):
            try:
                ct.reserve(0.008)
                succeeded += 1
            except BudgetExceeded:
                pass
        assert succeeded * 0.008 <= 0.10 + 1e-9
        assert ct.total_cost + ct.reserved <= 0.10 + 1e-9


# ─── parse_llm_response ─────────────────────────────────────────────

@dataclass
class Result:
    name: str
    score: float

    def validate(self):
        assert 0 <= self.score <= 100, f"score out of range: {self.score}"


class TestBuildIntentPromptClauses:
    """`with` was a parsed clause keyword that codegen never read at all —
    the value was silently dropped before ever reaching self.intent() or
    build_intent_prompt. Also: every clause is documented to accept
    comma-separated lists, not just `considering` — verify the prompt
    text actually reflects multiple values, not just the first."""

    def test_with_clause_appears_in_prompt(self):
        from drift.runtime.core import build_intent_prompt
        _system, prompt = build_intent_prompt(
            "summarize", "doc text", with_="formatting notes",
        )
        assert "formatting notes" in prompt

    def test_multi_value_context_clause_includes_all_values(self):
        from drift.runtime.core import build_intent_prompt
        _system, prompt = build_intent_prompt(
            "generate", "a reply", context=["ctx one", "ctx two"],
        )
        assert "ctx one" in prompt
        assert "ctx two" in prompt

    def test_multi_value_target_clause_includes_all_values(self):
        from drift.runtime.core import build_intent_prompt
        _system, prompt = build_intent_prompt(
            "translate", "doc", target=["French", "Spanish"],
        )
        assert "French" in prompt
        assert "Spanish" in prompt


class TestParseLLMResponse:
    def test_plain_string_passthrough(self):
        assert parse_llm_response("hello", output_schema=str) == "hello"
        assert parse_llm_response("  hello  ", output_schema=None) == "hello"

    def test_json_to_dataclass(self):
        out = parse_llm_response('{"name": "x", "score": 50}', output_schema=Result)
        assert isinstance(out, Result)
        assert out.name == "x"

    def test_markdown_fence_stripped(self):
        out = parse_llm_response(
            '```json\n{"name": "x", "score": 50}\n```',
            output_schema=Result,
        )
        assert isinstance(out, Result)

    def test_unknown_fields_dropped(self):
        out = parse_llm_response(
            '{"name": "x", "score": 50, "extra": "ignored"}',
            output_schema=Result,
        )
        assert out.name == "x"

    def test_invalid_json_raises_schema_violation(self):
        with pytest.raises(SchemaViolation):
            parse_llm_response("not json at all", output_schema=Result)

    def test_validate_failure_raises_schema_violation(self):
        # score=999 is out of range; .validate() raises AssertionError; parse
        # should wrap that as a SchemaViolation so the decorator can retry.
        with pytest.raises(SchemaViolation):
            parse_llm_response(
                '{"name": "x", "score": 999}',
                output_schema=Result,
            )

    def test_list_when_object_expected_raises(self):
        with pytest.raises(SchemaViolation):
            parse_llm_response("[1, 2, 3]", output_schema=Result)


# ─── Mock Provider ──────────────────────────────────────────────────

class TestMockProvider:
    @pytest.mark.asyncio
    async def test_mock_returns_valid_literal(self):
        from typing import Literal as L

        @dataclass
        class HasLiteral:
            kind: L["a", "b", "c"]

        p = MockProvider()
        text, _, _ = await p.call("any-model", "sys", "prompt", output_schema=HasLiteral)
        out = parse_llm_response(text, output_schema=HasLiteral)
        assert out.kind in ("a", "b", "c")


# ─── step_decorator retry behavior ──────────────────────────────────

class StubAgent:
    """Minimal agent for testing the decorator without the full Agent init."""
    def __init__(self):
        self.model = ModelRouter(default="m1", fallback=["m2"])
        self.cost_tracker = CostTracker(Budget())
        self.call_count = 0


class TestStepDecoratorRetry:
    @pytest.mark.asyncio
    async def test_retries_on_schema_violation(self):
        agent = StubAgent()

        @step_decorator()
        async def step(self):
            self.call_count += 1
            if self.call_count < 2:
                raise SchemaViolation("bad json")
            return "ok"

        result = await step(agent)
        assert result == "ok"
        assert agent.call_count == 2

    @pytest.mark.asyncio
    async def test_validate_failure_retries_not_crashes(self):
        # A dataclass's validate() (generated from `between`/`one of`
        # constraints) raising SchemaViolation must trigger the same retry
        # path as a malformed-JSON SchemaViolation — not propagate as an
        # unhandled crash. This is the runtime half of the codegen fix that
        # switched constraint checks from bare `assert` (AssertionError,
        # uncaught here) to `raise SchemaViolation(...)`.
        agent = StubAgent()

        @dataclass
        class Scored:
            score: float
            def validate(self):
                if not (0.0 <= self.score <= 100.0):
                    raise SchemaViolation(f"score out of range: {self.score}")
                return self

        @step_decorator(output=Scored)
        async def step(self):
            self.call_count += 1
            # First call: LLM "hallucinated" an out-of-range score.
            # Second call: valid.
            return Scored(score=150.0) if self.call_count < 2 else Scored(score=50.0)

        result = await step(agent)
        assert result.score == 50.0
        assert agent.call_count == 2

    @pytest.mark.asyncio
    async def test_model_unavailable_escaping_the_step_is_wrapped_once_not_retried(self):
        # ModelUnavailable/RateLimited retry now happens INSIDE intent()
        # itself, scoped to the individual failing call — see
        # TestIntentRetryScopedToCall below for that behavior. A step
        # function is only called ONCE by step_decorator; if one of these
        # escapes all the way out of the step body (intent() already
        # exhausted its own retries), the step wraps it in StepFailed on
        # the first occurrence rather than re-running the whole step body.
        agent = StubAgent()

        @step_decorator()
        async def step(self):
            self.call_count += 1
            raise ModelUnavailable("m1 down", model="m1")

        with pytest.raises(StepFailed):
            await step(agent)
        assert agent.call_count == 1

    @pytest.mark.asyncio
    async def test_auth_error_does_not_retry(self):
        agent = StubAgent()

        @step_decorator()
        async def step(self):
            self.call_count += 1
            raise AuthError("bad key")

        with pytest.raises(AuthError):
            await step(agent)
        assert agent.call_count == 1  # one shot, no retry

    @pytest.mark.asyncio
    async def test_budget_exceeded_does_not_retry(self):
        agent = StubAgent()

        @step_decorator()
        async def step(self):
            self.call_count += 1
            raise BudgetExceeded("over")

        with pytest.raises(BudgetExceeded):
            await step(agent)
        assert agent.call_count == 1

    @pytest.mark.asyncio
    async def test_rate_limited_escaping_the_step_is_wrapped_once_not_retried(self):
        # Same reasoning as test_model_unavailable_escaping_the_step_is_
        # wrapped_once_not_retried above — RateLimited's retry-and-wait
        # loop now lives inside intent(), not step_decorator.
        agent = StubAgent()

        @step_decorator()
        async def step(self):
            self.call_count += 1
            raise RateLimited("slow down", model="m1", retry_after=0.01)

        with pytest.raises(StepFailed):
            await step(agent)
        assert agent.call_count == 1

    @pytest.mark.asyncio
    async def test_availability_resets_at_step_start(self):
        agent = StubAgent()
        agent.model.mark_unavailable("m1")  # poisoned from a previous step

        @step_decorator()
        async def step(self):
            self.call_count += 1
            return "ok"

        await step(agent)
        # The decorator should have reset before invoking the body
        assert "m1" not in agent.model._unavailable


# ─── intent() retry — scoped to the individual call, not the step ────

class TestIntentRetryScopedToCall:
    """The core fix: SchemaViolation/ModelUnavailable/RateLimited retry
    now happens INSIDE intent(), around just the single failing LLM call
    — not in step_decorator, wrapping the entire step body. Previously, a
    step with two intent calls where only the SECOND failed re-ran the
    WHOLE step from the top on each retry, re-invoking (and, against a
    real provider, re-billing) the FIRST call up to max_retries times
    even though it had already succeeded and never needed retrying.
    """

    @pytest.mark.asyncio
    async def test_earlier_successful_intent_call_is_not_repeated_when_a_later_one_retries(
        self, transpile, tmp_path
    ):
        src = (
            'agent A { model: "claude-haiku" '
            '  step two_calls() -> string { '
            '    let first = summarize "a" as string '
            '    let second = classify "b" as string '
            '    return second '
            '  } '
            '}'
        )
        py = transpile(src).replace("Source: <drift_file>", "Source: inline")
        path = tmp_path / "gen_no_rebill.py"
        path.write_text(py)
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("gen_no_rebill", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["gen_no_rebill"] = mod
        spec.loader.exec_module(mod)

        agent = mod.A()
        call_log = []

        class FlakyProvider:
            async def call(self, model, system, prompt, output_schema=None,
                            temperature=None):
                call_log.append(prompt)
                # The SECOND intent call (classify "b") fails its first
                # attempt, then succeeds. The FIRST call (summarize "a")
                # must never be re-invoked because of that failure.
                if "b" in prompt and call_log.count(prompt) == 1:
                    raise SchemaViolation("bad json")
                return ('ok', 10, 5)

        agent._provider_for = lambda model_name: FlakyProvider()

        result = await agent.two_calls()
        assert result == "ok"
        # "summarize a" appears exactly once — it succeeded on its only
        # attempt and must not have been re-run when "classify b" retried.
        summarize_calls = [p for p in call_log if "a" in p and "b" not in p]
        assert len(summarize_calls) == 1, (
            f"expected summarize('a') to run exactly once, got {len(summarize_calls)} "
            f"— the earlier successful call was silently re-invoked when a "
            f"LATER call in the same step retried"
        )

    @pytest.mark.asyncio
    async def test_schema_violation_retries_within_intent_then_succeeds(
        self, transpile, tmp_path
    ):
        src = (
            'agent A { model: "claude-haiku" '
            '  step f() -> string { '
            '    let x = classify "x" as string '
            '    return x '
            '  } '
            '}'
        )
        py = transpile(src).replace("Source: <drift_file>", "Source: inline")
        path = tmp_path / "gen_intent_retry.py"
        path.write_text(py)
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("gen_intent_retry", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["gen_intent_retry"] = mod
        spec.loader.exec_module(mod)

        agent = mod.A()
        calls = {"n": 0}

        class FlakyProvider:
            async def call(self, model, system, prompt, output_schema=None,
                            temperature=None):
                calls["n"] += 1
                if calls["n"] < 3:
                    raise SchemaViolation("bad json")
                return ('ok', 10, 5)

        agent._provider_for = lambda model_name: FlakyProvider()

        result = await agent.f()
        assert result == "ok"
        assert calls["n"] == 3

    @pytest.mark.asyncio
    async def test_schema_violation_exhausts_retries_and_raises_raw_type(
        self, transpile, tmp_path
    ):
        # Terminal exhaustion re-raises the RAW SchemaViolation (not
        # StepFailed) from intent() itself — attempt/recover blocks match
        # the raw type directly (see gen_attempt), so wrapping it here
        # would silently break every attempt/recover keyed on it. A step
        # with NO attempt/recover around the call still ends up seeing
        # StepFailed — but from step_decorator's own wrap one layer up,
        # not from intent().
        src = (
            'agent A { model: "claude-haiku" '
            '  step f() -> string { '
            '    let x = classify "x" as string '
            '    return x '
            '  } '
            '}'
        )
        py = transpile(src).replace("Source: <drift_file>", "Source: inline")
        path = tmp_path / "gen_intent_exhaust.py"
        path.write_text(py)
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("gen_intent_exhaust", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["gen_intent_exhaust"] = mod
        spec.loader.exec_module(mod)

        agent = mod.A()

        class AlwaysFailProvider:
            async def call(self, model, system, prompt, output_schema=None,
                            temperature=None):
                raise SchemaViolation("always bad")

        agent._provider_for = lambda model_name: AlwaysFailProvider()

        # intent() raises the raw type on exhaustion...
        with pytest.raises(SchemaViolation):
            await agent.intent(verb="classify", input_data="x", output_schema=str)

        # ...but the step (with no attempt/recover) sees StepFailed, from
        # step_decorator's own terminal wrap.
        with pytest.raises(StepFailed):
            await agent.f()


# ─── Exception attributes ───────────────────────────────────────────

class TestExceptionShape:
    def test_model_unavailable_carries_model_name(self):
        e = ModelUnavailable("x", model="haiku")
        assert e.model == "haiku"

    def test_rate_limited_carries_retry_after(self):
        e = RateLimited("x", model="haiku", retry_after=5.0)
        assert e.retry_after == 5.0

    def test_auth_error_is_not_modelunavailable(self):
        # The taxonomy should keep these distinct
        assert not issubclass(AuthError, ModelUnavailable)
        assert not issubclass(ModelUnavailable, AuthError)


# ─── Entry-step selection ───────────────────────────────────────────

class TestEntryStepSelection:
    @pytest.mark.asyncio
    async def test_runs_first_declared_step_not_alphabetical(self):
        # `archive` sorts before `triage` alphabetically, but `triage` is
        # declared first and must be the default entry point.
        calls = []

        class Ordered(Agent):
            def __init__(self):
                super().__init__(name="Ordered", budget=Budget(max_per_run=1.0))

            @step_decorator()
            async def triage(self):
                calls.append("triage")
                return "triage"

            @step_decorator()
            async def archive(self):
                calls.append("archive")
                return "archive"

        result = await run_agent(Ordered)
        assert result == "triage"
        assert calls == ["triage"]


class TestFirstDeclared:
    """first_declared() — used by drift run (no --agent) and MCP's
    drift_run to recover source declaration order for AGENT selection,
    the class-level analogue of TestEntryStepSelection's step-level fix
    above. Both callers used to build their candidate dict via dir(module)
    (alphabetical), so `next(iter(...))`/`list(...)[0]` silently picked
    whichever agent's class NAME sorted first, not the one declared
    first in the .drift source."""

    def test_returns_first_declared_not_alphabetically_first(self):
        class Zeta(Agent):
            def __init__(self):
                super().__init__(name="Zeta")

        class Alpha(Agent):
            def __init__(self):
                super().__init__(name="Alpha")

        # Zeta is declared (defined) first in this file, even though
        # "Alpha" sorts first alphabetically.
        assert first_declared([Alpha, Zeta]) is Zeta
        assert first_declared([Zeta, Alpha]) is Zeta

    def test_single_class(self):
        class Solo(Agent):
            def __init__(self):
                super().__init__(name="Solo")

        assert first_declared([Solo]) is Solo

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            first_declared([])


# ─── --input JSON coercion into schema-typed step parameters ─────────

class TestInputCoercion:
    """run_agent's inputs (parsed JSON from CLI --input / MCP drift_run
    input) used to be spread as **kwargs with no type coercion — a
    schema-typed step parameter arrived as a bare dict, and `param.field`
    crashed with AttributeError even though this is the documented way to
    pass structured input (LLM.md: "--input takes a JSON object mapped to
    the step's parameters by name")."""

    @pytest.mark.asyncio
    async def test_dict_coerced_to_dataclass_param(self):
        @dataclass
        class Item:
            name: str
            qty: int

        class A(Agent):
            def __init__(self):
                super().__init__(name="A", budget=Budget(max_per_run=1.0))

            @step_decorator()
            async def f(self, item: Item):
                return item.name

        result = await run_agent(A, inputs={"item": {"name": "widget", "qty": 5}})
        assert result == "widget"

    @pytest.mark.asyncio
    async def test_bare_list_param_passes_through_unchanged(self):
        # `list` (no generic) parameter has no element type hint to coerce
        # against — coercion should safely no-op (dicts stay dicts) rather
        # than crash on the ambiguity.
        class A(Agent):
            def __init__(self):
                super().__init__(name="A", budget=Budget(max_per_run=1.0))

            @step_decorator()
            async def f(self, tickets: list):
                return [t["ticket_id"] for t in tickets]

        result = await run_agent(A, inputs={"tickets": [{"ticket_id": "T-1", "text": "x"}]})
        assert result == ["T-1"]

    @pytest.mark.asyncio
    async def test_typed_list_of_dataclass_param(self):
        @dataclass
        class Ticket:
            ticket_id: str
            text: str

        class A(Agent):
            def __init__(self):
                super().__init__(name="A", budget=Budget(max_per_run=1.0))

            @step_decorator()
            async def f(self, tickets: list[Ticket]):
                return [t.ticket_id for t in tickets]

        result = await run_agent(A, inputs={
            "tickets": [{"ticket_id": "T-1", "text": "x"}, {"ticket_id": "T-2", "text": "y"}]
        })
        assert result == ["T-1", "T-2"]

    @pytest.mark.asyncio
    async def test_primitive_values_pass_through_unchanged(self):
        class A(Agent):
            def __init__(self):
                super().__init__(name="A", budget=Budget(max_per_run=1.0))

            @step_decorator()
            async def f(self, name: str, count: int):
                return f"{name}:{count}"

        result = await run_agent(A, inputs={"name": "Riley", "count": 3})
        assert result == "Riley:3"

    @pytest.mark.asyncio
    async def test_already_correct_dataclass_instance_passes_through(self):
        # Internal callers (cross-agent calls, pipeline nodes with a
        # non-JSON-sourced value) may already pass a real instance —
        # coercion must not double-wrap or break that.
        @dataclass
        class Item:
            name: str

        class A(Agent):
            def __init__(self):
                super().__init__(name="A", budget=Budget(max_per_run=1.0))

            @step_decorator()
            async def f(self, item: Item):
                return item.name

        result = await run_agent(A, inputs={"item": Item(name="widget")})
        assert result == "widget"


# ─── Cost reporting on run_agent (success and failure) ───────────────

class TestRunAgentCostOut:
    """run_agent's `cost_out` param and the `_drift_cost` exception tag —
    a run can spend real budget before failing, and callers that can't read
    the printed stdout summary (the MCP server) need that spend as
    structured data instead of losing it."""

    @pytest.mark.asyncio
    async def test_cost_out_filled_on_success(self):
        class Spends(Agent):
            def __init__(self):
                super().__init__(name="Spends", budget=Budget(max_per_run=1.0))

            @step_decorator()
            async def go(self):
                self.cost_tracker.record(0.25, "mock-model", 10, 5)
                return "done"

        cost = {}
        result = await run_agent(Spends, cost_out=cost)
        assert result == "done"
        assert cost["total_cost"] == 0.25
        assert cost["budget"] == 1.0
        assert len(cost["calls"]) == 1
        assert cost["calls"][0]["model"] == "mock-model"

    @pytest.mark.asyncio
    async def test_cost_out_includes_respond_outputs(self):
        # cost_out also carries `outputs` (agent._outputs, the respond-
        # statement lines) — callers that can't rely on a human reading
        # stdout (e.g. the MCP server) need this as structured data too,
        # not just the numeric cost fields.
        class Narrates(Agent):
            def __init__(self):
                super().__init__(name="Narrates", budget=Budget(max_per_run=1.0))

            @step_decorator()
            async def go(self):
                self.output("step one")
                self.output("step two")
                return "done"

        cost = {}
        await run_agent(Narrates, cost_out=cost)
        assert cost["outputs"] == ["step one", "step two"]

    @pytest.mark.asyncio
    async def test_cost_out_filled_and_exception_tagged_on_budget_exceeded(self):
        class Overspends(Agent):
            def __init__(self):
                super().__init__(name="Overspends", budget=Budget(max_per_run=0.10))

            @step_decorator()
            async def go(self):
                self.cost_tracker.record(0.05, "mock-model", 10, 5)
                self.cost_tracker.reserve(1.0)  # exceeds the 0.10 cap
                return "unreachable"

        cost = {}
        with pytest.raises(BudgetExceeded) as exc_info:
            await run_agent(Overspends, cost_out=cost)

        # Spend that happened before the failure isn't lost.
        assert cost["total_cost"] == 0.05
        assert len(cost["calls"]) == 1
        assert exc_info.value._drift_cost == cost

    @pytest.mark.asyncio
    async def test_cost_out_filled_on_step_failed(self):
        class Fails(Agent):
            def __init__(self):
                super().__init__(name="Fails", budget=Budget(max_per_run=1.0))

            @step_decorator()
            async def go(self):
                self.cost_tracker.record(0.02, "mock-model", 10, 5)
                raise StepFailed("business logic gave up")

        cost = {}
        with pytest.raises(StepFailed) as exc_info:
            await run_agent(Fails, cost_out=cost)

        assert cost["total_cost"] == 0.02
        assert exc_info.value._drift_cost["total_cost"] == 0.02


# ─── gather_or_cancel — orphaned-task cleanup on partial failure ──────

class TestGatherOrCancel:
    """Regression: `for each ... parallel` and a pipeline's `=>` fan-out
    both compiled to a bare `asyncio.gather(*coros)` (no
    return_exceptions=True). gather() raises as soon as the FIRST task
    fails but does NOT cancel the others — they keep running in the
    background. Each holds a live CostTracker reservation (from
    Agent.intent's reserve-before-await) until it naturally finishes;
    until then, cost_tracker.reserved stays inflated, understating
    remaining budget for cost-aware routing and any subsequent reserve()
    check — for however long the stragglers take, or permanently if
    nothing ever awaits them again. gather_or_cancel preserves the exact
    same "first failure propagates, whole batch lost" semantics but
    cancels and awaits the still-pending siblings first.
    """

    @pytest.mark.asyncio
    async def test_preserves_first_failure_semantics(self):
        # Same observable outcome as bare asyncio.gather: the first
        # failure propagates, full stop.
        async def fails():
            raise ValueError("boom")

        async def succeeds():
            return "ok"

        with pytest.raises(ValueError):
            await gather_or_cancel(fails(), succeeds())

    @pytest.mark.asyncio
    async def test_still_pending_siblings_are_cancelled_and_cleaned_up(self):
        # A slow sibling's cleanup (its own try/finally) must have
        # actually run by the time gather_or_cancel raises — not still be
        # pending in the background.
        cleanup_ran = {"n": False}

        async def fails_fast():
            await asyncio.sleep(0.01)
            raise ValueError("boom")

        async def slow_but_cancellable():
            try:
                await asyncio.sleep(10)
                return "should never get here"
            finally:
                cleanup_ran["n"] = True

        with pytest.raises(ValueError):
            await gather_or_cancel(fails_fast(), slow_but_cancellable())

        # If the sibling were left running in the background (the old
        # bare-gather behavior), this would still be False immediately
        # after gather_or_cancel raises.
        assert cleanup_ran["n"] is True

    @pytest.mark.asyncio
    async def test_budget_reservation_released_promptly_on_partial_failure(
        self, transpile, tmp_path
    ):
        # End-to-end proof through real generated `for each ... parallel`
        # code: one item fails after a real intent() reservation is made;
        # a sibling item is still mid-flight (slow) when that happens.
        # cost_tracker.reserved must be back at 0 immediately after the
        # exception propagates out of the step — not stuck holding the
        # slow sibling's reservation.
        src = (
            'agent A { model: "claude-haiku" '
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
        py = transpile(src).replace("Source: <drift_file>", "Source: inline")
        path = tmp_path / "gen_orphan_cleanup.py"
        path.write_text(py)
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("gen_orphan_cleanup", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["gen_orphan_cleanup"] = mod
        spec.loader.exec_module(mod)

        agent = mod.A()

        class SlowThenFailProvider:
            async def call(self, model, system, prompt, output_schema=None,
                            temperature=None):
                if "bad" in prompt:
                    await asyncio.sleep(0.01)
                    raise ValueError("simulated LLM failure")
                # The "good" item is slower than the failing one, so it's
                # still in flight (and still holding its reservation)
                # when "bad" raises.
                await asyncio.sleep(0.2)
                return ('"ok"', 10, 5)

        agent._provider_for = lambda model_name: SlowThenFailProvider()

        with pytest.raises(ValueError):
            await agent.process(["good", "bad"])

        # The reservation from the still-in-flight "good" call must have
        # been released by gather_or_cancel's cleanup, not left stuck.
        assert agent.cost_tracker.reserved == pytest.approx(0.0)
