"""Runtime unit tests — behavior of router, budget, validation, errors."""
import asyncio
from dataclasses import dataclass

import pytest

from drift.runtime import (
    Agent, Budget, CostTracker, ModelRouter,
    BudgetExceeded, ModelUnavailable, RateLimited, AuthError,
    SchemaViolation, step_decorator, run_agent,
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
    async def test_falls_back_on_model_unavailable(self):
        agent = StubAgent()

        @step_decorator()
        async def step(self):
            self.call_count += 1
            if self.call_count < 2:
                raise ModelUnavailable("m1 down", model="m1")
            return "ok"

        result = await step(agent)
        assert result == "ok"
        # After the first failure, m1 should be marked unavailable
        assert "m1" in agent.model._unavailable

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
    async def test_rate_limit_waits_and_retries(self):
        agent = StubAgent()

        @step_decorator()
        async def step(self):
            self.call_count += 1
            if self.call_count < 2:
                raise RateLimited("slow down", model="m1", retry_after=0.01)
            return "ok"

        result = await step(agent)
        assert result == "ok"
        assert agent.call_count == 2

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
