"""Tests for the Confident<T> type — §3.2 of the spec."""
import asyncio

import pytest

from drift.runtime import Confident, SchemaViolation
from drift.runtime.core import parse_llm_response, MockProvider


class TestConfidentClass:
    def test_basic_construction(self):
        c = Confident("hello", 0.9)
        assert c.value == "hello"
        assert c.confidence == 0.9

    def test_is_confident_at_threshold(self):
        c = Confident("x", 0.85)
        assert c.is_confident(0.85) is True
        assert c.is_confident(0.86) is False

    def test_clamp_above_one(self):
        # An LLM might return "95" thinking that's 95%
        c = Confident("x", 95)
        assert c.confidence == 0.95

    def test_clamp_above_one_hundred(self):
        c = Confident("x", 200)
        assert c.confidence == 1.0

    def test_clamp_negative(self):
        c = Confident("x", -0.5)
        assert c.confidence == 0.0

    def test_invalid_confidence_coerces_to_zero(self):
        c = Confident("x", "not a number")
        assert c.confidence == 0.0

    def test_class_getitem_returns_tagged_subclass(self):
        # `Confident[T]` returns a tagged subclass carrying the inner type.
        # isinstance checks still work (subclass of Confident), but the runtime
        # can recover the inner type for schema-aware prompts.
        tagged = Confident[str]
        assert tagged is not Confident
        assert issubclass(tagged, Confident)
        assert tagged._inner_type is str


class TestConfidentParseResponse:
    def test_well_formed_wrapper(self):
        out = parse_llm_response(
            '{"value": "spam", "confidence": 0.92}',
            output_schema=Confident,
        )
        assert isinstance(out, Confident)
        assert out.value == "spam"
        assert out.confidence == 0.92

    def test_missing_value_raises(self):
        with pytest.raises(SchemaViolation):
            parse_llm_response('{"confidence": 0.9}', output_schema=Confident)

    def test_missing_confidence_raises(self):
        with pytest.raises(SchemaViolation):
            parse_llm_response('{"value": "x"}', output_schema=Confident)

    def test_non_object_raises(self):
        with pytest.raises(SchemaViolation):
            parse_llm_response('"just a string"', output_schema=Confident)


class TestConfidentMockProvider:
    @pytest.mark.asyncio
    async def test_mock_returns_valid_confident(self):
        p = MockProvider()
        text, _, _ = await p.call("m", "sys", "prompt", output_schema=Confident)
        out = parse_llm_response(text, output_schema=Confident)
        assert isinstance(out, Confident)
        assert 0 <= out.confidence <= 1


class TestConfidentCodegen:
    def test_confident_field_type(self, transpile):
        out = transpile("schema X { r: confident<string> }")
        assert "r: Confident" in out

    def test_is_confident_compiles(self, transpile):
        out = transpile(
            'agent A { step f(x: string) { '
            '  let result = classify x as confident<string> '
            '  if result is confident { respond "ok" } '
            '} }'
        )
        assert "result.is_confident(self.min_confidence)" in out
        assert "output_schema=Confident" in out

    def test_is_uncertain_compiles(self, transpile):
        out = transpile(
            'agent A { step f(x: string) { '
            '  let result = classify x as confident<string> '
            '  if result is uncertain { respond "flag" } '
            '} }'
        )
        assert "(not result.is_confident(self.min_confidence))" in out

    def test_confident_value_access(self, transpile):
        out = transpile(
            'agent A { step f(x: string) -> string { '
            '  let result = classify x as confident<string> '
            '  return result.value '
            '} }'
        )
        assert "result.value" in out


class TestConfidentDemoEndToEnd:
    @pytest.mark.asyncio
    async def test_confident_demo_runs(self, examples_dir):
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location(
            "drift_confident_demo", examples_dir / "confident_demo.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["drift_confident_demo"] = mod
        spec.loader.exec_module(mod)
        agent = mod.Triager()
        # Mock returns confidence 0.88; agent threshold is 0.85.
        # 0.88 >= 0.85 → confident branch fires.
        result = await agent.triage(item="hello")
        assert result == "mock_value"
        # One classify call
        assert len(agent.cost_tracker.call_log) == 1

    @pytest.mark.asyncio
    async def test_confident_demo_uncertain_branch(self, examples_dir):
        # Raise the threshold above the mock's 0.88 to force the uncertain branch.
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location(
            "drift_confident_demo2", examples_dir / "confident_demo.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["drift_confident_demo2"] = mod
        spec.loader.exec_module(mod)
        agent = mod.Triager()
        agent.min_confidence = 0.99
        result = await agent.triage(item="hello")
        assert result == "needs_review"


class TestConfidentDataclassRoundTrip:
    """Regression test for the codegen bug a cold-start subagent caught:

    Codegen used to emit bare `Confident` for `confident<MySchema>`, so the
    runtime never told the LLM the inner schema. scored.value came back as a
    raw string and `.field_access` blew up at runtime.

    These tests pin down the full pipeline:
      drift source → codegen emits `Confident[InnerType]`
                  → runtime prompt includes inner schema
                  → parse_llm_response builds a real dataclass for `value`
                  → field access on `scored.value` works
    """

    def test_codegen_emits_inner_type(self, transpile):
        # Was: `output_schema=Confident`. Now must include the inner schema.
        out = transpile(
            "schema TriageResult {\n"
            "  priority: string\n"
            "  summary: string\n"
            "}\n"
            "agent A {\n"
            "  step f(x: string) -> TriageResult {\n"
            "    let scored = classify x as confident<TriageResult>\n"
            "    if scored is confident { return scored.value }\n"
            "    fail \"uncertain\"\n"
            "  }\n"
            "}\n"
        )
        assert "output_schema=Confident[TriageResult]" in out, (
            "Codegen regressed to bare `Confident` and lost the inner type. "
            "This breaks every confident<MySchema> program at runtime."
        )

    def test_prompt_includes_inner_schema(self):
        # The LLM must see the inner schema in its prompt, not just
        # `{value, confidence}`. Otherwise it returns a string for value.
        from dataclasses import dataclass
        from drift.runtime.core import build_intent_prompt

        @dataclass
        class TriageResult:
            priority: str
            summary: str

        _system, prompt = build_intent_prompt(
            "classify", "ticket text",
            output_schema=Confident[TriageResult],
        )
        assert "priority" in prompt
        assert "summary" in prompt
        assert '"value"' in prompt
        assert '"confidence"' in prompt

    def test_parse_builds_real_dataclass(self):
        # The smoking gun. Pre-fix this returned `value` as the raw dict (or
        # raw string when the LLM hadn't been told the schema). Post-fix,
        # `value` is a real TriageResult instance with field access.
        from dataclasses import dataclass

        @dataclass
        class TriageResult:
            priority: str
            summary: str

        wrapped = parse_llm_response(
            '{"value": {"priority": "urgent", "summary": "card charged twice"}, "confidence": 0.93}',
            output_schema=Confident[TriageResult],
        )
        assert isinstance(wrapped, Confident)
        assert wrapped.confidence == 0.93
        # The critical assertion: scored.value is a dataclass, not a dict/str.
        assert isinstance(wrapped.value, TriageResult)
        assert wrapped.value.priority == "urgent"
        assert wrapped.value.summary == "card charged twice"

    def test_parse_ignores_extra_fields_in_value(self):
        from dataclasses import dataclass

        @dataclass
        class Small:
            a: str

        wrapped = parse_llm_response(
            '{"value": {"a": "x", "b": "extra"}, "confidence": 0.9}',
            output_schema=Confident[Small],
        )
        assert isinstance(wrapped.value, Small)
        assert wrapped.value.a == "x"

    def test_parse_bare_confident_still_works(self):
        # Backward compat: `confident<>` with no inner type behaves as before.
        wrapped = parse_llm_response(
            '{"value": "hello", "confidence": 0.9}',
            output_schema=Confident,
        )
        assert isinstance(wrapped, Confident)
        assert wrapped.value == "hello"

    @pytest.mark.asyncio
    async def test_mock_provider_returns_schema_shaped_value(self):
        # The mock must mirror the real shape: when Confident[Schema] is the
        # output schema, its mock value dict must match Schema's fields, so
        # parse_llm_response can round-trip it into a real dataclass.
        from dataclasses import dataclass

        @dataclass
        class Result:
            label: str
            score: float

        p = MockProvider()
        text, _, _ = await p.call("m", "sys", "prompt", output_schema=Confident[Result])
        wrapped = parse_llm_response(text, output_schema=Confident[Result])
        assert isinstance(wrapped.value, Result)
        assert isinstance(wrapped.value.label, str)
        assert isinstance(wrapped.value.score, float)
