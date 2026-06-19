"""Tests for §6.2 `define verb` — custom intent verbs."""
import pytest

from drift import ast_nodes as ast
from drift.runtime import CUSTOM_VERBS


@pytest.fixture(autouse=True)
def clean_verb_registry():
    """Each test starts with an empty custom-verb registry."""
    CUSTOM_VERBS.clear()
    yield
    CUSTOM_VERBS.clear()


class TestVerbParse:
    def test_basic_decl(self, parse_ast):
        p = parse_ast(
            'define verb score_fit { '
            '  pattern: "score_fit <x>" '
            '  prompt: "You are an expert." '
            '  output: FitScore '
            '  temperature: 0.2 '
            '}'
        )
        v = p.declarations[0]
        assert isinstance(v, ast.VerbDecl)
        assert v.name == "score_fit"
        assert v.pattern == "score_fit <x>"
        assert v.prompt == "You are an expert."
        assert v.output.name == "FitScore"
        assert v.temperature == 0.2

    def test_only_prompt_required(self, parse_ast):
        # No pattern, no output, no temperature — should still parse.
        p = parse_ast('define verb foo { prompt: "do thing" }')
        v = p.declarations[0]
        assert v.name == "foo"
        assert v.prompt == "do thing"
        assert v.output is None

    def test_unknown_field_raises(self, parse_ast):
        from drift.parser import ParseError
        with pytest.raises(ParseError):
            parse_ast('define verb foo { weather: "sunny" }')


class TestCallSiteParsing:
    """A custom verb declared at the top must be usable as an intent verb below."""

    def test_custom_verb_callable_in_step(self, parse_ast):
        p = parse_ast(
            'define verb my_classify { prompt: "p" }\n'
            'agent A { step f(x: string) { let r = my_classify x as string } }'
        )
        agent = p.declarations[1]
        intent = agent.steps[0].body[0].value
        assert isinstance(intent, ast.IntentExpr)
        assert intent.verb == "my_classify"

    def test_custom_verb_before_declaration_fails(self, parse_ast):
        # A verb used before its `define verb` decl is just a plain ident.
        # The parser doesn't pre-scan; it treats the call as something else
        # (here it'll try to parse as expression and fail or produce garbage).
        # We assert the verb is NOT recognized.
        p = parse_ast(
            'agent A { step f(x: string) { my_classify(x) } }\n'
            'define verb my_classify { prompt: "p" }'
        )
        # Without forward-decl scanning, my_classify(x) is parsed as a fn call.
        agent = p.declarations[0]
        body = agent.steps[0].body
        assert isinstance(body[0], ast.ExprStmt)
        # Either way, it's NOT an IntentExpr at this point.
        assert not isinstance(body[0].expr, ast.IntentExpr)


class TestCodegen:
    def test_register_call_emitted(self, transpile):
        out = transpile(
            'define verb my_summarize { '
            '  prompt: "Summarize briefly." '
            '  output: string '
            '  temperature: 0.1 '
            '}'
        )
        assert "register_custom_verb(" in out
        assert 'name="my_summarize"' in out
        assert "'Summarize briefly.'" in out
        assert "temperature=0.1" in out


class TestRuntimeRegistration:
    def test_register_stores_in_registry(self):
        from drift.runtime import register_custom_verb
        register_custom_verb(
            name="quack",
            prompt="You are a duck.",
            output_schema=str,
            pattern="quack <x>",
            temperature=0.5,
        )
        assert "quack" in CUSTOM_VERBS
        assert CUSTOM_VERBS["quack"]["prompt"] == "You are a duck."
        assert CUSTOM_VERBS["quack"]["temperature"] == 0.5

    def test_custom_prompt_used_in_build(self):
        from drift.runtime import register_custom_verb
        from drift.runtime.core import build_intent_prompt
        register_custom_verb(name="duckspeak", prompt="QUACK ONLY.")
        system, _ = build_intent_prompt("duckspeak", "hi")
        assert system == "QUACK ONLY."

    @pytest.mark.asyncio
    async def test_default_output_schema_inherited(self):
        from dataclasses import dataclass
        from drift.runtime import register_custom_verb, Agent, ModelRouter, Budget

        @dataclass
        class Tag:
            label: str
            score: float

        register_custom_verb(
            name="tag", prompt="Tag the input.", output_schema=Tag,
        )
        a = Agent(name="T", model=ModelRouter(), budget=Budget())
        # Call without an explicit output_schema; custom registry should fill in.
        result = await a.intent(verb="tag", input_data="hello")
        # Mock provider returns mock fields for the dataclass.
        assert isinstance(result, Tag)
