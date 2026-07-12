"""Tests for §6.1 `match` keyword: intent verb vs. pattern-match statement.

The parser disambiguates by looking ahead: if `against` appears before
the next LBRACE/newline/EOF, it's the intent expression; otherwise it's
the pattern-matching statement.
"""
import pytest

from drift import ast_nodes as ast


def parse(source, parse_ast):
    return parse_ast(source).declarations[0]


class TestMatchStatementForm:
    def test_simple_match_statement(self, parse_ast):
        d = parse(
            'agent A { step f(x: string) { '
            '  match x { "a" -> respond "alpha" any other -> respond "?" } '
            '} }',
            parse_ast,
        )
        body = d.steps[0].body
        assert isinstance(body[0], ast.MatchStmt)
        assert len(body[0].arms) == 2

    def test_match_on_field_access(self, parse_ast):
        d = parse(
            'agent A { step f(x: string) { '
            '  match result.priority { '
            '    "urgent" -> respond "now" '
            '    any other -> respond "later" '
            '  } '
            '} }',
            parse_ast,
        )
        body = d.steps[0].body
        assert isinstance(body[0], ast.MatchStmt)

    def test_underscore_catchall_is_recognized_as_default(self, parse_ast):
        # Regression: LLM.md's OWN documented `match` example uses `_` as
        # the catch-all (`_ -> { <statements> }`), not `any other` — but
        # the parser only ever checked for `any`/`any other`. A bare `_`
        # fell through to being parsed as an ordinary Ident pattern
        # (compared for equality against an undefined Python name `_` at
        # codegen), crashing with NameError on every match that reached
        # the catch-all arm. No prior test used `_` at all — every
        # existing match test used `any other` instead.
        d = parse(
            'agent A { step f(x: string) { '
            '  match x { "a" -> respond "alpha" _ -> respond "?" } '
            '} }',
            parse_ast,
        )
        body = d.steps[0].body
        assert isinstance(body[0], ast.MatchStmt)
        assert len(body[0].arms) == 2
        assert body[0].arms[1].is_default is True


class TestMatchCodegen:
    def test_default_arm_first_emits_valid_python(self, transpile):
        # A default arm written before pattern arms must still produce valid
        # Python (default emitted last as `else:`), not `else:` before `if`.
        import ast as py_ast
        out = transpile(
            'agent A { step f(x: string) { '
            '  match x { any other -> { respond "def" } '
            '            "a" -> { respond "a" } } '
            '} }'
        )
        py_ast.parse(out)  # must not raise
        assert "if x == " in out
        assert "else:" in out
        # the `if` must come before the `else` in the emitted source
        assert out.index("if x ==") < out.index("else:")

    def test_match_with_only_default_arm(self, transpile):
        import ast as py_ast
        out = transpile(
            'agent A { step f(x: string) { '
            '  match x { any other -> { respond "always" } } '
            '} }'
        )
        py_ast.parse(out)
        assert "else:" not in out  # nothing to attach else to

    def test_underscore_catchall_emits_else_not_equality_check(self, transpile):
        # Must NOT emit `elif x == _:` (undefined-name NameError bait) —
        # `_` has to become a plain `else:`, same as `any other` does.
        out = transpile(
            'agent A { step f(x: string) -> string { '
            '  match x { "a" -> { return "alpha" } '
            '             _ -> { return "other" } } '
            '} }'
        )
        assert "== _" not in out
        assert "else:" in out

    @pytest.mark.asyncio
    async def test_underscore_catchall_executes_without_nameerror(
        self, transpile, tmp_path
    ):
        # End-to-end proof: reaching the `_` arm must not crash. Before
        # the fix, this raised NameError on every input that fell through
        # to the catch-all — i.e. every input except the exact literal
        # pattern matches.
        src = (
            'agent A { step f(x: string) -> string { '
            '  match x { '
            '    "a" -> { return "matched-a" } '
            '    _ -> { return "fell-through" } '
            '  } '
            '} }'
        )
        py = transpile(src).replace("Source: <drift_file>", "Source: inline")
        path = tmp_path / "gen_underscore.py"
        path.write_text(py)
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("gen_underscore", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["gen_underscore"] = mod
        spec.loader.exec_module(mod)
        agent = mod.A()
        assert await agent.f(x="a") == "matched-a"
        assert await agent.f(x="anything else") == "fell-through"


class TestMatchIntentForm:
    def test_intent_with_against_and_as(self, parse_ast):
        d = parse(
            'agent A { step f(x: string) { '
            '  let r = match x against criteria as Result '
            '} }',
            parse_ast,
        )
        intent = d.steps[0].body[0].value
        assert isinstance(intent, ast.IntentExpr)
        assert intent.verb == "match"
        assert "against" in intent.clauses

    def test_intent_at_statement_position(self, parse_ast):
        # `match X against Y as T` as a top-level statement (no `let`)
        d = parse(
            'agent A { step f(x: string) { '
            '  match x against criteria as Result '
            '} }',
            parse_ast,
        )
        stmt = d.steps[0].body[0]
        assert isinstance(stmt, ast.ExprStmt)
        assert isinstance(stmt.expr, ast.IntentExpr)
        assert stmt.expr.verb == "match"

    def test_intent_followed_by_statement_match(self, parse_ast):
        # Both forms in the same step, back to back.
        d = parse(
            'agent A { step f(x: string) { '
            '  let r = match x against criteria as Result '
            '  match r.recommendation { any other -> respond "ok" } '
            '} }',
            parse_ast,
        )
        body = d.steps[0].body
        assert len(body) == 2
        assert isinstance(body[0], ast.LetStmt)
        assert isinstance(body[0].value, ast.IntentExpr)
        assert isinstance(body[1], ast.MatchStmt)


class TestEdgeCases:
    def test_match_with_brace_before_against_is_statement(self, parse_ast):
        # The brace comes first, so this is a statement even though `against`
        # appears later in a string.
        d = parse(
            'agent A { step f(x: string) { '
            '  match x { "against" -> respond "?" any other -> respond "!" } '
            '} }',
            parse_ast,
        )
        assert isinstance(d.steps[0].body[0], ast.MatchStmt)
