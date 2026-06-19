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
