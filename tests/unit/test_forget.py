"""Tests for the forget keyword — parse + codegen for three variants:
  - forget memories tagged "x"          → store.forget(tag=...)
  - forget memories older than 90d      → store.forget(older_than_days=...)
  - forget memories where temp < 0.3    → store.forget(below_temp=...)

The adapter side (query-then-delete for tag/age vs native by-temp) is
covered in test_dendric_integration.py."""
from drift import ast_nodes as ast


# ── Parse ──────────────────────────────────────────────────────────────


class TestForgetParse:
    def test_by_tag_literal(self, parse_ast):
        d = parse_ast(
            'agent A { memory: dendric("g") '
            'step f() { forget memories tagged "deprecated" } }'
        ).declarations[0]
        stmt = d.steps[0].body[0]
        assert isinstance(stmt, ast.ForgetStmt)
        assert stmt.mode == "by_tag"

    def test_by_age_duration(self, parse_ast):
        d = parse_ast(
            'agent A { memory: dendric("g") '
            'step f() { forget memories older than 90d } }'
        ).declarations[0]
        stmt = d.steps[0].body[0]
        assert stmt.mode == "by_age"
        assert stmt.older_than_days == 90

    def test_by_age_hours_round_down(self, parse_ast):
        """1 hour is less than a day; rounds to 0 days. (The agent should
        probably not write this, but we shouldn't crash on it.)"""
        d = parse_ast(
            'agent A { memory: dendric("g") '
            'step f() { forget memories older than 1h } }'
        ).declarations[0]
        stmt = d.steps[0].body[0]
        assert stmt.mode == "by_age"
        assert stmt.older_than_days == 0

    def test_by_temp_lt(self, parse_ast):
        d = parse_ast(
            'agent A { memory: dendric("g") '
            'step f() { forget memories where temp < 0.3 } }'
        ).declarations[0]
        stmt = d.steps[0].body[0]
        assert stmt.mode == "by_temp"
        assert stmt.below_temp == 0.3

    def test_by_temp_lte(self, parse_ast):
        """<= should also work — both clip the cold tail."""
        d = parse_ast(
            'agent A { memory: dendric("g") '
            'step f() { forget memories where temp <= 0.05 } }'
        ).declarations[0]
        stmt = d.steps[0].body[0]
        assert stmt.mode == "by_temp"
        assert stmt.below_temp == 0.05


# ── Codegen ────────────────────────────────────────────────────────────


class TestForgetCodegen:
    def test_by_tag_emits_tag_kwarg(self, transpile):
        py = transpile(
            'agent A { memory: dendric("g") '
            'step f() { forget memories tagged "old" } }'
        )
        assert 'self.memory.forget(tag="old")' in py

    def test_by_age_emits_older_than_days_kwarg(self, transpile):
        py = transpile(
            'agent A { memory: dendric("g") '
            'step f() { forget memories older than 30d } }'
        )
        assert "self.memory.forget(older_than_days=30)" in py

    def test_by_temp_emits_below_temp_kwarg(self, transpile):
        py = transpile(
            'agent A { memory: dendric("g") '
            'step f() { forget memories where temp < 0.1 } }'
        )
        assert "self.memory.forget(below_temp=0.1)" in py
