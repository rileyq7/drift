"""Tests for the deja_vu keyword — parse + codegen.

Runtime semantics (does the archive trigger actually fire?) are tested by
the two-run demo against real Dendric. Here we just verify the language
layer: the .drift syntax parses, the AST shape is right, and the
generated Python calls memory.deja_vu_check + dispatches via
match.matches(pattern)."""
from drift import ast_nodes as ast


# ── Parse ──────────────────────────────────────────────────────────────


class TestDejaVuParse:
    def test_basic_match_on(self, parse_ast):
        d = parse_ast(
            'agent A { memory: dendric("g") '
            'step f(x: string) { '
            '  deja_vu match on x { '
            '    "similar_rejection" -> { respond "warn" } '
            '  } '
            '} }'
        ).declarations[0]
        stmt = d.steps[0].body[0]
        assert isinstance(stmt, ast.DejaVuStmt)
        assert len(stmt.arms) == 1
        assert stmt.arms[0].pattern == "similar_rejection"

    def test_multiple_arms(self, parse_ast):
        d = parse_ast(
            'agent A { memory: dendric("g") '
            'step f(x: string) { '
            '  deja_vu match on x { '
            '    "a" -> { respond "A" } '
            '    "b" -> { respond "B" } '
            '    "c" -> { respond "C" } '
            '  } '
            '} }'
        ).declarations[0]
        arms = d.steps[0].body[0].arms
        assert [a.pattern for a in arms] == ["a", "b", "c"]
        assert not any(a.is_default for a in arms)

    def test_any_other_default_arm(self, parse_ast):
        d = parse_ast(
            'agent A { memory: dendric("g") '
            'step f(x: string) { '
            '  deja_vu match on x { '
            '    "known" -> { respond "k" } '
            '    any other -> { respond "novel" } '
            '  } '
            '} }'
        ).declarations[0]
        arms = d.steps[0].body[0].arms
        assert len(arms) == 2
        assert arms[0].pattern == "known" and not arms[0].is_default
        assert arms[1].is_default is True

    def test_empty_arm_body_allowed(self, parse_ast):
        """Empty arm bodies should parse — the codegen emits a `pass`."""
        d = parse_ast(
            'agent A { memory: dendric("g") '
            'step f(x: string) { '
            '  deja_vu match on x { "x" -> { } } '
            '} }'
        ).declarations[0]
        assert d.steps[0].body[0].arms[0].body == []


# ── Codegen ────────────────────────────────────────────────────────────


class TestDejaVuCodegen:
    def test_emits_deja_vu_check(self, transpile):
        py = transpile(
            'agent A { memory: dendric("g") '
            'step f(x: string) { '
            '  deja_vu match on x { '
            '    "p" -> { respond "hit" } '
            '  } '
            '} }'
        )
        assert "self.memory.deja_vu_check(context=x)" in py
        # The arm dispatches via match.matches(pattern)
        assert "match.matches('p')" in py

    def test_default_arm_emits_else(self, transpile):
        py = transpile(
            'agent A { memory: dendric("g") '
            'step f(x: string) { '
            '  deja_vu match on x { '
            '    "p" -> { respond "p" } '
            '    any other -> { respond "novel" } '
            '  } '
            '} }'
        )
        assert "if match.matches('p'):" in py
        assert "else:" in py

    def test_only_named_arms_no_else(self, transpile):
        py = transpile(
            'agent A { memory: dendric("g") '
            'step f(x: string) { '
            '  deja_vu match on x { '
            '    "a" -> { respond "A" } '
            '    "b" -> { respond "B" } '
            '  } '
            '} }'
        )
        # if/elif chain without trailing else
        assert "if match.matches('a'):" in py
        assert "elif match.matches('b'):" in py

    def test_sequential_blocks_get_unique_temp_vars(self, transpile):
        """Two deja_vu blocks in one step must not shadow each other's
        intermediate variable — that's why the codegen uses a counter."""
        py = transpile(
            'agent A { memory: dendric("g") '
            'step f(x: string) { '
            '  deja_vu match on x { "a" -> { respond "a" } } '
            '  deja_vu match on x { "b" -> { respond "b" } } '
            '} }'
        )
        assert "_dv_1" in py and "_dv_2" in py
