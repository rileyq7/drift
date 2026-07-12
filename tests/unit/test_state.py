"""Tests for §9 state blocks — agent-instance state that persists across steps."""
import pytest

from drift import ast_nodes as ast


class TestStateParse:
    def test_empty_state_block(self, parse_ast):
        d = parse_ast(
            'agent A { state { } step f() { respond "x" } }'
        ).declarations[0]
        assert d.state_block == []

    def test_field_with_default(self, parse_ast):
        d = parse_ast(
            'agent A { state { count: number = 0 } step f() { respond "x" } }'
        ).declarations[0]
        f = d.state_block[0]
        assert f.name == "count"
        assert f.type_expr.name == "number"
        assert isinstance(f.default, ast.NumberLit)
        assert f.default.value == 0

    def test_list_field(self, parse_ast):
        d = parse_ast(
            'agent A { state { items: list<string> = [] } '
            'step f() { respond "x" } }'
        ).declarations[0]
        f = d.state_block[0]
        assert isinstance(f.type_expr, ast.ListType)
        assert isinstance(f.default, ast.ListLit)

    def test_field_without_default(self, parse_ast):
        d = parse_ast(
            'agent A { state { label: string } step f() { respond "x" } }'
        ).declarations[0]
        f = d.state_block[0]
        assert f.default is None

    def test_multiple_fields(self, parse_ast):
        d = parse_ast(
            'agent A { state { '
            '  a: string = "x" '
            '  b: number = 5 '
            '  c: bool = true '
            '} step f() { respond "x" } }'
        ).declarations[0]
        assert len(d.state_block) == 3


class TestStateCodegen:
    BASE = 'agent A { state { %s } step f() { respond "x" } }'

    def test_field_with_default_emitted(self, transpile):
        out = transpile(self.BASE % "count: number = 0")
        assert "self.count = 0" in out

    def test_field_without_default_gets_zero_value(self, transpile):
        out = transpile(self.BASE % "label: string")
        assert 'self.label = ""' in out

    def test_list_field_default_empty_list(self, transpile):
        out = transpile(self.BASE % "items: list<string>")
        assert "self.items = []" in out

    def test_state_comment_marker(self, transpile):
        out = transpile(self.BASE % "count: number = 0")
        assert "# Agent state" in out

    def test_bare_state_field_read_resolves_to_self(self, transpile):
        # Regression: a bare reference to a state field name inside a step
        # body used to emit the bare Python name (`count`), not `self.count`
        # — Python's scoping rules make ANY name assigned anywhere in a
        # function local to that function for its ENTIRE body, so even a
        # pure read like `return count` would raise UnboundLocalError the
        # instant the same function also did `let count = ...` anywhere
        # (extremely likely, since mutating state IS the point of state).
        out = transpile(
            'agent A { state { count: int = 0 } '
            'step f() -> int { return count } }'
        )
        assert "_result = self.count" in out

    def test_let_assignment_to_state_field_targets_self(self, transpile):
        # `let count = count + 1` must become `self.count = (self.count +
        # 1)` — mutating the persisted attribute — not a fresh local that
        # shadows it and crashes reading its own uninitialized value.
        out = transpile(
            'agent A { state { count: int = 0 } '
            'step f() -> int { let count = count + 1 return count } }'
        )
        assert "self.count = (self.count + 1)" in out
        assert "_result = self.count" in out

    def test_local_variable_with_different_name_is_unaffected(self, transpile):
        # Only names that are actually declared state fields get rewritten
        # — an ordinary local variable must stay a plain local.
        out = transpile(
            'agent A { state { count: int = 0 } '
            'step f() -> string { let label = "hi" return label } }'
        )
        assert "label = " in out
        assert "self.label" not in out

    def test_state_field_in_string_interpolation_resolves_to_self(self, transpile):
        # respond/string interpolation embeds the interpolation body as
        # raw source text, a separate code path from gen_let/gen_expr's
        # Ident case — needs its own fix (_rewrite_state_refs) to avoid
        # referencing an undefined bare name inside the f-string.
        out = transpile(
            'agent A { state { count: int = 0 } '
            'step f() { respond "count is {count}" } }'
        )
        assert 'f"count is {self.count}"' in out


class TestStateEndToEnd:
    @pytest.mark.asyncio
    async def test_state_persists_across_step_calls(self, transpile, tmp_path):
        src = (
            'agent A { '
            '  state { count: number = 0 } '
            '  step bump() -> number { '
            '    let c = self.count '
            '    return c '
            '  } '
            '}'
        )
        py = transpile(src).replace("Source: <drift_file>", "Source: inline")
        path = tmp_path / "gen.py"
        path.write_text(py)
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("gen_state", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["gen_state"] = mod
        spec.loader.exec_module(mod)

        agent = mod.A()
        assert agent.count == 0
        agent.count = 5  # caller mutates state
        result = await agent.bump()
        assert result == 5

    @pytest.mark.asyncio
    async def test_state_mutated_from_pure_drift_source_persists(self, transpile, tmp_path):
        # End-to-end proof that state is actually usable from Drift syntax
        # alone (not just settable by a Python caller reaching into
        # agent.count directly, which test_state_persists_across_step_calls
        # above already covered as a pre-existing capability). Before this
        # fix, `let count = count + 1` inside a step body crashed with
        # UnboundLocalError — there was no way to write .drift source that
        # both read and mutated state.
        src = (
            'agent A { '
            '  state { count: int = 0 } '
            '  step bump() -> int { '
            '    let count = count + 1 '
            '    return count '
            '  } '
            '}'
        )
        py = transpile(src).replace("Source: <drift_file>", "Source: inline")
        path = tmp_path / "gen_state_mutate.py"
        path.write_text(py)
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("gen_state_mutate", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["gen_state_mutate"] = mod
        spec.loader.exec_module(mod)

        agent = mod.A()
        assert await agent.bump() == 1
        assert await agent.bump() == 2
        assert await agent.bump() == 3
