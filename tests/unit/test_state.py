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
