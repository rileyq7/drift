"""Tests for §9 memory block + recall/remember statements."""
import tempfile
from pathlib import Path

import pytest

from drift import ast_nodes as ast
from drift.runtime import MemoryStore


class TestMemoryConfigParse:
    def test_basic_block(self, parse_ast):
        d = parse_ast(
            'agent A { memory { '
            '  store: "sqlite://./m.db" '
            '  recall strategy: "semantic" '
            '  max recall: 10 items '
            '  decay: enabled '
            '} step f() { respond "x" } }'
        ).declarations[0]
        m = d.memory_config
        assert m.store == "sqlite://./m.db"
        assert m.recall_strategy == "semantic"
        assert m.max_recall == 10
        assert m.decay_enabled is True

    def test_decay_disabled(self, parse_ast):
        d = parse_ast(
            'agent A { memory { decay: disabled } '
            'step f() { respond "x" } }'
        ).declarations[0]
        assert d.memory_config.decay_enabled is False

    def test_defaults(self, parse_ast):
        d = parse_ast(
            'agent A { memory { store: "sqlite://:memory:" } '
            'step f() { respond "x" } }'
        ).declarations[0]
        assert d.memory_config.recall_strategy == "recent"
        assert d.memory_config.max_recall == 20


class TestRecallRememberParse:
    def test_recall_with_for_key(self, parse_ast):
        d = parse_ast(
            'agent A { memory { } '
            'step f(topic: string) { '
            '  let past = recall similar items for topic '
            '} }'
        ).declarations[0]
        let_stmt = d.steps[0].body[0]
        assert isinstance(let_stmt.value, ast.RecallStmt)
        assert "similar" in let_stmt.value.description
        # The `for topic` parses to an Ident
        assert let_stmt.value.key.name == "topic"

    def test_remember_tagged(self, parse_ast):
        d = parse_ast(
            'agent A { memory { } '
            'step f(x: string) { remember x tagged "note" } }'
        ).declarations[0]
        stmt = d.steps[0].body[0]
        assert isinstance(stmt, ast.RememberStmt)
        assert stmt.tag.value == "note"


class TestCodegen:
    def test_memory_store_in_init(self, transpile):
        out = transpile(
            'agent A { memory { '
            '  store: "sqlite://./m.db" '
            '  recall strategy: "relevant" '
            '} step f() { respond "x" } }'
        )
        assert "MemoryStore(store_url='sqlite://./m.db'" in out
        assert "recall_strategy='relevant'" in out

    def test_recall_compiles_to_method_call(self, transpile):
        out = transpile(
            'agent A { memory { } '
            'step f(t: string) { '
            '  let past = recall similar notes for t '
            '} }'
        )
        assert "self.memory.recall(" in out
        assert "key=t" in out

    def test_remember_compiles_to_method_call(self, transpile):
        out = transpile(
            'agent A { memory { } '
            'step f(x: string) { remember x tagged "k" } }'
        )
        assert 'self.memory.remember(x, tag="k")' in out


class TestMemoryStoreRuntime:
    def test_remember_and_recall_round_trip(self):
        store = MemoryStore(store_url="sqlite://:memory:", recall_strategy="recent")
        store.remember("first")
        store.remember("second")
        store.remember("third")
        recalled = store.recall()
        # Most recent first
        assert recalled == ["third", "second", "first"]

    def test_relevant_strategy_filters_by_tag(self):
        store = MemoryStore(store_url="sqlite://:memory:", recall_strategy="relevant")
        store.remember("python tip", tag="python")
        store.remember("rust tip", tag="rust")
        store.remember("python tip 2", tag="python")
        out = store.recall(key="python")
        assert "python tip" in out
        assert "python tip 2" in out
        assert "rust tip" not in out

    def test_max_recall_limits_results(self):
        store = MemoryStore(store_url="sqlite://:memory:", max_recall=2)
        for i in range(5):
            store.remember(f"item {i}")
        out = store.recall()
        assert len(out) == 2

    def test_dataclass_round_trip(self):
        from dataclasses import dataclass
        @dataclass
        class Tag:
            name: str
            score: float

        store = MemoryStore(store_url="sqlite://:memory:")
        store.remember(Tag(name="a", score=0.5), tag="x")
        store.remember(Tag(name="b", score=0.9), tag="x")
        out = store.recall()
        # Stored dataclasses come back as plain field dicts (type info is lost,
        # but the serialization envelope is unwrapped — recall is symmetric
        # with remember).
        assert out[0]["name"] == "b"
        assert out[0]["score"] == 0.9

    def test_file_backed_store_persists(self, tmp_path):
        path = tmp_path / "mem.db"
        s1 = MemoryStore(store_url=f"sqlite://{path}")
        s1.remember("persistent", tag="key")
        s1.close()
        s2 = MemoryStore(store_url=f"sqlite://{path}")
        assert "persistent" in s2.recall()
        s2.close()

    def test_semantic_falls_back_to_relevant(self, capsys):
        store = MemoryStore(store_url="sqlite://:memory:", recall_strategy="semantic")
        store.remember("py tip", tag="python")
        out = store.recall(key="python")
        assert "py tip" in out
        # And we should have warned about semantic mode
        captured = capsys.readouterr()
        assert "semantic" in captured.out.lower()

    def test_unsupported_url_scheme_fails(self):
        with pytest.raises(ValueError):
            MemoryStore(store_url="redis://localhost")


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_step_uses_recall_remember(self, transpile, tmp_path):
        src = (
            'agent A { memory { } '
            '  step note(topic: string, body: string) -> string { '
            '    remember body tagged topic '
            '    return body '
            '  } '
            '}'
        )
        py = transpile(src).replace("Source: <drift_file>", "Source: inline")
        path = tmp_path / "gen.py"
        path.write_text(py)
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("gen_mem", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["gen_mem"] = mod
        spec.loader.exec_module(mod)
        agent = mod.A()
        await agent.note(topic="python", body="lists are mutable")
        await agent.note(topic="python", body="dicts are ordered now")
        recalled = agent.memory.recall(key="python")
        assert any("mutable" in r for r in recalled)
        assert any("ordered" in r for r in recalled)
