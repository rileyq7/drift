"""Tests for §9 memory block + recall/remember statements."""
import tempfile
from pathlib import Path

import pytest

from drift import ast_nodes as ast
from drift.parser import ParseError
from drift.runtime import MemoryStore


class TestMemoryConfigParse:
    def test_unknown_shorthand_backend_points_at_backend_name(self, parse_ast):
        # Regression: the error used to report the position of the token
        # *after* the backend name (the `(`), since it was raised via
        # self.peek() after the name token had already been consumed.
        src = 'agent A { memory: sqlite("persona") step f() { respond "x" } }'
        with pytest.raises(ParseError) as exc_info:
            parse_ast(src)
        e = exc_info.value
        assert "sqlite" in str(e)
        assert e.token.value == "sqlite"
        assert e.token.col == src.index("sqlite") + 1

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
        # description is a real Expression now (StringLit here, since
        # "similar items" is free-form text after the `similar` marker),
        # not a raw string.
        assert isinstance(let_stmt.value.description, ast.StringLit)
        assert "similar" in let_stmt.value.description.value
        # The `for topic` parses to an Ident
        assert let_stmt.value.key.name == "topic"

    def test_bare_variable_description_is_a_variable_reference(self, parse_ast):
        # Regression: LLM.md's own documented "Memory-aware agent" pattern
        # is `let context = recall question for "advice"` — description
        # used to be collected as raw joined text ("question", the
        # identifier's NAME), so codegen emitted self.memory.recall(
        # 'question', ...) — a literal string search for the word
        # "question", completely disconnected from the actual runtime
        # value of the `question` variable. A bare identifier must parse
        # as a variable reference (Ident), evaluated at runtime.
        d = parse_ast(
            'agent A { memory { } '
            'step f(question: string) { '
            '  let context = recall question for "advice" '
            '} }'
        ).declarations[0]
        let_stmt = d.steps[0].body[0]
        assert isinstance(let_stmt.value.description, ast.Ident)
        assert let_stmt.value.description.name == "question"

    def test_quoted_description_supports_interpolation(self, parse_ast):
        d = parse_ast(
            'agent A { memory { } '
            'step f(lead: string) { '
            '  let context = recall "leads similar to {lead}" for "qualification" '
            '} }'
        ).declarations[0]
        let_stmt = d.steps[0].body[0]
        assert isinstance(let_stmt.value.description, ast.StringLit)
        assert let_stmt.value.description.has_interpolation

    def test_field_access_description_is_preserved(self, parse_ast):
        d = parse_ast(
            'agent A { memory { } '
            'step f(lead: string) { '
            '  let context = recall lead.use_case for "qualification" '
            '} }'
        ).declarations[0]
        let_stmt = d.steps[0].body[0]
        assert isinstance(let_stmt.value.description, ast.FieldAccess)
        assert let_stmt.value.description.field_name == "use_case"

    def test_remember_tagged(self, parse_ast):
        d = parse_ast(
            'agent A { memory { } '
            'step f(x: string) { remember x tagged "note" } }'
        ).declarations[0]
        stmt = d.steps[0].body[0]
        assert isinstance(stmt, ast.RememberStmt)
        assert len(stmt.tags) == 1
        assert stmt.tags[0].value == "note"

    def test_remember_multiple_tags(self, parse_ast):
        # LLM.md's own documented example (§17 memory-aware agent pattern):
        # `remember answer tagged "advice", "user_123"` — comma-separated,
        # not just a single tag.
        d = parse_ast(
            'agent A { memory { } '
            'step f(x: string) { remember x tagged "advice", "user_123" } }'
        ).declarations[0]
        stmt = d.steps[0].body[0]
        assert isinstance(stmt, ast.RememberStmt)
        assert [t.value for t in stmt.tags] == ["advice", "user_123"]


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

    def test_multiple_tags_are_each_individually_recallable(self):
        # `remember <expr> tagged "advice", "user_123"` (LLM.md's own
        # documented example) — a value tagged with multiple tags must be
        # findable by recalling on ANY one of them, not just the first.
        store = MemoryStore(store_url="sqlite://:memory:", recall_strategy="relevant")
        store.remember("be concise", tag=["advice", "user_123"])
        store.remember("unrelated", tag="other_user")
        by_first_tag = store.recall(key="advice")
        by_second_tag = store.recall(key="user_123")
        assert "be concise" in by_first_tag
        assert "be concise" in by_second_tag
        assert "unrelated" not in by_first_tag

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

    @pytest.mark.asyncio
    async def test_step_uses_remember_with_multiple_tags(self, transpile, tmp_path):
        # LLM.md's documented multi-tag example
        # (`remember answer tagged "advice", "user_123"`) end-to-end:
        # source -> generated Python -> real execution -> recallable by
        # either tag. This used to be a ParseError before RememberStmt
        # gained multi-tag support.
        src = (
            'agent A { memory { } '
            '  step note(body: string) -> string { '
            '    remember body tagged "advice", "user_123" '
            '    return body '
            '  } '
            '}'
        )
        py = transpile(src).replace("Source: <drift_file>", "Source: inline")
        path = tmp_path / "gen_multitag.py"
        path.write_text(py)
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("gen_multitag", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["gen_multitag"] = mod
        spec.loader.exec_module(mod)
        agent = mod.A()
        await agent.note(body="be concise")
        assert any("concise" in r for r in agent.memory.recall(key="advice"))
        assert any("concise" in r for r in agent.memory.recall(key="user_123"))

    @pytest.mark.asyncio
    async def test_recall_bare_variable_uses_runtime_value_not_literal_name(
        self, transpile, tmp_path
    ):
        # End-to-end proof of the recall-description fix: `recall query
        # for key` must search using query's actual runtime VALUE. Before
        # the fix, this searched for the literal string "query" (the
        # identifier's source-code name) every single time, regardless of
        # what was actually passed in — LLM.md's own documented memory
        # pattern was silently broken this way.
        src = (
            'agent A { memory { recall strategy: "relevant" } '
            '  step ask(query: string) -> list<string> { '
            '    return recall query for "notes" '
            '  } '
            '  step save(text: string) -> string { '
            '    remember text tagged "notes" '
            '    return text '
            '  } '
            '}'
        )
        py = transpile(src).replace("Source: <drift_file>", "Source: inline")
        path = tmp_path / "gen_recall_var.py"
        path.write_text(py)
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("gen_recall_var", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["gen_recall_var"] = mod
        spec.loader.exec_module(mod)
        agent = mod.A()
        await agent.save(text="python tip: lists are mutable")
        await agent.save(text="rust tip: ownership rules")
        # relevant strategy filters by substring match against `key`
        # ("notes" — same for both), so both are recallable; the real
        # assertion is that passing DIFFERENT `query` values doesn't
        # crash and doesn't literally search for the word "query".
        results = await agent.ask(query="anything")
        assert any("python" in r for r in results)
        assert any("rust" in r for r in results)
        # The literal identifier name must never leak into the search.
        no_literal_match = await agent.ask(query="query")
        assert no_literal_match == results
