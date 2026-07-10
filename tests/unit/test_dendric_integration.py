"""Tests for the Dendric integration — `memory: dendric("persona")` syntax,
DendricStore adapter (mock path only, so no Postgres/OpenAI needed), and
the mock store's Dendric-compatible surface (deja_vu_check / consolidate /
forget).

Real-Dendric path is exercised by the two-run demo at
examples/grant_checker_memory_demo.py — that needs DATABASE_URL +
OPENAI_API_KEY and is intentionally out-of-scope for unit tests.
"""
from __future__ import annotations

import os

import pytest

from drift import ast_nodes as ast
from drift.runtime import MemoryStore, make_memory_store
from drift.runtime.dendric_store import _serialize, _tag_to_context


# ── Parser: memory: dendric("persona") ─────────────────────────────────


class TestDendricMemoryConfigParse:
    def test_shorthand_form(self, parse_ast):
        d = parse_ast(
            'agent A { memory: dendric("grants") '
            'step f() { respond "x" } }'
        ).declarations[0]
        m = d.memory_config
        assert m.backend == "dendric"
        assert m.persona == "grants"

    def test_legacy_block_still_works(self, parse_ast):
        """Adding the shorthand must not regress the existing block syntax."""
        d = parse_ast(
            'agent A { memory { '
            '  store: "sqlite://./m.db" '
            '  recall strategy: "semantic" '
            '} step f() { respond "x" } }'
        ).declarations[0]
        m = d.memory_config
        assert m.backend == "sqlite"
        assert m.store == "sqlite://./m.db"

    def test_default_backend_is_sqlite(self):
        """An empty MemoryConfig should still default to sqlite, not dendric,
        so existing examples without `dendric(...)` keep their old behavior."""
        m = ast.MemoryConfig()
        assert m.backend == "sqlite"


# ── Codegen: dendric → make_memory_store ───────────────────────────────


class TestDendricCodegen:
    def test_dendric_emits_make_memory_store(self, transpile):
        py = transpile(
            'agent A { memory: dendric("grants") '
            'step f() { respond "x" } }'
        )
        assert "make_memory_store(persona='grants')" in py

    def test_sqlite_still_emits_MemoryStore(self, transpile):
        py = transpile(
            'agent A { memory { store: "sqlite://./m.db" } '
            'step f() { respond "x" } }'
        )
        assert "MemoryStore(store_url='sqlite://./m.db'" in py
        assert "make_memory_store" not in py.replace(
            "make_memory_store,", "",  # the import line is fine
        )


# ── Factory fallback ───────────────────────────────────────────────────


class TestMakeMemoryStoreFallback:
    def test_falls_back_to_local_sqlite_without_DATABASE_URL(
        self, monkeypatch, capsys, tmp_path
    ):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.setenv("DRIFT_MEMORY_DIR", str(tmp_path))

        store = make_memory_store(persona="test")
        captured = capsys.readouterr()

        assert isinstance(store, MemoryStore)
        # The fallback is announced...
        assert "SQLite memory" in captured.out
        # ...and is file-backed (persists across runs), not :memory:.
        assert ":memory:" not in store.store_url
        assert str(tmp_path) in store.store_url

    def test_fallback_persists_across_stores(self, monkeypatch, tmp_path):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.setenv("DRIFT_MEMORY_DIR", str(tmp_path))

        s1 = make_memory_store(persona="alice")
        s1.remember("remembered fact", tag="k")
        s1.close()

        # A fresh store for the same persona sees the earlier write.
        s2 = make_memory_store(persona="alice")
        recalled = s2.recall()
        assert any("remembered fact" == r for r in recalled)


# ── Mock MemoryStore implements the DendricStore surface ───────────────


class TestMockStoreDendricSurface:
    """Generated code targets DendricStore's interface. The SQLite mock
    must implement the same methods (as no-ops where appropriate) so the
    same .drift file runs in both modes without AttributeError."""

    def _fresh(self) -> MemoryStore:
        return MemoryStore(
            store_url="sqlite://:memory:",
            recall_strategy="relevant",
            max_recall=20,
        )

    def test_deja_vu_check_returns_none(self):
        store = self._fresh()
        store.remember("hello mango", tag="dog walk")
        # Mock has no archive lifecycle, so deja_vu never fires.
        assert store.deja_vu_check(context="mango") is None

    def test_consolidate_is_a_noop(self):
        store = self._fresh()
        result = store.consolidate()
        # Mock returns a sentinel dict marking the no-op so callers can
        # distinguish it from a real Dendric consolidate result.
        assert result.get("mock") is True

    def test_forget_by_tag_deletes_matching(self):
        store = self._fresh()
        store.remember("Walk Mango", tag="dog")
        store.remember("Read news", tag="news")
        store.remember("Feed Mango", tag="dog")
        result = store.forget(tag="dog")
        assert result["forgotten"] == 2
        remaining = store.recall("", key="news")
        assert any("news" in str(r) or "Read" in str(r) for r in remaining)

    def test_forget_by_below_temp_is_noop_on_mock(self):
        store = self._fresh()
        store.remember("x", tag="t")
        # Mock has no temperature; this should silently no-op so the same
        # .drift file (`forget memories where temp < 0.3`) doesn't crash.
        result = store.forget(below_temp=0.3)
        assert result["forgotten"] == 0


# ── DendricStore helpers (pure functions, no Postgres) ─────────────────


class TestSerializers:
    def test_serialize_string(self):
        assert _serialize("hello") == "hello"

    def test_serialize_dict_preserves_keys(self):
        s = _serialize({"company": "TechCo", "score": 82})
        assert "company" in s and "TechCo" in s and "82" in s

    def test_serialize_dataclass(self):
        from dataclasses import dataclass

        @dataclass
        class Score:
            company: str
            score: int

        s = _serialize(Score(company="TechCo", score=82))
        assert "TechCo" in s and "82" in s

    def test_tag_to_context_list_joins_with_space(self):
        # entity-extraction reads context as free text — words separated
        # by spaces are what it wants
        assert _tag_to_context(["healthcare", "innovate_uk"]) == "healthcare innovate_uk"

    def test_tag_to_context_string_passthrough(self):
        assert _tag_to_context("healthcare innovate_uk") == "healthcare innovate_uk"

    def test_tag_to_context_empty(self):
        assert _tag_to_context(None) == ""
        assert _tag_to_context("") == ""
