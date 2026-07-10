"""DendricStore — adapter that lets Drift's codegen target Dendric's MemoryEngine.

Drift's generated Python calls a memory store with this shape:

    self.memory.remember(value, tag="...")
    results = self.memory.recall(description, key=...)
    match = self.memory.deja_vu_check(context=...)
    self.memory.forget(memory_id=..., below_temp=..., tag=..., older_than_days=...)
    self.memory.consolidate()

Dendric's MemoryEngine has a narrower native surface (see Dendric README and
src/engine/core/engine.py). This module translates between the two.

Dendric is the source of truth — when reality conflicts with the HTML spec,
this adapter sides with Dendric.

Translations applied here:
  - remember(value, tag): tags are folded into the `context` parameter so
    Dendric's entity-extractor picks them up. Non-string values are
    serialized via a structured repr that preserves field names.
  - recall(description, key): both get concatenated into the query string.
  - deja_vu_check(context): calls recall(min_temp=0.0) and filters for
    results in the `archive` region. Those memories only reach top_k via
    the spreading-activation archive trigger — i.e. they're surfaced
    because something in the current context resonated with a dormant
    pattern. That IS deja_vu. No separate Dendric method needed.
  - forget: by-id and by-temp pass through directly. by-tag and
    by-older-than are query-then-delete shims (Dendric doesn't support
    predicate-based pruning natively).
  - consolidate: pass-through to eng.consolidate().

Environment / config:
  DENDRIC_HOME    — path to the Dendric repo. Defaults to ~/Dendric.
                    Dendric has no pyproject.toml, so we prepend this to
                    sys.path before importing src.engine.*
  DATABASE_URL    — Postgres connection string. If unset, the factory in
                    this module falls back to the SQLite mock store.
  OPENAI_API_KEY  — required for real embeddings. Without it Dendric will
                    refuse to start unless MEMORY_ENGINE_ALLOW_HASH_EMBED=1
                    is set (offline pseudo-embedding mode, for tests).
"""

from __future__ import annotations

import os
import sys
import json
from dataclasses import asdict, is_dataclass
from typing import Any, Optional

# ── Dendric import via sys.path injection ──────────────────────────────
# Dendric has no installable package; its source dir is prepended to sys.path
# so `from src.engine.core.engine import MemoryEngine` works. The path is
# configurable via DENDRIC_HOME for users who put it elsewhere.
#
# This is done LAZILY (see _ensure_dendric_on_path), only when a DendricStore
# is actually constructed — importing this module must not mutate sys.path for
# every drift process, which would let anything under ~/Dendric shadow imports.
_DENDRIC_HOME = os.environ.get(
    "DENDRIC_HOME",
    os.path.expanduser("~/Dendric"),
)


def _ensure_dendric_on_path() -> None:
    if _DENDRIC_HOME not in sys.path:
        sys.path.insert(0, _DENDRIC_HOME)


def _serialize(value: Any) -> str:
    """Turn an arbitrary Drift value into a string Dendric can store.

    Dendric stores raw text. We preserve field names for dataclasses and
    dicts so entity extraction can still hit them, and so a later recall
    can match on field values.
    """
    if isinstance(value, str):
        return value
    if is_dataclass(value):
        return json.dumps(asdict(value), default=str, sort_keys=True)
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str, sort_keys=True)
    return str(value)


def _tag_to_context(tag: Any) -> str:
    """Render a tag (string, list, or comma-joined string) into a context
    string Dendric's entity extractor can read. Multi-word tags survive
    as-is — the entity extractor handles capitalization heuristics."""
    if tag is None or tag == "":
        return ""
    if isinstance(tag, (list, tuple)):
        return " ".join(str(t) for t in tag)
    return str(tag)


class DejaVuMatch:
    """Returned by DendricStore.deja_vu_check when a dormant memory
    fires via the archive trigger.

    Attributes:
        memory: the underlying Dendric memory dict (region='archive')
        pattern_type: a string used by Drift's `deja_vu match on ...`
            arms for dispatch. v1 classification: the memory's source +
            context concatenated; arms match via substring containment.
            v2 (later) can plug an LLM classifier here.
        activation: the result's temperature (proxy for archive-trigger
            strength).
    """

    def __init__(self, memory: dict):
        self.memory = memory
        self.activation = float(memory.get("temperature", 0.0))
        # v1 pattern classification: source + context fields, lowercased.
        # Drift's `deja_vu match on X { "pattern_name" -> ... }` arms
        # fire when their pattern string appears as a substring here.
        source = str(memory.get("source", ""))
        context = str(memory.get("context", ""))
        self.pattern_type = f"{source} {context}".strip().lower()

    def matches(self, pattern: str) -> bool:
        """Used by generated code to dispatch deja_vu match arms."""
        return pattern.lower() in self.pattern_type

    # Convenience attribute access — generated code may reference
    # match.something where 'something' was set as a tag at remember time.
    # Forward unknown attribute access to the underlying memory dict so
    # `match.grant_title` works when grant_title was in the original data.
    def __getattr__(self, name: str) -> Any:
        # __getattr__ only fires when normal lookup fails, so `self.memory`
        # here would recurse infinitely if `memory` itself is missing (e.g.
        # during unpickling, before __init__ runs). Guard by reading the
        # instance dict directly and bailing out if the backing store is absent.
        if name in ("memory", "__dict__"):
            raise AttributeError(name)
        memory = self.__dict__.get("memory")
        if memory is None:
            raise AttributeError(name)
        if name in memory:
            return memory[name]
        # Try to parse raw_content as JSON (for serialized dataclasses)
        raw = memory.get("raw_content", "")
        if raw and raw.startswith("{"):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict) and name in parsed:
                    return parsed[name]
            except (json.JSONDecodeError, ValueError):
                pass
        raise AttributeError(name)


def _schema_is_initialized(db_url: str) -> bool:
    """Probe whether Dendric's schema already exists on this database.

    We open a short-lived connection in autocommit mode (no transaction,
    so no locks lingering), check for the `memories` table, and close.
    Used by DendricStore.__init__ to skip Dendric's run_migrations on
    second-and-later instances within the same process — see comment
    on the monkeypatch below."""
    import psycopg2
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name = 'memories'"
            )
            return cur.fetchone() is not None
    finally:
        conn.close()


class DendricStore:
    """The real adapter — wraps a Dendric MemoryEngine."""

    def __init__(
        self,
        persona: str = "",
        db_url: Optional[str] = None,
    ):
        # Import lazily so a missing/misconfigured Dendric doesn't crash
        # the whole drift runtime at import time — the factory in this
        # module catches ImportError and falls back to the mock. The sys.path
        # injection also happens here (not at module import) so unused drift
        # processes never get ~/Dendric on their path.
        _ensure_dendric_on_path()
        from src.engine.core.engine import MemoryEngine  # type: ignore
        from src.engine.config import EngineConfig  # type: ignore
        from src.engine.storage import postgres as _pg_store  # type: ignore

        url = db_url or os.environ.get("DATABASE_URL")
        if not url:
            raise RuntimeError(
                "DendricStore requires DATABASE_URL or db_url. "
                "Set DATABASE_URL to a Postgres+pgvector URL, or use "
                "make_memory_store() which falls back to the mock."
            )

        # ── Idempotent migration guard ───────────────────────────────
        # Dendric's PostgresStore.__init__ unconditionally calls
        # run_migrations(conn), which issues ~40 CREATE/ALTER statements.
        # Each acquires an AccessExclusiveLock on its table inside the
        # connection's open transaction (autocommit=False). For a
        # second connection from the same process, those locks block —
        # not a Python-level race but a server-side wait, with no
        # deadlock cycle for Postgres to detect.
        #
        # Fix: if `memories` already exists, neutralize run_migrations
        # for the duration of MemoryEngine() construction so the second
        # connection skips the lock-taking DDL entirely. Idempotent
        # because IF NOT EXISTS already made the statements a no-op
        # logically; we're just avoiding the lock acquisition.
        self._persona = persona
        if _schema_is_initialized(url):
            orig = _pg_store.run_migrations
            _pg_store.run_migrations = lambda _conn: None
            try:
                self._eng = MemoryEngine(config=EngineConfig(
                    db_url=url,
                    persona=persona,
                ))
            finally:
                _pg_store.run_migrations = orig
        else:
            self._eng = MemoryEngine(config=EngineConfig(
                db_url=url,
                persona=persona,
            ))

    # ── Drift codegen surface ─────────────────────────────────────────

    def remember(self, value: Any, tag: str = "") -> dict:
        """Drift: remember <value> [tagged <tag>]
        Dendric: eng.remember(content, source, context)
        Tags fold into context so entity-extraction picks them up."""
        content = _serialize(value)
        context = _tag_to_context(tag)
        return self._eng.remember(
            content=content,
            source=f"drift_agent:{self._persona}" if self._persona else "drift_agent",
            context=context,
        )

    def _safe_recall(self, query: str, **kwargs) -> list:
        """eng.recall(...) with pgvector real-precision overflow defense.

        Hash-pseudo-embedding mode (MEMORY_ENGINE_ALLOW_HASH_EMBED=1, used
        for offline tests) can produce query vectors with float64
        components smaller than float32 min, which pgvector rejects with
        NumericValueOutOfRange. Real OpenAI embeddings don't trip this.

        The error aborts the psycopg2 transaction, so we also roll back
        the connection — otherwise subsequent calls fail with
        InFailedSqlTransaction. Returns [] on overflow."""
        try:
            return self._eng.recall(query=query, **kwargs)
        except Exception as e:
            if "out of range for type real" in str(e):
                conn = getattr(self._eng.store, "conn", None)
                if conn is not None:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                return []
            raise

    def recall(self, query: str = "", key: Any = None) -> list:
        """Drift: recall <description> [for <key>]
        Dendric: eng.recall(query, top_k)"""
        # Concatenate description + key so both inform the query.
        # If key is a complex object, serialize it.
        q_parts = [str(query)] if query else []
        if key is not None and key != "":
            q_parts.append(_serialize(key))
        full_query = " ".join(p for p in q_parts if p)
        if not full_query:
            return []
        return self._safe_recall(full_query, top_k=10)

    def deja_vu_check(self, context: Any) -> Optional[DejaVuMatch]:
        """Drift: deja_vu match on <context> { ... }
        Not a separate Dendric method — emergent from recall() when the
        spreading-activation archive trigger fires. min_temp=0.1 matches
        the archive band ceiling; Dendric's vector path excludes archive
        by SQL regardless, so dormant memories reach us via the
        associative spreading-activation path."""
        query = _serialize(context)
        if not query:
            return None
        results = self._safe_recall(query, top_k=5, min_temp=0.1)
        archive_hits = [r for r in results if r.get("region") == "archive"]
        if not archive_hits:
            return None
        # Strongest archive hit by temperature (which here acts as a
        # proxy for activation — see fusion.py archive_modulation_override).
        best = max(archive_hits, key=lambda r: r.get("temperature", 0.0))
        return DejaVuMatch(best)

    def forget(
        self,
        memory_id: Optional[str] = None,
        below_temp: Optional[float] = None,
        tag: Optional[str] = None,
        older_than_days: Optional[int] = None,
    ) -> dict:
        """Drift: forget memories tagged X / older than Nd / where temp < X
        Dendric: native by-id and by-temp; tag/age are query-then-delete shims."""
        if memory_id is not None:
            return self._eng.forget(memory_id=memory_id)
        if below_temp is not None:
            return self._eng.forget(below_temp=below_temp)
        # Tag or age: query, then forget each by ID.
        # by-tag: recall using the tag as query, filter by context containment.
        # by-age: query everything, filter by created_at.
        forgotten = 0
        if tag is not None:
            results = self._safe_recall(str(tag), top_k=200, min_temp=0.1)
            tag_l = str(tag).lower()
            for r in results:
                ctx = str(r.get("context", "")).lower()
                if tag_l in ctx:
                    self._eng.forget(memory_id=r["id"])
                    forgotten += 1
        if older_than_days is not None:
            from datetime import datetime, timezone, timedelta
            cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
            all_mems = self._eng.get_all(limit=10_000)
            for r in all_mems:
                created = r.get("created_at")
                if isinstance(created, str):
                    try:
                        created = datetime.fromisoformat(created)
                    except ValueError:
                        continue
                if isinstance(created, datetime) and created < cutoff:
                    self._eng.forget(memory_id=r["id"])
                    forgotten += 1
        return {"forgotten": forgotten}

    def consolidate(self) -> dict:
        """Drift runtime calls this at agent run boundaries."""
        return self._eng.consolidate()

    def close(self):
        if hasattr(self._eng, "close"):
            self._eng.close()


# ── Factory + mock fallback ────────────────────────────────────────────


def _fallback_sqlite_path(persona: str) -> str:
    """A stable on-disk SQLite path for a persona's fallback memory.

    File-backed (not :memory:) so `remember` survives between `drift run`
    invocations — an in-memory store would evaporate at process exit and make
    a "persistent" agent silently forgetful.
    """
    import hashlib
    base = os.environ.get("DRIFT_MEMORY_DIR") or os.path.join(
        os.path.expanduser("~"), ".drift", "memory"
    )
    os.makedirs(base, exist_ok=True)
    slug = hashlib.sha1((persona or "default").encode()).hexdigest()[:16]
    return os.path.join(base, f"{slug}.db")


def make_memory_store(persona: str = "", **kwargs) -> Any:
    """Try real Dendric; fall back to a file-backed SQLite store if it can't
    be reached. Announces the fallback whenever it changes durability so a
    misconfigured prod env doesn't silently downgrade.

    Generated agent __init__ calls this for `memory: dendric("name")`."""
    db_url = kwargs.get("db_url") or os.environ.get("DATABASE_URL")
    if db_url:
        try:
            return DendricStore(persona=persona, db_url=db_url)
        except Exception as e:
            # Real Dendric was requested but failed — surface it loudly EVERY
            # time (not once per process): each downgraded persona is losing
            # the durability/semantics it asked for.
            print(
                f"  ⚠  DendricStore unavailable for persona {persona!r} "
                f"({type(e).__name__}: {e}). Falling back to local SQLite memory."
            )
    else:
        print(
            "  ℹ  No DATABASE_URL — using local file-backed SQLite memory "
            f"for persona {persona!r} (set DATABASE_URL for Dendric)."
        )

    # Fall back to the file-backed SQLite MemoryStore from runtime/core.py so
    # memory persists across runs. Tag-substring recall is good enough for dev.
    from .core import MemoryStore
    return MemoryStore(
        store_url=f"sqlite://{_fallback_sqlite_path(persona)}",
        recall_strategy="relevant",
        max_recall=20,
    )
