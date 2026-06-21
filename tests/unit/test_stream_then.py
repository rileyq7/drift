"""Tests for `model: stream "fast" then "slow"` — temporal model routing.

Three layers:
  1. Parser sets mode + stream_model + then_model
  2. Codegen emits StreamThenRouter (not ModelRouter)
  3. Runtime: stream_then_call fires both concurrently and the bridge
     callback fires before the reasoning result returns
"""
import asyncio

import pytest

from drift import ast_nodes as ast
from drift.runtime import StreamThenRouter


# ── Parser ─────────────────────────────────────────────────────────────


class TestStreamThenParse:
    def test_basic(self, parse_ast):
        d = parse_ast(
            'agent J { '
            'model: stream "haiku" then "sonnet" '
            'step r(u: string) -> string { return "ok" } '
            '}'
        ).declarations[0]
        m = d.model_config
        assert m.mode == "stream_then"
        assert m.stream_model == "haiku"
        assert m.then_model == "sonnet"
        # default points at the slow model so other systems that only
        # know about ModelConfig.default still get the reasoning model.
        assert m.default == "sonnet"

    def test_prefer_still_works(self, parse_ast):
        """Adding `stream` keyword must not regress prefer/fallback."""
        d = parse_ast(
            'agent A { '
            'model: prefer "claude" fallback "gpt" '
            'step r() -> string { return "x" } '
            '}'
        ).declarations[0]
        m = d.model_config
        assert m.mode == "prefer"
        assert m.prefer == "claude"
        assert m.fallback == "gpt"

    def test_default_form_still_works(self, parse_ast):
        d = parse_ast(
            'agent A { model: "haiku" '
            'step r() -> string { return "x" } '
            '}'
        ).declarations[0]
        assert d.model_config.mode == "default"
        assert d.model_config.default == "haiku"


# ── Codegen ────────────────────────────────────────────────────────────


class TestStreamThenCodegen:
    def test_emits_StreamThenRouter(self, transpile):
        py = transpile(
            'agent J { '
            'model: stream "haiku" then "sonnet" '
            'step r(u: string) -> string { return "ok" } '
            '}'
        )
        assert "StreamThenRouter(" in py
        assert 'stream_model="haiku"' in py
        assert 'then_model="sonnet"' in py

    def test_default_still_uses_ModelRouter(self, transpile):
        py = transpile(
            'agent A { model: "haiku" step r() -> string { return "x" } }'
        )
        # ModelRouter, not StreamThenRouter
        assert "ModelRouter(" in py
        assert "StreamThenRouter(" not in py


# ── Runtime ────────────────────────────────────────────────────────────


class _StubProvider:
    """Records every call() and lets the test control timing via per-model
    delays. Returns (text, tokens_in, tokens_out) tuple — matches the
    real provider contract."""

    def __init__(self, delays: dict):
        self.delays = delays           # {model_name: seconds}
        self.calls: list[str] = []

    async def call(self, model_name, system_prompt, user_prompt,
                   output_schema):
        self.calls.append(model_name)
        delay = self.delays.get(model_name, 0)
        if delay:
            await asyncio.sleep(delay)
        return (f"<reply from {model_name}>", 10, 5)


class TestStreamThenRuntime:
    @pytest.mark.asyncio
    async def test_both_models_fire_concurrently(self):
        # Bridge model finishes fast; reasoning model takes longer.
        # If they were sequential, total would be ~0.20s.
        # Concurrent → ~0.15s (the slower one).
        provider = _StubProvider(delays={"fast": 0.05, "slow": 0.15})
        router = StreamThenRouter(
            default="slow", stream_model="fast", then_model="slow",
        )
        t0 = asyncio.get_event_loop().time()
        result = await router.stream_then_call(
            provider, "sys", "user", None, on_bridge=None,
        )
        elapsed = asyncio.get_event_loop().time() - t0
        # Both fired
        assert set(provider.calls) == {"fast", "slow"}
        # Result is the slow model's reply
        assert "slow" in result[0]
        # Concurrent — total should be near max(0.05, 0.15) = 0.15s,
        # NOT 0.05 + 0.15 = 0.20s. Allow generous slack for scheduler.
        assert elapsed < 0.19, f"stream-then was sequential ({elapsed:.3f}s)"

    @pytest.mark.asyncio
    async def test_bridge_callback_fires_before_reasoning(self):
        """The whole point of stream-then: the user hears the bridge
        response while the reasoning model is still thinking."""
        bridge_at = []
        reasoning_at = []

        async def cb(text):
            bridge_at.append((text, asyncio.get_event_loop().time()))

        provider = _StubProvider(delays={"fast": 0.02, "slow": 0.20})
        router = StreamThenRouter(
            default="slow", stream_model="fast", then_model="slow",
        )

        t0 = asyncio.get_event_loop().time()
        result = await router.stream_then_call(
            provider, "sys", "user", None, on_bridge=cb,
        )
        reasoning_at.append(asyncio.get_event_loop().time())

        # Allow the scheduled bridge callback to actually run
        await asyncio.sleep(0)

        assert bridge_at, "bridge callback never fired"
        assert "fast" in bridge_at[0][0]
        # Bridge timestamp must be before reasoning timestamp.
        assert bridge_at[0][1] < reasoning_at[0]
        # And bridge fires well before the reasoning completes — within
        # ~3x the fast model's delay, not waiting for the slow one.
        assert bridge_at[0][1] - t0 < 0.10

    @pytest.mark.asyncio
    async def test_bridge_callback_exception_does_not_break_reasoning(self):
        """A failing bridge callback must NOT poison the reasoning result —
        the user still gets the real answer even if the acknowledgement
        layer throws."""
        async def bad_cb(text):
            raise RuntimeError("TTS pipeline broken")

        provider = _StubProvider(delays={"fast": 0.01, "slow": 0.05})
        router = StreamThenRouter(
            default="slow", stream_model="fast", then_model="slow",
        )
        result = await router.stream_then_call(
            provider, "sys", "user", None, on_bridge=bad_cb,
        )
        # Reasoning result came through despite the bridge error.
        assert "slow" in result[0]
        # Let the bridge task complete its print() before the test ends.
        await asyncio.sleep(0)
