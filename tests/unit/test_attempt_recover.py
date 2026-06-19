"""Tests for §8 attempt/recover — language-level error handling."""
import pytest

from drift import ast_nodes as ast
from drift.lexer import lex
from drift.parser import Parser, ParseError


def parse(source: str):
    return Parser(lex(source)).parse()


def find_attempt(src: str) -> ast.AttemptStmt:
    agent_src = f'agent A {{ step f() {{ {src} }} }}'
    d = parse(agent_src).declarations[0]
    for stmt in d.steps[0].body:
        if isinstance(stmt, ast.AttemptStmt):
            return stmt
    raise AssertionError("no AttemptStmt in body")


class TestAttemptParse:
    def test_minimal_attempt(self):
        a = find_attempt(
            'attempt { respond "x" } recover from { any error -> respond "fail" }'
        )
        assert len(a.body) == 1
        assert len(a.arms) == 1
        assert a.arms[0].is_default

    def test_multiple_specific_arms(self):
        a = find_attempt(
            'attempt { let x = classify y as Z } '
            'recover from { '
            '  ModelUnavailable -> respond "down" '
            '  SchemaViolation -> retry '
            '  any error -> respond "?" '
            '}'
        )
        assert len(a.arms) == 3
        assert a.arms[0].error_type == "ModelUnavailable"
        assert a.arms[1].error_type == "SchemaViolation"
        assert isinstance(a.arms[1].body[0], ast.RetryStmt)
        assert a.arms[2].is_default

    def test_arm_with_block_body(self):
        a = find_attempt(
            'attempt { respond "x" } '
            'recover from { '
            '  RateLimited -> { respond "wait" respond "slow" } '
            '  any error -> retry '
            '}'
        )
        assert len(a.arms[0].body) == 2

    def test_fail_with_message(self):
        a = find_attempt(
            'attempt { respond "x" } '
            'recover from { any error -> fail with "bad: {_err}" }'
        )
        body = a.arms[0].body
        assert len(body) == 1
        assert isinstance(body[0], ast.FailStmt)


class TestAttemptCodegen:
    def _gen(self, src: str) -> str:
        from drift.codegen import CodeGenerator
        return CodeGenerator().generate(parse(
            f'agent A {{ step f() {{ {src} }} }}'
        ))

    def test_emits_try_except_for_specific(self):
        out = self._gen(
            'attempt { respond "x" } '
            'recover from { ModelUnavailable -> respond "down" }'
        )
        assert "try:" in out
        assert "except ModelUnavailable" in out

    def test_default_becomes_drift_error_catch(self):
        out = self._gen(
            'attempt { respond "x" } recover from { any error -> respond "f" }'
        )
        assert "except DriftError" in out

    def test_retry_compiles_to_continue(self):
        out = self._gen(
            'attempt { respond "x" } '
            'recover from { SchemaViolation -> retry }'
        )
        assert "continue" in out

    def test_fail_with_compiles_to_raise(self):
        out = self._gen(
            'attempt { respond "x" } '
            'recover from { any error -> fail with "boom" }'
        )
        assert "raise StepFailed" in out

    def test_specific_arm_before_default(self):
        # Python requires subclass-arms before base-class arms. Even if the
        # user puts `any error` first in source, codegen must emit it last.
        out = self._gen(
            'attempt { respond "x" } '
            'recover from { '
            '  any error -> respond "any" '
            '  ModelUnavailable -> respond "down" '
            '}'
        )
        any_pos = out.find("except DriftError")
        mu_pos = out.find("except ModelUnavailable")
        assert mu_pos != -1 and any_pos != -1
        assert mu_pos < any_pos, "specific arm must precede `any error`"

    def test_loop_breaks_on_success(self):
        # The try body should end with `break` so a successful attempt
        # exits the retry loop.
        out = self._gen(
            'attempt { respond "x" } recover from { any error -> retry }'
        )
        # The break should land inside the try block. Lazy check: it's there.
        assert "break" in out
        assert "for _attempt in range(" in out


class TestAttemptRecoverEndToEnd:
    """Compile + execute an attempt block, verifying retry actually happens."""

    @pytest.mark.asyncio
    async def test_retry_then_succeed(self, transpile, tmp_path):
        from drift.runtime import SchemaViolation
        src = (
            'agent A { step f() -> string { '
            '  attempt { let x = classify "x" as string return x } '
            '  recover from { SchemaViolation -> retry } '
            '} }'
        )
        py = transpile(src)
        py = py.replace("Source: <drift_file>", "Source: inline")
        path = tmp_path / "gen.py"
        path.write_text(py)
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("gen_attempt", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["gen_attempt"] = mod
        spec.loader.exec_module(mod)

        # Patch the agent's intent to fail twice then succeed
        agent = mod.A()
        original_intent = agent.intent
        calls = {"n": 0}

        async def flaky_intent(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise SchemaViolation("bad")
            return "good"

        agent.intent = flaky_intent
        result = await agent.f()
        assert result == "good"
        assert calls["n"] == 3

    @pytest.mark.asyncio
    async def test_retry_exhausts_then_raises(self, transpile, tmp_path):
        from drift.runtime import SchemaViolation
        src = (
            'agent A { step f() -> string { '
            '  attempt { let x = classify "x" as string return x } '
            '  recover from { SchemaViolation -> retry } '
            '} }'
        )
        py = transpile(src).replace("Source: <drift_file>", "Source: inline")
        path = tmp_path / "gen.py"
        path.write_text(py)
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("gen_exhaust", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["gen_exhaust"] = mod
        spec.loader.exec_module(mod)

        agent = mod.A()

        async def always_fail(*args, **kwargs):
            raise SchemaViolation("nope")

        agent.intent = always_fail
        # The step decorator wraps SchemaViolation in StepFailed after retries
        from drift.runtime import StepFailed
        with pytest.raises((SchemaViolation, StepFailed)):
            await agent.f()

    @pytest.mark.asyncio
    async def test_fallthrough_recovery_exits_loop(self, transpile, tmp_path):
        """An arm that doesn't retry should NOT re-run the attempt block."""
        from drift.runtime import SchemaViolation
        src = (
            'agent A { step f() -> string { '
            '  attempt { let x = classify "x" as string return x } '
            '  recover from { SchemaViolation -> return "fallback" } '
            '} }'
        )
        py = transpile(src).replace("Source: <drift_file>", "Source: inline")
        path = tmp_path / "gen.py"
        path.write_text(py)
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("gen_fall", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["gen_fall"] = mod
        spec.loader.exec_module(mod)

        agent = mod.A()
        calls = {"n": 0}

        async def fail_once(*args, **kwargs):
            calls["n"] += 1
            raise SchemaViolation("x")

        agent.intent = fail_once
        result = await agent.f()
        assert result == "fallback"
        # Body ran once; arm returned; no retry
        assert calls["n"] == 1
