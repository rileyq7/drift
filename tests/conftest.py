"""Shared pytest fixtures for the Drift test suite.

Tests always run against the mock provider — no real LLM calls. The mock
returns plausible schema-shaped data so we can exercise the full pipeline
end-to-end without network or cost.
"""
import os
import sys
from pathlib import Path

import pytest

# Make the project importable without installing.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def force_mock_provider(monkeypatch):
    """Every test gets the mock provider, even if the dev has a real key set."""
    monkeypatch.setenv("DRIFT_USE_MOCK", "1")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture
def project_root() -> Path:
    return ROOT


@pytest.fixture
def examples_dir(project_root) -> Path:
    return project_root / "examples"


@pytest.fixture
def transpile():
    """Transpile a .drift source string and return the generated Python."""
    from drift.lexer import lex
    from drift.parser import Parser
    from drift.codegen import CodeGenerator

    def _transpile(source: str) -> str:
        tokens = lex(source)
        program = Parser(tokens).parse()
        return CodeGenerator().generate(program)

    return _transpile


@pytest.fixture
def parse_ast():
    """Lex + parse a .drift source string, return the Program AST."""
    from drift.lexer import lex
    from drift.parser import Parser

    def _parse(source: str):
        return Parser(lex(source)).parse()

    return _parse
