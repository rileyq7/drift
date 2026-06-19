"""Smoke tests: each example must execute end-to-end via the mock provider
without crashing, and must record at least one LLM call in the cost tracker.
"""
import importlib.util
import sys
from pathlib import Path

import pytest


def _load_module(py_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, py_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.asyncio
async def test_hello_runs(examples_dir):
    mod = _load_module(examples_dir / "hello.py", "drift_hello")
    agent = mod.Greeter()
    result = await agent.hello(name="Riley")
    # hello returns the generated greeting string
    assert isinstance(result, str)
    assert agent.cost_tracker.total_cost > 0
    assert len(agent.cost_tracker.call_log) >= 1


@pytest.mark.asyncio
async def test_inbox_sorter_runs(examples_dir):
    mod = _load_module(examples_dir / "inbox_sorter.py", "drift_inbox")
    agent = mod.InboxSorter()
    assert agent.budget.max_per_run == 0.5
    result = await agent.sort_emails(emails=[
        "Hey, quick question about the proposal",
        "URGENT: server is down",
        "Newsletter: weekly digest",
    ])
    assert isinstance(result, list)
    assert len(result) == 3
    for analysis in result:
        assert isinstance(analysis, mod.EmailAnalysis)
        assert analysis.priority in ("urgent", "normal", "low")
    # One classify call per email
    assert len(agent.cost_tracker.call_log) == 3


@pytest.mark.asyncio
async def test_grant_checker_runs(examples_dir):
    mod = _load_module(examples_dir / "grant_checker.py", "drift_grant")
    agent = mod.GrantChecker()
    result = await agent.evaluate(
        company_profile="TechCo Ltd, AI startup, 15 employees, healthcare.",
        call_text="Innovate UK Smart Grants for SMEs in AI/health, £25k-£500k.",
    )
    assert isinstance(result, mod.FitScore)
    assert 0 <= result.overall_score <= 100
    assert 0 <= result.confidence <= 1
    # Two intent calls: parse_call (extract) + evaluate (rate)
    assert len(agent.cost_tracker.call_log) >= 2
    assert agent.cost_tracker.total_cost < agent.budget.max_per_run
