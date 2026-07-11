"""Tests for the CLI command handlers in drift/cli.py.

Focused on error-surfacing consistency across `check`/`transpile`/`run` —
each should report a CodegenError the same clean way, not leak a raw
Python traceback or misreport a compile-time rejection as a runtime error.
"""
import argparse

import pytest

from drift.cli import cmd_check, cmd_transpile, _run_once


SCHEDULE_PIPELINE = (
    'pipeline Triage {\n'
    '  schedule: "every morning"\n'
    '  input_email -> Classifier.tag\n'
    '}\n'
)


def _args(file, **overrides):
    ns = argparse.Namespace(file=file, output=None, step=None, agent=None,
                             pipeline=None, input=None, trace=False, watch=False)
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


class TestCodegenErrorSurfacing:
    """A CodegenError (parses fine, rejected at compile time — e.g. an
    unimplemented `schedule:` pipeline modifier) must be reported the same
    clean way by every command that transpiles, not just `drift check`."""

    def test_check_reports_codegen_error_cleanly(self, tmp_path, capsys):
        f = tmp_path / "p.drift"
        f.write_text(SCHEDULE_PIPELINE)
        with pytest.raises(SystemExit):
            cmd_check(_args(str(f)))
        out = capsys.readouterr().out
        assert "schedule" in out
        assert "not implemented" in out

    def test_transpile_reports_codegen_error_cleanly(self, tmp_path, capsys):
        # Regression: cmd_transpile didn't catch CodegenError at all, so
        # this used to raise an uncaught Python exception (raw traceback)
        # instead of a clean error message.
        f = tmp_path / "p.drift"
        f.write_text(SCHEDULE_PIPELINE)
        with pytest.raises(SystemExit):
            cmd_transpile(_args(str(f)))
        out = capsys.readouterr().out
        assert "schedule" in out
        assert "not implemented" in out

    def test_run_reports_codegen_error_not_runtime_error(self, tmp_path, capsys):
        # Regression: _run_once had no CodegenError handler, so it fell
        # through to the generic `except Exception` runtime-error path and
        # was mislabeled "Runtime error" even though nothing ever ran.
        f = tmp_path / "p.drift"
        f.write_text(SCHEDULE_PIPELINE)
        rc = _run_once(_args(str(f)))
        assert rc == 1
        out = capsys.readouterr().out
        assert "Runtime error" not in out
        assert "schedule" in out
        assert "not implemented" in out
