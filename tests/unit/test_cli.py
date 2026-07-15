"""Tests for the CLI command handlers in drift/cli.py.

Focused on error-surfacing consistency across `check`/`transpile`/`run` —
each should report a CodegenError the same clean way, not leak a raw
Python traceback or misreport a compile-time rejection as a runtime error.
"""
import argparse
import io
import json

import pytest

from drift.cli import (
    cmd_check, cmd_transpile, cmd_new, cmd_fmt, cmd_schema, _run_once,
    _discover_drift_dependencies,
)


SCHEDULE_PIPELINE = (
    'pipeline Triage {\n'
    '  schedule: "every morning"\n'
    '  input_email -> Classifier.tag\n'
    '}\n'
)


def _args(file, **overrides):
    ns = argparse.Namespace(file=file, output=None, step=None, agent=None,
                             pipeline=None, input=None, trace=False, watch=False,
                             check=False, stdout=False, json=False, name=None)
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


class TestAgentSelectionOrder:
    """`drift run` with no --agent must run the first-DECLARED agent, not
    the alphabetically-first one. Regression: agents were collected via
    dir(module), which returns names alphabetically, so `list(agents.
    values())[0]` silently picked whichever class name sorted first —
    contradicting LLM.md's documented "runs the first agent's first
    step", with no error or indication the "wrong" agent ran."""

    SRC = (
        'agent Zeta { step greet() -> string { return "Hello from Zeta" } } '
        'agent Alpha { step greet() -> string { return "Hello from Alpha" } }'
    )

    def test_first_declared_agent_runs_not_alphabetically_first(
        self, tmp_path, capsys
    ):
        f = tmp_path / "order.drift"
        f.write_text(self.SRC)
        rc = _run_once(_args(str(f)))
        assert rc == 0
        out = capsys.readouterr().out
        assert "Hello from Zeta" in out
        assert "Hello from Alpha" not in out


class TestNewScaffold:
    """`drift new` output should pass `drift check` and `drift fmt --check`
    cleanly out of the box — an LLM agent scaffolding a project and then
    sanity-checking it shouldn't hit a false-alarm-looking fmt failure on
    a template it never touched."""

    def test_scaffold_passes_check_and_fmt_check(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        cmd_new(argparse.Namespace(name="proj", model=None))
        capsys.readouterr()

        drift_file = tmp_path / "proj" / "proj.drift"
        assert drift_file.exists()

        cmd_check(_args(str(drift_file)))  # no SystemExit on success
        out = capsys.readouterr().out
        assert "syntax OK" in out

        cmd_fmt(_args(str(drift_file), check=True))
        out = capsys.readouterr().out
        assert "already formatted" in out


class TestCrossFileImportResolution:
    """Regression: `import { X } from "./other.drift"` codegens to a plain
    sibling-file Python import (gen_import's own docstring: "Sibling
    .drift file -> strip dirs and extension"), which only works if (a)
    the sibling's .py has been transpiled and (b) the sibling's directory
    is importable. Neither was true for `drift run` before this fix —
    running with a relative/absolute path from outside the .drift file's
    own directory raised ModuleNotFoundError even with the dependency
    manually pre-transpiled, and there was no auto-transpile of `.drift`
    dependencies at all despite nothing in LLM.md's own `import` example
    suggesting a manual pre-transpile step was needed."""

    SCHEMA_SRC = 'schema Shared { name: string }\n'
    MAIN_SRC = (
        'import { Shared } from "./schema_dep.drift"\n'
        'agent A { model: "claude-haiku" '
        '  step f(item: Shared) -> string { return item.name } }'
    )

    def test_dependency_is_auto_transpiled(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DRIFT_USE_MOCK", "1")
        (tmp_path / "schema_dep.drift").write_text(self.SCHEMA_SRC)
        main_file = tmp_path / "main.drift"
        main_file.write_text(self.MAIN_SRC)

        assert not (tmp_path / "schema_dep.py").exists()
        rc = _run_once(_args(str(main_file), input='{"item": {"name": "widget"}}'))
        assert rc == 0
        assert (tmp_path / "schema_dep.py").exists()

    def test_run_works_regardless_of_caller_cwd(self, tmp_path, monkeypatch):
        # The bug: this worked when CWD == the .drift file's directory,
        # and failed with ModuleNotFoundError from anywhere else — e.g.
        # a repo root, which is how `drift run examples/apps/x.drift` is
        # actually invoked in practice.
        monkeypatch.setenv("DRIFT_USE_MOCK", "1")
        (tmp_path / "schema_dep.drift").write_text(self.SCHEMA_SRC)
        main_file = tmp_path / "main.drift"
        main_file.write_text(self.MAIN_SRC)

        elsewhere = tmp_path.parent
        monkeypatch.chdir(elsewhere)
        rc = _run_once(_args(str(main_file), input='{"item": {"name": "widget"}}'))
        assert rc == 0

    def test_subdirectory_organized_import_resolves(self, tmp_path, monkeypatch):
        # LLM.md §14's own SECOND documented example is a subdirectory
        # import: `import GrantChecker from "./agents/checker.drift"`.
        # gen_import strips the `agents/` prefix entirely (`from checker
        # import GrantChecker` — no subdirectory in the generated import
        # statement at all), so that subdirectory needs to be on sys.path
        # too, not just the main file's own directory.
        monkeypatch.setenv("DRIFT_USE_MOCK", "1")
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "checker.drift").write_text(self.SCHEMA_SRC)
        main_file = tmp_path / "main.drift"
        main_file.write_text(
            'import { Shared } from "./agents/checker.drift"\n'
            'agent A { model: "claude-haiku" '
            '  step f(item: Shared) -> string { return item.name } }'
        )

        rc = _run_once(_args(str(main_file), input='{"item": {"name": "widget"}}'))
        assert rc == 0
        assert (agents_dir / "checker.py").exists()

    def test_stale_dependency_module_not_reused_across_runs(self, tmp_path, monkeypatch, capsys):
        # Regression: `sys.modules.pop('drift_generated', None)` already
        # force-clears the MAIN module before each run (for --watch mode),
        # but the fix that made dependency imports work at all introduced
        # the same staleness risk one level down — `from schema_dep import
        # Shared` caches sys.modules['schema_dep'] process-wide the first
        # time it's imported. Without also clearing that, a later
        # _run_once call — in --watch mode watching a file whose
        # dependency changed, or any long-lived process making repeated
        # calls — would silently reuse the FIRST run's cached dependency
        # module even after the dependency's .drift source (and freshly
        # re-transpiled .py) changed.
        monkeypatch.setenv("DRIFT_USE_MOCK", "1")
        dep_file = tmp_path / "schema_dep.drift"
        main_file = tmp_path / "main.drift"
        main_file.write_text(self.MAIN_SRC)

        dep_file.write_text('schema Shared { name: string }\n')
        rc1 = _run_once(_args(str(main_file), input='{"item": {"name": "first"}}'))
        assert rc1 == 0
        out1 = capsys.readouterr().out
        assert "first" in out1

        # Change the dependency's shape entirely (different field name) —
        # a stale cached module would still have the OLD `name` field and
        # either silently return the wrong thing or crash differently
        # than a correct fresh import would.
        dep_file.write_text('schema Shared { different_field: string }\n')
        main_file.write_text(
            'import { Shared } from "./schema_dep.drift"\n'
            'agent A { model: "claude-haiku" '
            '  step f(item: Shared) -> string { return item.different_field } }'
        )
        rc2 = _run_once(_args(str(main_file), input='{"item": {"different_field": "second"}}'))
        assert rc2 == 0
        out2 = capsys.readouterr().out
        assert "second" in out2

    def test_dependency_syntax_error_attributed_to_dependency_not_importer(
        self, tmp_path, monkeypatch, capsys
    ):
        # Regression: a genuine ParseError in the DEPENDENCY's own source
        # used to propagate uncaught out of _transpile_drift_dependencies
        # and get caught by _run_once's generic `except ParseError`
        # handler, which unconditionally prints using args.file (the
        # IMPORTER) and the importer's OWN source text — but the
        # exception's line/col refer to positions in the dependency's
        # text. Since the two source strings are unrelated, this produced
        # a caret pointing at a essentially random character in the
        # wrong file, actively misleading about where the real problem is.
        monkeypatch.setenv("DRIFT_USE_MOCK", "1")
        # Comma-separated schema fields are invalid (fields are newline-
        # separated) — a real, easy mistake to make.
        (tmp_path / "schema_dep.drift").write_text(
            "schema Shared { name: string, extra: string }\n"
        )
        main_file = tmp_path / "main.drift"
        main_file.write_text(self.MAIN_SRC)

        rc = _run_once(_args(str(main_file), input='{"item": {"name": "x"}}'))
        assert rc == 1
        err = capsys.readouterr().err
        assert "schema_dep.drift" in err
        assert "main.drift" not in err
        assert "schema Shared { name: string, extra: string }" in err


class TestWatchModeDependencyDiscovery:
    """Regression: `--watch` mode only polled the main file's mtime — a
    cross-file `import` dependency changing (a shared schema file, most
    commonly) triggered no re-run at all, leaving the watcher silently
    showing stale output until the main file itself was also touched."""

    def test_discovers_direct_dependency(self, tmp_path):
        (tmp_path / "dep.drift").write_text("schema Shared { name: string }\n")
        main_file = tmp_path / "main.drift"
        main_file.write_text(
            'import { Shared } from "./dep.drift"\n'
            'agent A { model: "claude-haiku" '
            '  step f(item: Shared) -> string { return item.name } }'
        )
        deps = _discover_drift_dependencies(main_file)
        assert deps == {(tmp_path / "dep.drift").resolve()}

    def test_discovers_transitive_dependency(self, tmp_path):
        (tmp_path / "grandchild.drift").write_text("schema Inner { x: string }\n")
        (tmp_path / "child.drift").write_text(
            'import { Inner } from "./grandchild.drift"\n'
            "schema Shared { inner: Inner }\n"
        )
        main_file = tmp_path / "main.drift"
        main_file.write_text(
            'import { Shared } from "./child.drift"\n'
            'agent A { model: "claude-haiku" '
            '  step f(item: Shared) -> string { return "x" } }'
        )
        deps = _discover_drift_dependencies(main_file)
        assert deps == {
            (tmp_path / "child.drift").resolve(),
            (tmp_path / "grandchild.drift").resolve(),
        }

    def test_no_dependencies_returns_empty_set(self, tmp_path):
        main_file = tmp_path / "main.drift"
        main_file.write_text(
            'agent A { model: "claude-haiku" step f() -> string { return "x" } }'
        )
        assert _discover_drift_dependencies(main_file) == set()

    def test_malformed_dependency_does_not_crash_discovery(self, tmp_path):
        # Discovery is used inside the --watch polling loop, which must
        # keep running even if a dependency is mid-edit and momentarily
        # has a syntax error — the next _run_once call surfaces that
        # properly; discovery itself should just return what it found
        # before the error rather than raising.
        (tmp_path / "dep.drift").write_text("schema Shared { name: string, }\n")
        main_file = tmp_path / "main.drift"
        main_file.write_text(
            'import { Shared } from "./dep.drift"\n'
            'agent A { model: "claude-haiku" '
            '  step f(item: Shared) -> string { return item.name } }'
        )
        deps = _discover_drift_dependencies(main_file)
        assert isinstance(deps, set)  # no exception raised


TWO_SCHEMA_SRC = (
    'schema Grant {\n'
    '  amount: number between 0 and 1000000\n'
    '  status: one of "pending", "approved", "denied"\n'
    '}\n'
    'schema Result {\n'
    '  ok: bool\n'
    '}\n'
    'agent Checker {\n'
    '  step check(g: Grant) -> Result { respond Result { ok: true } }\n'
    '}\n'
)


class TestSchemaCommand:
    """`drift schema` renders a program's schema block(s) as JSON Schema —
    free (no agent run, no budget spent), for a caller (e.g. a tool
    registry) that needs a schema's shape without executing anything."""

    def test_no_name_prints_all_schemas(self, tmp_path, capsys):
        f = tmp_path / "s.drift"
        f.write_text(TWO_SCHEMA_SRC)
        cmd_schema(_args(str(f)))
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert list(payload.keys()) == ["Grant", "Result"]
        assert payload["Result"] == {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        }

    def test_name_prints_single_schema(self, tmp_path, capsys):
        f = tmp_path / "s.drift"
        f.write_text(TWO_SCHEMA_SRC)
        cmd_schema(_args(str(f), name="Result"))
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert payload == {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        }

    def test_unknown_name_exits_nonzero_with_actionable_message(self, tmp_path, capsys):
        f = tmp_path / "s.drift"
        f.write_text(TWO_SCHEMA_SRC)
        with pytest.raises(SystemExit):
            cmd_schema(_args(str(f), name="DoesNotExist"))
        err = capsys.readouterr().err
        assert "not found" in err
        assert "Grant" in err and "Result" in err

    def test_no_schemas_in_program_exits_nonzero(self, tmp_path, capsys):
        f = tmp_path / "s.drift"
        f.write_text('agent A { step f() -> string { return "x" } }')
        with pytest.raises(SystemExit):
            cmd_schema(_args(str(f)))
        err = capsys.readouterr().err
        assert "No schemas" in err


class TestRunJsonMode:
    """`drift run --json` emits one machine-readable result object instead
    of the human banner/box-drawing output, for a caller (e.g. a sandboxed
    execution wrapper) that has no human reading the terminal."""

    SRC = (
        'schema Result { ok: bool }\n'
        'agent Checker {\n'
        '  step check(g: string) -> Result { respond Result { ok: true } }\n'
        '}\n'
    )

    def test_json_mode_prints_exactly_one_json_object(self, tmp_path, capsys):
        f = tmp_path / "p.drift"
        f.write_text(self.SRC)
        rc = _run_once(_args(str(f), json=True, input='{"g": "hi"}'))
        assert rc == 0
        out = capsys.readouterr().out
        payload = json.loads(out)  # raises if anything but one JSON object was printed
        assert payload["ok"] is True
        assert "cost" in payload and "outputs" in payload

    def test_json_mode_reads_input_from_stdin(self, tmp_path, capsys, monkeypatch):
        f = tmp_path / "p.drift"
        f.write_text(self.SRC)
        monkeypatch.setattr("sys.stdin", io.StringIO('{"g": "hi"}'))
        rc = _run_once(_args(str(f), json=True, input='-'))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["ok"] is True

    def test_json_mode_reports_agent_and_step_on_failure(self, tmp_path, capsys):
        f = tmp_path / "fail.drift"
        f.write_text('agent Failer { step go() { fail "broken" } }')
        rc = _run_once(_args(str(f), json=True))
        assert rc == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["ok"] is False
        assert payload["stage"] == "run"
        assert payload["kind"] == "business-logic"
        assert payload["agent"] == "Failer"
        assert payload["step"] == "go"

    def test_json_mode_reports_compile_error_with_stage(self, tmp_path, capsys):
        f = tmp_path / "bad.drift"
        f.write_text(SCHEDULE_PIPELINE)
        rc = _run_once(_args(str(f), json=True))
        assert rc == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["ok"] is False
        assert payload["stage"] == "codegen"

    def test_json_mode_reports_invalid_input_json(self, tmp_path, capsys):
        f = tmp_path / "p.drift"
        f.write_text(self.SRC)
        rc = _run_once(_args(str(f), json=True, input='not json'))
        assert rc == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["ok"] is False
        assert payload["stage"] == "input"

    def test_json_mode_prints_no_banner_or_box_drawing(self, tmp_path, capsys):
        f = tmp_path / "p.drift"
        f.write_text(self.SRC)
        _run_once(_args(str(f), json=True, input='{"g": "hi"}'))
        out = capsys.readouterr().out
        assert out.count("{") == out.count("}")  # one JSON object, nothing else
        assert "═" not in out
        assert "Transpiled" not in out
