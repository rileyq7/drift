#!/usr/bin/env python3
"""
Drift CLI — Transpile and run .drift files.

Usage:
    drift new <name>                  # Scaffold a new starter project
    drift run <file.drift>            # Transpile + execute
    drift check <file.drift>          # Validate syntax without running
    drift transpile <file.drift>      # Output Python to stdout
    drift transpile <file> -o out.py  # Write Python to file
    drift schema <file.drift>         # Output schema block(s) as JSON Schema
    drift lex <file.drift>            # Show token stream (debug)
    drift parse <file.drift>          # Show AST (debug)
"""

import sys
import os
import json
import time
import asyncio
import argparse
import importlib.util
from pathlib import Path

from drift import __version__
from drift.lexer import lex, LexError
from drift.parser import Parser, ParseError
from drift.codegen import CodeGenerator, CodegenError
from drift.formatter import format_source
from drift.ast_nodes import ImportDecl


# ─── .env auto-loading ──────────────────────────────────────────────────

def _load_env(start: Path) -> int:
    """Walk up from `start` looking for .env. Load matching lines into os.environ.
    Returns the number of variables set (skipping ones already in the environment).
    """
    p = start.resolve()
    for parent in [p.parent, *p.parents]:
        env_path = parent / ".env"
        if env_path.is_file():
            return _apply_env_file(env_path)
        # stop at filesystem root
        if parent == parent.parent:
            break
    return 0


def _apply_env_file(env_path: Path) -> int:
    n = 0
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key or key in os.environ:
            continue
        os.environ[key] = value
        n += 1
    return n


def _provider_name() -> str:
    has_a = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_o = bool(os.environ.get("OPENAI_API_KEY"))
    if has_a and has_o:
        return "anthropic + openai (auto-routed by model)"
    if has_a:
        return "anthropic"
    if has_o:
        return "openai"
    return "mock"


# ─── Source-aware error formatting ──────────────────────────────────────

def _format_source_error(file: str, source: str, line: int, col: int, message: str, kind: str) -> str:
    """Render a friendly compile error with line + caret."""
    lines = source.splitlines()
    src_line = lines[line - 1] if 1 <= line <= len(lines) else ""
    gutter = f" {line} | "
    pad = " " * (len(gutter) - 3) + "| "
    caret = " " * max(col - 1, 0) + "^"
    return (
        f"\n  {kind} error: {message}\n"
        f"    → {file}:{line}:{col}\n\n"
        f"{gutter}{src_line}\n"
        f"{pad}{caret}\n"
    )


def _print_lex_error(file: str, source: str, e: LexError):
    sys.stderr.write(_format_source_error(file, source, e.line, e.col, str(e).split(": ", 1)[-1], "Lex"))


def _print_parse_error(file: str, source: str, e: ParseError):
    tok = e.token
    msg = str(e).split(": ", 1)[-1]
    sys.stderr.write(_format_source_error(file, source, tok.line, tok.col, msg, "Parse"))


def _print_runtime_error(drift_file: str, e: Exception, show_trace: bool = False):
    import traceback
    py_file = str(Path(drift_file).with_suffix('.py'))
    kind = type(e).__name__
    sys.stderr.write(f"\n  ✗ Runtime error ({kind}): {e}\n")

    agent = getattr(e, '_drift_agent', None)
    step = getattr(e, '_drift_step', None)
    if agent and step:
        sys.stderr.write(f"    → step '{step}' of agent '{agent}' in {drift_file}\n")
    elif step:
        sys.stderr.write(f"    → step '{step}' in {drift_file}\n")
    else:
        sys.stderr.write(f"    → in {drift_file}\n")

    tb = traceback.extract_tb(e.__traceback__)
    py_frames = [f for f in tb if f.filename == py_file]
    if py_frames:
        last = py_frames[-1]
        sys.stderr.write(
            f"    → generated {py_file}:{last.lineno}\n        {last.line.strip()}\n"
        )

    if show_trace:
        sys.stderr.write("\n  Traceback (generated Python frames):\n")
        traceback.print_exc()
        return

    sys.stderr.write(
        f"\n  Hint: re-run with --trace for the full Python traceback.\n\n"
    )


def read_source(path: str) -> str:
    if not os.path.exists(path):
        sys.stderr.write(f"\n  ✗ File not found: {path}\n\n")
        sys.exit(1)
    with open(path, 'r') as f:
        return f.read()


# ─── Commands ───────────────────────────────────────────────────────────

def cmd_lex(args):
    source = read_source(args.file)
    try:
        tokens = lex(source)
        for tok in tokens:
            print(tok)
    except LexError as e:
        _print_lex_error(args.file, source, e)
        sys.exit(1)


def cmd_parse(args):
    source = read_source(args.file)
    try:
        tokens = lex(source)
        parser = Parser(tokens)
        program = parser.parse()
        _print_ast(program, indent=0)
    except LexError as e:
        _print_lex_error(args.file, source, e)
        sys.exit(1)
    except ParseError as e:
        _print_parse_error(args.file, source, e)
        sys.exit(1)


def _print_ast(node, indent=0):
    prefix = "  " * indent
    if isinstance(node, list):
        for item in node:
            _print_ast(item, indent)
        return

    name = type(node).__name__
    if hasattr(node, '__dataclass_fields__'):
        print(f"{prefix}{name}:")
        for field_name, field_obj in node.__dataclass_fields__.items():
            val = getattr(node, field_name)
            if isinstance(val, list) and val:
                print(f"{prefix}  {field_name}:")
                for item in val:
                    _print_ast(item, indent + 2)
            elif hasattr(val, '__dataclass_fields__'):
                print(f"{prefix}  {field_name}:")
                _print_ast(val, indent + 2)
            elif isinstance(val, dict) and val:
                print(f"{prefix}  {field_name}:")
                for k, v in val.items():
                    if hasattr(v, '__dataclass_fields__'):
                        print(f"{prefix}    {k}:")
                        _print_ast(v, indent + 3)
                    else:
                        print(f"{prefix}    {k}: {v!r}")
            elif val:
                print(f"{prefix}  {field_name}: {val!r}")
    else:
        print(f"{prefix}{node!r}")


def cmd_fmt(args):
    """Rewrite a .drift file with canonical formatting."""
    source = read_source(args.file)
    try:
        formatted = format_source(source)
    except LexError as e:
        _print_lex_error(args.file, source, e)
        sys.exit(1)

    if args.check:
        if source == formatted:
            print(f"  ✓ {args.file} — already formatted")
            return
        sys.stderr.write(f"  ✗ {args.file} — not formatted (run `drift fmt {args.file}`)\n")
        sys.exit(1)

    if args.stdout:
        sys.stdout.write(formatted)
        return

    if source == formatted:
        print(f"  ✓ {args.file} — unchanged")
        return

    with open(args.file, 'w') as f:
        f.write(formatted)
    print(f"  ✓ formatted {args.file}")


def cmd_check(args):
    """Validate syntax AND that it lowers to Python, without running it.

    Codegen runs (discarding its output) so constructs that parse but can't
    be safely/meaningfully compiled — e.g. `~>`/`|>` pipeline edges,
    `parallel step` — are caught here rather than only at `run`/`transpile`.
    """
    source = read_source(args.file)
    try:
        tokens = lex(source)
        parser = Parser(tokens)
        program = parser.parse()
        CodeGenerator().generate(program)
    except LexError as e:
        _print_lex_error(args.file, source, e)
        sys.exit(1)
    except ParseError as e:
        _print_parse_error(args.file, source, e)
        sys.exit(1)
    except CodegenError as e:
        print(f"  ✗ {args.file} — {e}")
        sys.exit(1)
    print(f"  ✓ {args.file} — syntax OK")


def cmd_schema(args):
    """Render a .drift file's schema block(s) as JSON Schema."""
    source = read_source(args.file)
    from drift.mcp_server import _schema_result
    result = _schema_result(source, args.name)

    if not result["ok"]:
        sys.stderr.write(f"\n  ✗ {args.file} — {result['error']}\n\n")
        sys.exit(1)

    payload = result.get("schema", result.get("schemas"))
    print(json.dumps(payload, indent=2))


def cmd_transpile(args):
    source = read_source(args.file)
    try:
        tokens = lex(source)
        parser = Parser(tokens)
        program = parser.parse()
        codegen = CodeGenerator()
        python_source = codegen.generate(program)
        python_source = python_source.replace(
            "Source: <drift_file>",
            f"Source: {args.file}"
        )
        if args.output:
            with open(args.output, 'w') as f:
                f.write(python_source)
            print(f"  ✓ Transpiled {args.file} → {args.output}")
        else:
            print(python_source)
    except LexError as e:
        _print_lex_error(args.file, source, e)
        sys.exit(1)
    except ParseError as e:
        _print_parse_error(args.file, source, e)
        sys.exit(1)
    except CodegenError as e:
        print(f"  ✗ {args.file} — {e}")
        sys.exit(1)


class _DependencyError(Exception):
    """Raised when a cross-file `.drift` dependency itself fails to lex/
    parse/codegen. The error has already been printed (correctly
    attributed to the dependency's own file/source) by the raise site in
    _transpile_drift_dependencies — this just signals _run_once's caller
    to stop and report failure without re-printing anything, since the
    real LexError/ParseError/CodegenError line/col refer to the
    dependency's source, not the importer's (which _run_once's own
    generic handlers would otherwise incorrectly assume)."""
    def __init__(self, dep_path: Path):
        self.dep_path = dep_path
        super().__init__(f"dependency {dep_path} failed to compile")


def _discover_drift_dependencies(drift_file: Path, seen: set = None) -> set:
    """Lightweight, side-effect-free version of _transpile_drift_dependencies'
    traversal — lex+parse only, no codegen/writing .py files. Used by
    `--watch` mode to know which files' mtimes to poll: the watch loop
    only checked the main file's mtime, so editing a cross-file `import`
    dependency (a shared schema file, say) triggered no re-run at all —
    the watcher would sit there showing stale output until the main file
    itself was also touched. Malformed dependency source is swallowed
    (returns what was found before the error) since a lex/parse failure
    here shouldn't crash the watch loop — the next _run_once call will
    surface it properly as a real error.
    """
    if seen is None:
        seen = set()
    base_dir = drift_file.resolve().parent
    try:
        program = Parser(lex(drift_file.read_text())).parse()
    except Exception:
        return seen
    for decl in program.declarations:
        if not (isinstance(decl, ImportDecl) and decl.source_path.endswith(".drift")):
            continue
        dep_path = (base_dir / decl.source_path).resolve()
        if dep_path in seen or not dep_path.exists():
            continue
        seen.add(dep_path)
        _discover_drift_dependencies(dep_path, seen)
    return seen


def _transpile_drift_dependencies(program, drift_file: Path, seen: set) -> set:
    """Recursively transpile `.drift` files referenced via `import { X }
    from "./other.drift"` so their sibling .py exists before the importer
    runs. gen_import resolves cross-file imports as plain sibling-file
    Python imports (`from other import X`) with no runtime auto-transpile
    of its own — without this, `drift run` on a file with a cross-file
    import failed with ModuleNotFoundError even though nothing in LLM.md's
    own `import` example suggests a manual pre-transpile step is needed.
    `seen` (a set of resolved absolute paths) guards against re-transpiling
    the same file twice in a diamond-shaped import graph and against
    infinite recursion on a circular import.

    Returns the set of directories (as resolved Paths) each transpiled
    dependency lives in — gen_import strips the import path down to just
    the target file's basename (`from ./agents/checker.drift` becomes
    `from checker import X`, dropping the `agents/` prefix entirely, per
    gen_import's own docstring), so a subdirectory-organized import (LLM.md
    §14's own second example: `import GrantChecker from
    "./agents/checker.drift"`) only resolves at runtime if that
    subdirectory is itself on sys.path — not just the main file's own
    directory. The caller adds every returned directory to sys.path.
    """
    dirs: set = set()
    base_dir = drift_file.resolve().parent
    for decl in program.declarations:
        if not (isinstance(decl, ImportDecl) and decl.source_path.endswith(".drift")):
            continue
        dep_path = (base_dir / decl.source_path).resolve()
        if dep_path in seen:
            continue
        if not dep_path.exists():
            continue
        seen.add(dep_path)
        dirs.add(dep_path.parent)
        dep_source = dep_path.read_text()
        # A lex/parse/codegen error HERE is in the DEPENDENCY's source,
        # not the importer's — letting it propagate uncaught would hit
        # _run_once's own except LexError/ParseError/CodegenError blocks,
        # which unconditionally print using args.file (the importer) and
        # the IMPORTER's source text, while the exception's line/col
        # refer to positions in the DEPENDENCY's text. That produces a
        # genuinely misleading error: the importer's path and content
        # with a caret pointing at unrelated characters, since the two
        # source strings have nothing to do with each other. Catch and
        # print with the correct (dependency) file/source right here.
        try:
            dep_tokens = lex(dep_source)
            dep_program = Parser(dep_tokens).parse()
        except LexError as e:
            _print_lex_error(str(dep_path), dep_source, e)
            raise _DependencyError(dep_path) from e
        except ParseError as e:
            _print_parse_error(str(dep_path), dep_source, e)
            raise _DependencyError(dep_path) from e
        # Recurse first so transitive dependencies exist before this one
        # is written (matters if this file's own codegen were ever made
        # dependency-aware; harmless no-op today since gen_import doesn't
        # validate against the target's actual exports).
        dirs |= _transpile_drift_dependencies(dep_program, dep_path, seen)
        try:
            dep_python = CodeGenerator().generate(dep_program).replace(
                "Source: <drift_file>", f"Source: {dep_path}"
            )
        except CodegenError as e:
            print(f"  ✗ {dep_path} — {e}")
            raise _DependencyError(dep_path) from e
        dep_py_path = dep_path.with_suffix('.py')
        dep_py_path.write_text(dep_python)
        print(f"  ✓ Transpiled dependency → {dep_py_path}")
    return dirs


def _read_input(raw: str | None) -> str | None:
    """Resolve --input's value: '-' reads a JSON blob from stdin (so a
    sandbox wrapper can pipe input without shell-escaping large/quoted
    JSON into an argv string), anything else is returned as-is (a literal
    JSON string, matching --input's existing behavior)."""
    if raw == '-':
        return sys.stdin.read()
    return raw


def _run_once(args):
    """Transpile + execute a single time. Returns 0 on success, nonzero on failure."""
    if getattr(args, 'json', False):
        return _run_once_json(args)
    source = read_source(args.file)
    try:
        tokens = lex(source)
        parser = Parser(tokens)
        program = parser.parse()
        seen_deps = set()  # mutated in-place by _transpile_drift_dependencies
        dep_dirs = _transpile_drift_dependencies(program, Path(args.file), seen_deps)
        codegen = CodeGenerator()
        python_source = codegen.generate(program)
        python_source = python_source.replace(
            "Source: <drift_file>",
            f"Source: {args.file}"
        )

        py_path = Path(args.file).with_suffix('.py')
        with open(py_path, 'w') as f:
            f.write(python_source)
        print(f"  ✓ Transpiled → {py_path}")

        # Force a re-import each watch iteration. Dependency modules
        # (from a cross-file `import`) need the same treatment as the
        # main module: `from schema_dep import Shared` caches
        # sys.modules['schema_dep'] process-wide the first time it's
        # imported, so a later run — in --watch mode, or any other
        # scenario where this process imports a DIFFERENT file that also
        # happens to depend on a same-named sibling — would silently get
        # the stale cached module instead of the current one, even though
        # its .py was freshly re-transpiled above.
        sys.modules.pop('drift_generated', None)
        for dep_path in seen_deps:
            sys.modules.pop(dep_path.stem, None)
        spec = importlib.util.spec_from_file_location("drift_generated", py_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules['drift_generated'] = module
        # A cross-file `import { X } from "./other.drift"` codegens to a
        # plain `from other import X` (sibling-file resolution — see
        # gen_import's docstring, which strips any subdirectory prefix
        # down to just the target's basename), which only resolves if the
        # sibling's transpiled .py is importable. Python's default import
        # search uses the CWD/sys.path[0], not the .drift file's own
        # directory (or a dependency's own subdirectory) — so `drift run
        # some/dir/file.drift` from elsewhere raised ModuleNotFoundError
        # even with the sibling already transpiled, and a
        # subdirectory-organized import (LLM.md §14's own second example,
        # `./agents/checker.drift`) would still fail even from the right
        # CWD. Add the .drift file's own directory plus every dependency's
        # directory so imports resolve regardless of the caller's CWD or
        # how dependencies are organized into subdirectories, matching how
        # --input/.env resolution already doesn't require the caller to
        # `cd` first.
        candidate_dirs = {py_path.resolve().parent} | dep_dirs
        added_dirs = [str(d) for d in candidate_dirs if str(d) not in sys.path]
        sys.path[0:0] = added_dirs
        try:
            spec.loader.exec_module(module)
        finally:
            for d in added_dirs:
                sys.path.remove(d)

        from drift.runtime.core import Agent, run_agent, first_declared
        agents = {}
        pipelines = {}
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and issubclass(obj, Agent) and obj is not Agent:
                agents[name] = obj
            # Pipelines are plain classes (not Agent subclasses) with an
            # async `run` method. Detect by shape.
            elif isinstance(obj, type) and not name.startswith('_') and hasattr(obj, 'run'):
                if obj is Agent:
                    continue
                run_attr = getattr(obj, 'run', None)
                if run_attr is not None and asyncio.iscoroutinefunction(run_attr):
                    pipelines[name] = obj

        if args.pipeline:
            pipe_cls = pipelines.get(args.pipeline)
            if not pipe_cls:
                avail = ', '.join(pipelines) if pipelines else '(none)'
                sys.stderr.write(
                    f"\n  ✗ Pipeline '{args.pipeline}' not found. Available: {avail}\n\n"
                )
                return 1
            raw_input = _read_input(args.input)
            initial = json.loads(raw_input) if raw_input else None
            asyncio.run(pipe_cls().run(initial_input=initial))
            return 0

        if not agents:
            if pipelines:
                sys.stderr.write(
                    f"\n  ✗ No agents found, but pipelines are: {', '.join(pipelines)}.\n"
                    f"    Re-run with --pipeline <name>.\n\n"
                )
            else:
                sys.stderr.write("\n  ✗ No agents found in the generated code.\n\n")
            return 1

        if args.agent:
            agent_cls = agents.get(args.agent)
            if not agent_cls:
                sys.stderr.write(
                    f"\n  ✗ Agent '{args.agent}' not found. Available: {', '.join(agents.keys())}\n\n"
                )
                return 1
        else:
            # LLM.md documents "drift run (no --agent/--pipeline) runs the
            # first agent's first step" — but `agents` was built by
            # iterating dir(module), which returns names ALPHABETICALLY,
            # not in declaration order. Whichever agent's class name
            # happened to sort first silently won regardless of source
            # order, with no indication the "wrong" agent was chosen.
            # Recover declaration order the same way run_agent() already
            # does for STEP selection within an agent (co_firstlineno):
            # every generated agent class has a real __init__, whose code
            # object's line number reflects source order.
            agent_cls = first_declared(agents.values())

        raw_input = _read_input(args.input)
        inputs = json.loads(raw_input) if raw_input else {}

        asyncio.run(run_agent(agent_cls, step_name=args.step, inputs=inputs))
        return 0

    except _DependencyError:
        # Already printed with the correct (dependency) file/source by
        # the raise site — must be caught before the generic handlers
        # below, which would otherwise mis-attribute it to args.file
        # (the importer) using the IMPORTER's source text while the
        # underlying LexError/ParseError's line/col refer to the
        # dependency's text, producing a caret pointing at unrelated
        # characters in the wrong file.
        return 1
    except LexError as e:
        _print_lex_error(args.file, source, e)
        return 1
    except ParseError as e:
        _print_parse_error(args.file, source, e)
        return 1
    except CodegenError as e:
        # Distinct from _print_runtime_error: nothing ran, so there's no
        # generated-code frame or cost banner to show — this is a rejected
        # compile, not a failed execution.
        print(f"  ✗ {args.file} — {e}")
        return 1
    except Exception as e:
        _print_runtime_error(args.file, e, show_trace=args.trace)
        return 1


def _run_once_json(args) -> int:
    """`--json` entrypoint for `drift run`: a stable stdin-in/stdout-out
    contract for a sandbox wrapper that has no human reading the terminal —
    every human-oriented print() (transpile confirmation, box-drawing
    banner, cost summary text) is captured and discarded, and exactly one
    JSON object (build_run_outcome's shape, plus a `stage` field on
    compile-time failures to match the MCP server's convention) is written
    to stdout. Prints nothing else and never raises — errors become
    {ok: false, ...} like every other stage of this pipeline.
    """
    import io
    if not os.path.exists(args.file):
        json.dump({"ok": False, "stage": "read", "error": f"File not found: {args.file}"}, sys.stdout)
        return 1
    with open(args.file, 'r') as f:
        source = f.read()

    old_out = sys.stdout
    sys.stdout = io.StringIO()
    try:
        tokens = lex(source)
        parser = Parser(tokens)
        program = parser.parse()
        seen_deps = set()
        dep_dirs = _transpile_drift_dependencies(program, Path(args.file), seen_deps)
        codegen = CodeGenerator()
        python_source = codegen.generate(program)
        python_source = python_source.replace("Source: <drift_file>", f"Source: {args.file}")
    except LexError as e:
        sys.stdout = old_out
        return _emit_json({"ok": False, "stage": "lex", "error": str(e)})
    except ParseError as e:
        sys.stdout = old_out
        return _emit_json({"ok": False, "stage": "parse", "error": str(e)})
    except CodegenError as e:
        sys.stdout = old_out
        return _emit_json({"ok": False, "stage": "codegen", "error": str(e)})
    except _DependencyError as e:
        sys.stdout = old_out
        return _emit_json({"ok": False, "stage": "import", "error": f"dependency {e.dep_path} failed to compile"})

    try:
        py_path = Path(args.file).with_suffix('.py')
        with open(py_path, 'w') as f:
            f.write(python_source)

        sys.modules.pop('drift_generated', None)
        for dep_path in seen_deps:
            sys.modules.pop(dep_path.stem, None)
        spec = importlib.util.spec_from_file_location("drift_generated", py_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules['drift_generated'] = module
        candidate_dirs = {py_path.resolve().parent} | dep_dirs
        added_dirs = [str(d) for d in candidate_dirs if str(d) not in sys.path]
        sys.path[0:0] = added_dirs
        try:
            spec.loader.exec_module(module)
        finally:
            for d in added_dirs:
                sys.path.remove(d)
    except Exception as e:
        sys.stdout = old_out
        return _emit_json({"ok": False, "stage": "import", "error": f"{type(e).__name__}: {e}"})

    from drift.runtime.core import Agent, run_agent, first_declared, build_run_outcome
    agents = {}
    pipelines = {}
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and issubclass(obj, Agent) and obj is not Agent:
            agents[name] = obj
        elif isinstance(obj, type) and not name.startswith('_') and hasattr(obj, 'run'):
            if obj is Agent:
                continue
            run_attr = getattr(obj, 'run', None)
            if run_attr is not None and asyncio.iscoroutinefunction(run_attr):
                pipelines[name] = obj

    raw_input = _read_input(args.input)
    try:
        parsed_input = json.loads(raw_input) if raw_input else None
    except json.JSONDecodeError as e:
        sys.stdout = old_out
        return _emit_json({"ok": False, "stage": "input", "error": f"invalid input JSON: {e}"})

    if args.pipeline:
        pipe_cls = pipelines.get(args.pipeline)
        if not pipe_cls:
            avail = ', '.join(pipelines) if pipelines else '(none)'
            sys.stdout = old_out
            return _emit_json({"ok": False, "stage": "discover", "error": f"Pipeline {args.pipeline!r} not found. Available: {avail}"})
        try:
            result = asyncio.run(pipe_cls().run(initial_input=parsed_input))
        except Exception as e:
            sys.stdout = old_out
            return _emit_json({"ok": False, "stage": "run", "kind": "bug", "error": f"{type(e).__name__}: {e}"})
        sys.stdout = old_out
        from drift.runtime.core import _to_jsonable
        return _emit_json({"ok": True, "result": _to_jsonable(result)})

    if not agents:
        sys.stdout = old_out
        if pipelines:
            return _emit_json({
                "ok": False, "stage": "discover",
                "error": f"No agents found, but pipelines are: {', '.join(pipelines)}. Pass --pipeline <name>.",
            })
        return _emit_json({"ok": False, "stage": "discover", "error": "No agents found in the generated code."})

    if args.agent:
        agent_cls = agents.get(args.agent)
        if not agent_cls:
            sys.stdout = old_out
            return _emit_json({"ok": False, "stage": "discover", "error": f"Agent {args.agent!r} not found. Available: {', '.join(agents.keys())}"})
    else:
        agent_cls = first_declared(agents.values())

    inputs = parsed_input if parsed_input is not None else {}
    cost: dict = {}
    try:
        result = asyncio.run(run_agent(agent_cls, step_name=args.step, inputs=inputs, cost_out=cost))
    except Exception as e:
        sys.stdout = old_out
        return _emit_json(build_run_outcome(error=e, cost=getattr(e, "_drift_cost", cost or None)))

    sys.stdout = old_out
    return _emit_json(build_run_outcome(result=result, cost=cost))


def _emit_json(outcome: dict) -> int:
    print(json.dumps(outcome, indent=2, default=str))
    return 0 if outcome.get("ok") else 1


def cmd_run(args):
    n_loaded = _load_env(Path(args.file))

    if args.json:
        if args.watch:
            sys.stderr.write("\n  ✗ --json and --watch cannot be combined.\n\n")
            sys.exit(1)
        rc = _run_once(args)
        sys.exit(rc)

    provider = _provider_name()
    banner = f"  ▸ provider: {provider}"
    if n_loaded:
        banner += f"  ·  loaded {n_loaded} var{'s' if n_loaded != 1 else ''} from .env"
    print(banner)

    if not args.watch:
        rc = _run_once(args)
        sys.exit(rc)

    # Watch mode: poll mtime, re-run on change. Watches the main file AND
    # every `.drift` file it (transitively) imports — watching only the
    # main file meant editing a shared/imported schema triggered no
    # re-run at all, so the watcher would keep showing stale output
    # until the main file itself was also touched. Re-discovers the
    # dependency set after every re-run so adding/removing an `import`
    # (or a dependency's own dependency changing) is picked up too.
    path = Path(args.file)
    watched_mtimes: dict = {}
    print(f"  ▸ watching {args.file} — Ctrl-C to exit")
    try:
        while True:
            watched_paths = {path} | _discover_drift_dependencies(path)
            changed = False
            for p in watched_paths:
                try:
                    mtime = p.stat().st_mtime
                except FileNotFoundError:
                    continue
                if watched_mtimes.get(p) != mtime:
                    changed = True
                watched_mtimes[p] = mtime
            if changed:
                print("\n" + "─" * 50)
                _run_once(args)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n  ▸ stopped")
        sys.exit(0)


def cmd_mcp(args):
    from drift.mcp_server import serve_stdio
    serve_stdio()


_DEFAULT_MODELS = {
    "openai": "gpt-5.4-nano",
    "anthropic": "claude-haiku-4-5",
    "mock": "gpt-5.4-nano",
}


def _pick_starter_model(args) -> str:
    """Pick the model for `drift new`, in priority order:
      1. --model flag (explicit override)
      2. DRIFT_DEFAULT_MODEL env var
      3. Whichever provider key is set in the environment
      4. Interactive prompt (if stdin is a TTY)
      5. Safe default (gpt-5.4-nano + mock fallback)
    """
    if args.model:
        return args.model

    env_default = os.environ.get("DRIFT_DEFAULT_MODEL")
    if env_default:
        return env_default

    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))

    if has_openai and not has_anthropic:
        print(f"  ▸ detected OPENAI_API_KEY — using {_DEFAULT_MODELS['openai']}")
        return _DEFAULT_MODELS["openai"]
    if has_anthropic and not has_openai:
        print(f"  ▸ detected ANTHROPIC_API_KEY — using {_DEFAULT_MODELS['anthropic']}")
        return _DEFAULT_MODELS["anthropic"]

    # Both keys set, or neither. Ask if we can.
    if sys.stdin.isatty():
        print()
        print("  Which model should this agent use?")
        print(f"    1) OpenAI       ({_DEFAULT_MODELS['openai']})")
        print(f"    2) Anthropic    ({_DEFAULT_MODELS['anthropic']})")
        print("    3) Custom       (type model name)")
        print()
        choice = input("  Choice [1]: ").strip() or "1"
        if choice == "1":
            return _DEFAULT_MODELS["openai"]
        if choice == "2":
            return _DEFAULT_MODELS["anthropic"]
        if choice == "3":
            custom = input("  Model name: ").strip()
            return custom or _DEFAULT_MODELS["openai"]
        # Anything else: fall through.

    return _DEFAULT_MODELS["openai"]


def cmd_new(args):
    """Scaffold a new starter project."""
    name = args.name
    if not name.replace("_", "").replace("-", "").isalnum():
        sys.stderr.write(f"\n  ✗ Invalid name: '{name}' — use letters, digits, _ or -.\n\n")
        sys.exit(1)

    target_dir = Path(name)
    if target_dir.exists() and any(target_dir.iterdir()):
        sys.stderr.write(f"\n  ✗ Directory '{name}' already exists and is not empty.\n\n")
        sys.exit(1)
    target_dir.mkdir(exist_ok=True)

    pascal = "".join(p.capitalize() for p in name.replace("-", "_").split("_"))
    model = _pick_starter_model(args)

    tpl_dir = Path(__file__).parent / "templates"
    starter_src = (tpl_dir / "starter.drift").read_text()
    env_src = (tpl_dir / "env.example").read_text()

    drift_file = target_dir / f"{name}.drift"
    drift_file.write_text(starter_src.format(name=name, Name=pascal, model=model))
    (target_dir / ".env.example").write_text(env_src)

    print(f"  ✓ Created {drift_file}")
    print(f"  ✓ Created {target_dir / '.env.example'}")
    print()
    print("  Next steps:")
    print(f"    cd {name}")
    print(f"    drift run {name}.drift --input '{{\"name\":\"Riley\"}}'")
    print()
    print("  (No API key set? Drift uses a mock provider and still runs.)")


# ─── Entrypoint ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog='drift',
        description='Drift — An intent-based language for agentic systems',
    )
    parser.add_argument('--version', action='version', version=f'drift {__version__}')
    subparsers = parser.add_subparsers(dest='command', required=True)

    p_new = subparsers.add_parser('new', help='Scaffold a new starter project')
    p_new.add_argument('name', help='Project name (creates a directory of this name)')
    p_new.add_argument('--model', help='Default model for the starter agent (e.g. gpt-5.4-nano, claude-haiku-4-5)')
    p_new.set_defaults(func=cmd_new)

    p_run = subparsers.add_parser('run', help='Transpile and execute a .drift file')
    p_run.add_argument('file', help='Path to .drift file')
    p_run.add_argument('--step', help='Specific step to run')
    p_run.add_argument('--agent', help='Specific agent to run')
    p_run.add_argument('--input', help="JSON input string, or '-' to read JSON from stdin")
    p_run.add_argument('--pipeline', help='Run a declared pipeline by name (instead of an agent)')
    p_run.add_argument('--trace', action='store_true', help='Show full Python traceback on runtime errors')
    p_run.add_argument('--watch', action='store_true', help='Re-run when the .drift file changes')
    p_run.add_argument('--json', action='store_true',
                        help='Emit one JSON result object to stdout instead of human-readable output')
    p_run.set_defaults(func=cmd_run)

    p_check = subparsers.add_parser('check', help='Validate syntax without running')
    p_check.add_argument('file', help='Path to .drift file')
    p_check.set_defaults(func=cmd_check)

    p_fmt = subparsers.add_parser('fmt', help='Format a .drift file in place')
    p_fmt.add_argument('file', help='Path to .drift file')
    p_fmt.add_argument('--check', action='store_true', help='Exit 1 if file is not formatted (CI mode)')
    p_fmt.add_argument('--stdout', action='store_true', help='Write to stdout instead of rewriting the file')
    p_fmt.set_defaults(func=cmd_fmt)

    p_trans = subparsers.add_parser('transpile', help='Transpile .drift to Python')
    p_trans.add_argument('file', help='Path to .drift file')
    p_trans.add_argument('-o', '--output', help='Output .py file path')
    p_trans.set_defaults(func=cmd_transpile)

    p_schema = subparsers.add_parser('schema', help="Render a .drift file's schema block(s) as JSON Schema")
    p_schema.add_argument('file', help='Path to .drift file')
    p_schema.add_argument('--name', help='Render only this schema (default: all)')
    p_schema.set_defaults(func=cmd_schema)

    p_lex = subparsers.add_parser('lex', help='Show token stream (debug)')
    p_lex.add_argument('file', help='Path to .drift file')
    p_lex.set_defaults(func=cmd_lex)

    p_parse = subparsers.add_parser('parse', help='Show AST (debug)')
    p_parse.add_argument('file', help='Path to .drift file')
    p_parse.set_defaults(func=cmd_parse)

    p_mcp = subparsers.add_parser('mcp', help='Run as an MCP stdio server (drift_check / drift_transpile / drift_run)')
    p_mcp.set_defaults(func=cmd_mcp)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
