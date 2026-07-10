#!/usr/bin/env python3
"""
Drift CLI — Transpile and run .drift files.

Usage:
    drift new <name>                  # Scaffold a new starter project
    drift run <file.drift>            # Transpile + execute
    drift check <file.drift>          # Validate syntax without running
    drift transpile <file.drift>      # Output Python to stdout
    drift transpile <file> -o out.py  # Write Python to file
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


def _run_once(args):
    """Transpile + execute a single time. Returns 0 on success, nonzero on failure."""
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

        py_path = Path(args.file).with_suffix('.py')
        with open(py_path, 'w') as f:
            f.write(python_source)
        print(f"  ✓ Transpiled → {py_path}")

        # Force a re-import each watch iteration.
        sys.modules.pop('drift_generated', None)
        spec = importlib.util.spec_from_file_location("drift_generated", py_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules['drift_generated'] = module
        spec.loader.exec_module(module)

        from drift.runtime.core import Agent, run_agent
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
            initial = json.loads(args.input) if args.input else None
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
            agent_cls = list(agents.values())[0]

        inputs = {}
        if args.input:
            inputs = json.loads(args.input)

        asyncio.run(run_agent(agent_cls, step_name=args.step, inputs=inputs))
        return 0

    except LexError as e:
        _print_lex_error(args.file, source, e)
        return 1
    except ParseError as e:
        _print_parse_error(args.file, source, e)
        return 1
    except Exception as e:
        _print_runtime_error(args.file, e, show_trace=args.trace)
        return 1


def cmd_run(args):
    # Load .env from the .drift file's directory tree.
    n_loaded = _load_env(Path(args.file))
    provider = _provider_name()
    banner = f"  ▸ provider: {provider}"
    if n_loaded:
        banner += f"  ·  loaded {n_loaded} var{'s' if n_loaded != 1 else ''} from .env"
    print(banner)

    if not args.watch:
        rc = _run_once(args)
        sys.exit(rc)

    # Watch mode: poll mtime, re-run on change.
    path = Path(args.file)
    last_mtime = 0.0
    print(f"  ▸ watching {args.file} — Ctrl-C to exit")
    try:
        while True:
            try:
                mtime = path.stat().st_mtime
            except FileNotFoundError:
                time.sleep(0.5)
                continue
            if mtime != last_mtime:
                last_mtime = mtime
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
    p_run.add_argument('--input', help='JSON input string')
    p_run.add_argument('--pipeline', help='Run a declared pipeline by name (instead of an agent)')
    p_run.add_argument('--trace', action='store_true', help='Show full Python traceback on runtime errors')
    p_run.add_argument('--watch', action='store_true', help='Re-run when the .drift file changes')
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
