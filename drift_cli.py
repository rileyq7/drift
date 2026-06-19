#!/usr/bin/env python3
"""
Drift CLI — Transpile and run .drift files.

Usage:
    python drift_cli.py transpile <file.drift>              # Output Python to stdout
    python drift_cli.py transpile <file.drift> -o out.py    # Write Python to file
    python drift_cli.py run <file.drift>                    # Transpile + execute
    python drift_cli.py run <file.drift> --step check       # Run specific step
    python drift_cli.py lex <file.drift>                    # Show token stream (debug)
    python drift_cli.py parse <file.drift>                  # Show AST (debug)
"""

import sys
import os
import json
import asyncio
import argparse
import importlib.util
import tempfile
from pathlib import Path

# Add parent dir to path so drift package is importable
sys.path.insert(0, str(Path(__file__).parent))

from drift.lexer import lex, LexError
from drift.parser import Parser, ParseError
from drift.codegen import CodeGenerator


def read_source(path: str) -> str:
    with open(path, 'r') as f:
        return f.read()


def cmd_lex(args):
    """Show the token stream — useful for debugging the lexer."""
    source = read_source(args.file)
    try:
        tokens = lex(source)
        for tok in tokens:
            print(tok)
    except LexError as e:
        print(f"Lex error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_parse(args):
    """Show the AST — useful for debugging the parser."""
    source = read_source(args.file)
    try:
        tokens = lex(source)
        parser = Parser(tokens)
        program = parser.parse()
        _print_ast(program, indent=0)
    except (LexError, ParseError) as e:
        print(f"Parse error: {e}", file=sys.stderr)
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
            elif val:  # skip empty/falsy
                print(f"{prefix}  {field_name}: {val!r}")
    else:
        print(f"{prefix}{node!r}")


def cmd_transpile(args):
    """Transpile a .drift file to Python."""
    source = read_source(args.file)

    try:
        # Lex
        tokens = lex(source)

        # Parse
        parser = Parser(tokens)
        program = parser.parse()

        # Generate Python
        codegen = CodeGenerator()
        python_source = codegen.generate(program)

        # Update source reference in header
        python_source = python_source.replace(
            "Source: <drift_file>",
            f"Source: {args.file}"
        )

        if args.output:
            with open(args.output, 'w') as f:
                f.write(python_source)
            print(f"✓ Transpiled {args.file} → {args.output}")
        else:
            print(python_source)

    except LexError as e:
        print(f"\n  ✗ Lex error: {e}", file=sys.stderr)
        sys.exit(1)
    except ParseError as e:
        print(f"\n  ✗ Parse error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_run(args):
    """Transpile and execute a .drift file."""
    source = read_source(args.file)

    try:
        # Transpile
        tokens = lex(source)
        parser = Parser(tokens)
        program = parser.parse()
        codegen = CodeGenerator()
        python_source = codegen.generate(program)
        python_source = python_source.replace(
            "Source: <drift_file>",
            f"Source: {args.file}"
        )

        # Also write the .py file so user can inspect it
        py_path = Path(args.file).with_suffix('.py')
        with open(py_path, 'w') as f:
            f.write(python_source)
        print(f"  ✓ Transpiled → {py_path}")

        # Load the generated module
        spec = importlib.util.spec_from_file_location("drift_generated", py_path)
        module = importlib.util.module_from_spec(spec)

        # Make drift package importable from the generated code
        sys.modules['drift_generated'] = module
        spec.loader.exec_module(module)

        # Find agent classes
        from drift.runtime.core import Agent, run_agent
        agents = {}
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and issubclass(obj, Agent) and obj is not Agent:
                agents[name] = obj

        if not agents:
            print("  ✗ No agents found in the generated code.", file=sys.stderr)
            sys.exit(1)

        # Pick the agent to run
        if args.agent:
            agent_cls = agents.get(args.agent)
            if not agent_cls:
                print(f"  ✗ Agent '{args.agent}' not found. Available: {', '.join(agents.keys())}")
                sys.exit(1)
        else:
            agent_cls = list(agents.values())[0]

        # Parse input JSON
        inputs = {}
        if args.input:
            inputs = json.loads(args.input)

        # Run it
        asyncio.run(run_agent(agent_cls, step_name=args.step, inputs=inputs))

    except LexError as e:
        print(f"\n  ✗ Lex error: {e}", file=sys.stderr)
        sys.exit(1)
    except ParseError as e:
        print(f"\n  ✗ Parse error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n  ✗ Runtime error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog='drift',
        description='Drift — An intent-based language for agentic systems',
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # drift transpile
    p_trans = subparsers.add_parser('transpile', help='Transpile .drift to Python')
    p_trans.add_argument('file', help='Path to .drift file')
    p_trans.add_argument('-o', '--output', help='Output .py file path')
    p_trans.set_defaults(func=cmd_transpile)

    # drift run
    p_run = subparsers.add_parser('run', help='Transpile and execute a .drift file')
    p_run.add_argument('file', help='Path to .drift file')
    p_run.add_argument('--step', help='Specific step to run')
    p_run.add_argument('--agent', help='Specific agent to run')
    p_run.add_argument('--input', help='JSON input string')
    p_run.set_defaults(func=cmd_run)

    # drift lex
    p_lex = subparsers.add_parser('lex', help='Show token stream (debug)')
    p_lex.add_argument('file', help='Path to .drift file')
    p_lex.set_defaults(func=cmd_lex)

    # drift parse
    p_parse = subparsers.add_parser('parse', help='Show AST (debug)')
    p_parse.add_argument('file', help='Path to .drift file')
    p_parse.set_defaults(func=cmd_parse)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
