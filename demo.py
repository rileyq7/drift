#!/usr/bin/env python3
"""
Drift End-to-End Demo

This script demonstrates the complete pipeline:
  1. Read a .drift source file
  2. Lex it into tokens
  3. Parse tokens into an AST
  4. Generate Python from the AST
  5. Execute the generated Python

Run: python demo.py
"""

import sys
import os
import asyncio
import importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from drift.lexer import lex
from drift.parser import Parser
from drift.codegen import CodeGenerator


DIVIDER = "─" * 60

def section(title):
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}\n")


async def demo():
    # ── Step 1: Read the source ──
    # These are the golden examples covered by tests/golden/test_transpile_examples.py
    # (EXAMPLE_NAMES). Keep this list in sync so `python demo.py` regenerates
    # every committed .py golden.
    drift_files = [
        "examples/hello.drift",
        "examples/inbox_sorter.drift",
        "examples/grant_checker.drift",
        "examples/confident_demo.drift",
        "examples/inbox_triage_live.drift",
        "examples/grant_checker_compare.drift",
    ]

    for drift_file in drift_files:
        if not os.path.exists(drift_file):
            continue

        section(f"TRANSPILING: {drift_file}")

        with open(drift_file) as f:
            source = f.read()

        print("Source (.drift):")
        print(DIVIDER)
        for i, line in enumerate(source.split('\n'), 1):
            print(f"  {i:3}  {line}")
        print(DIVIDER)

        # ── Step 2: Lex ──
        tokens = lex(source)
        print(f"\n  Lexer: {len(tokens)} tokens generated")

        # ── Step 3: Parse ──
        parser = Parser(tokens)
        program = parser.parse()
        decl_types = [type(d).__name__ for d in program.declarations]
        print(f"  Parser: {len(program.declarations)} declarations → {', '.join(decl_types)}")

        # ── Step 4: Generate Python ──
        codegen = CodeGenerator()
        python_source = codegen.generate(program)
        python_source = python_source.replace("Source: <drift_file>", f"Source: {drift_file}")

        py_path = drift_file.replace('.drift', '.py')
        with open(py_path, 'w') as f:
            f.write(python_source)

        print(f"  Codegen: {len(python_source.splitlines())} lines of Python")
        print(f"  Output: {py_path}")

        print(f"\nGenerated Python:")
        print(DIVIDER)
        for i, line in enumerate(python_source.split('\n'), 1):
            print(f"  {i:3}  {line}")
        print(DIVIDER)

    # ── Step 5: Execute the grant checker ──
    section("EXECUTING: grant_checker.drift")

    py_path = "examples/grant_checker.py"
    if os.path.exists(py_path):
        spec = importlib.util.spec_from_file_location("drift_generated", py_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules['drift_generated'] = module
        spec.loader.exec_module(module)

        from drift.runtime.core import Agent, run_agent

        # Find the GrantChecker agent
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and issubclass(obj, Agent) and obj is not Agent:
                print(f"  Found agent: {name}")

                result = await run_agent(
                    obj,
                    step_name='evaluate',
                    inputs={
                        'company_profile': (
                            'TechCo Ltd is a London-based AI startup with 15 employees. '
                            'They specialise in NLP and machine learning for healthcare, '
                            'specifically drug discovery and clinical trial optimisation. '
                            'Annual revenue £800k, founded 2021.'
                        ),
                        'call_text': (
                            'Innovate UK Smart Grants: Open to UK-registered SMEs. '
                            'Funding range £25k-£500k for disruptive R&D innovations. '
                            'Focus areas: AI, machine learning, health technology, '
                            'and digital transformation. Companies must have fewer than '
                            '250 employees and demonstrate clear path to commercialisation.'
                        ),
                    }
                )
                break


def main():
    section("DRIFT LANGUAGE — END-TO-END DEMO")
    print("  This demonstrates the complete transpiler pipeline:")
    print("  .drift source → lexer → parser → AST → codegen → Python → execution")
    print()
    print("  The transpiler is a deterministic Python program.")
    print("  No AI is involved in the translation.")
    print("  AI only enters at runtime, when the generated code calls an LLM.")

    asyncio.run(demo())

    section("DONE")
    print("  All example .drift files transpiled to .py files.")
    print("  One agent executed end-to-end with cost tracking.")
    print()
    print("  Next steps:")
    print("    • Set ANTHROPIC_API_KEY for real LLM calls")
    print("    • Try: drift transpile examples/hello.drift")
    print("    • Try: drift run examples/grant_checker.drift --step evaluate")
    print("      (or, without installing: python -m drift.cli transpile examples/hello.drift)")
    print()


if __name__ == '__main__':
    main()
