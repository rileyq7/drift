"""Golden tests: each .drift file in examples/ must transpile to exactly the
committed .py file. If you intentionally change codegen, regenerate the
golden files with:

    python demo.py

and commit the updated .py files.

A failing test here means either:
  (a) you changed codegen and forgot to update the golden, or
  (b) a parser/lexer change silently broke the example.
"""
from pathlib import Path

import pytest


EXAMPLE_NAMES = [
    "hello", "inbox_sorter", "grant_checker", "confident_demo",
    "inbox_triage_live", "grant_checker_compare",
]


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_example_transpiles_to_committed_python(name, examples_dir, transpile):
    drift_path = examples_dir / f"{name}.drift"
    py_path = examples_dir / f"{name}.py"

    source = drift_path.read_text()
    generated = transpile(source)
    # The codegen header includes a Source: placeholder that demo.py rewrites.
    # Match the same substitution here so the comparison is fair.
    generated = generated.replace("Source: <drift_file>", f"Source: examples/{name}.drift")

    expected = py_path.read_text().rstrip("\n")
    actual = generated.rstrip("\n")

    if actual != expected:
        # Surface a tight diff rather than dumping both blobs.
        import difflib
        diff = "\n".join(difflib.unified_diff(
            expected.splitlines(),
            actual.splitlines(),
            fromfile=f"golden/{name}.py",
            tofile=f"generated/{name}.py",
            lineterm="",
        ))
        pytest.fail(f"Transpile output drifted from golden:\n{diff}")
