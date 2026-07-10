"""Regression tests for code injection through generated Python.

A .drift file transpiles to Python source. User-controlled string content must
never be able to break out of its literal and execute as code, and string
interpolation must not be an arbitrary-code channel. These tests pin both
guarantees so the escaping/validation cannot silently regress.
"""
import ast as py_ast

import pytest

from drift.codegen import CodegenError


def _is_pure_string_assign(line: str) -> bool:
    """True if `line` is `<name> = <str-literal>` with an inert RHS."""
    tree = py_ast.parse(line.strip())
    stmt = tree.body[0]
    return (
        isinstance(stmt, py_ast.Assign)
        and isinstance(stmt.value, py_ast.Constant)
        and isinstance(stmt.value.value, str)
    )


class TestStringLiteralInjection:
    def test_escaped_quotes_do_not_break_out(self, transpile):
        # The payload tries to close the string and run os.system.
        payload = 'end" ; __import__("os").system("id") ; y = "'
        # Written in .drift source with the quotes escaped.
        drift_str = payload.replace('"', '\\"')
        out = transpile(
            'agent A { step f() { let x = "' + drift_str + '" return x } }'
        )
        # Whole module must still be valid Python...
        py_ast.parse(out)
        # ...and the assignment must be a pure string, not executable code.
        line = next(l for l in out.splitlines() if l.strip().startswith("x ="))
        assert _is_pure_string_assign(line)

    def test_roundtrips_to_original_value(self, transpile):
        payload = 'he said "hi" and \\ backslash'
        drift_str = payload.replace("\\", "\\\\").replace('"', '\\"')
        out = transpile(
            'agent A { step f() { let x = "' + drift_str + '" return x } }'
        )
        line = next(l for l in out.splitlines() if l.strip().startswith("x ="))
        rhs = line.split("=", 1)[1].strip()
        assert py_ast.literal_eval(rhs) == payload


class TestInterpolationInjection:
    def test_dunder_call_is_rejected(self, transpile):
        with pytest.raises(CodegenError):
            transpile(
                "agent A { step f() { "
                "respond \"hi {__import__('os').system('id')}\" } }"
            )

    def test_dunder_attribute_escape_is_rejected(self, transpile):
        with pytest.raises(CodegenError):
            transpile(
                'agent A { step f() { respond "{().__class__.__bases__}" } }'
            )

    def test_empty_interpolation_is_rejected(self, transpile):
        with pytest.raises(CodegenError):
            transpile('agent A { step f() { respond "value: {}" } }')

    @pytest.mark.parametrize(
        "body",
        ["{r.s}", "{1 + 2}", "{r.s.upper()}", "{r.s}: {r.n * 100}"],
    )
    def test_legitimate_interpolation_still_works(self, transpile, body):
        out = transpile(
            "schema R { s: string n: number } "
            'agent A { step f(r: R) { respond "' + body + '" } }'
        )
        # Emitted as a real f-string and parses cleanly.
        assert 'f"' in out
        py_ast.parse(out)
