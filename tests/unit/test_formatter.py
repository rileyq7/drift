"""Tests for `drift fmt` (drift.formatter.format_source).

The documented invariant is idempotence: format(format(x)) == format(x).
String escaping must survive a round trip.
"""
from drift.formatter import format_source


class TestFormatterIdempotence:
    def test_plain_program_is_idempotent(self):
        src = 'agent A { step f() { respond "hi" } }'
        once = format_source(src)
        assert format_source(once) == once

    def test_escaped_quotes_survive_round_trip(self):
        src = 'config { name: "he said \\"hi\\"" }'
        once = format_source(src)
        # Formatting again must be stable (no quote corruption).
        assert format_source(once) == once
        # And the escaped quote must still be present, not bare.
        assert '\\"hi\\"' in once

    def test_backslash_survives_round_trip(self):
        src = 'config { path: "C:\\\\Users\\\\name" }'
        once = format_source(src)
        assert format_source(once) == once
        assert '\\\\Users' in once

    def test_multiline_block_comment_is_idempotent(self):
        src = (
            'agent Foo {\n'
            '  model: "claude-haiku"\n'
            '  {- multi\n'
            '     line\n'
            '     comment -}\n'
            '  step bar() -> string {\n'
            '    return "hi"\n'
            '  }\n'
            '}\n'
        )
        once = format_source(src)
        assert format_source(once) == once

    def test_multiline_block_comment_preserves_relative_indentation(self):
        src = (
            'agent Foo {\n'
            '  step bar() -> string {\n'
            '    {- a note:\n'
            '         - point one\n'
            '         - point two\n'
            '       done -}\n'
            '    return "hi"\n'
            '  }\n'
            '}\n'
        )
        once = format_source(src)
        assert format_source(once) == once
        lines = once.splitlines()
        note_line = next(l for l in lines if 'a note:' in l)
        point_line = next(l for l in lines if 'point one' in l)
        note_indent = len(note_line) - len(note_line.lstrip(' '))
        point_indent = len(point_line) - len(point_line.lstrip(' '))
        # "point one" was nested deeper than "a note:" in the source —
        # that relative nesting must survive, not just absolute stability.
        assert point_indent > note_indent

    def test_confident_generic_stays_tight_no_spaces(self):
        # Regression: `confident<T>` wasn't in _TYPE_GENERIC_HEADS
        # (list/dict/set/tuple/optional were, confident was missing), so
        # the formatter added spaces around the angle brackets —
        # `confident<Foo>` became `confident < Foo >` — directly
        # contradicting LLM.md's own claim that the formatter normalizes
        # (removes) spaces inside generic types.
        src = (
            'agent A { step f() -> string { '
            'let x = rate y against z as confident<Foo> '
            'return x } }'
        )
        once = format_source(src)
        assert 'confident<Foo>' in once
        assert 'confident < Foo >' not in once
        assert format_source(once) == once
