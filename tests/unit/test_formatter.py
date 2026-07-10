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
