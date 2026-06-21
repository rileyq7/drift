"""Lexer unit tests — token-type level."""
import pytest

from drift.lexer import lex, TT, LexError


def types_of(source):
    """Return token types, dropping NEWLINE/EOF noise for readability."""
    return [t.type for t in lex(source) if t.type not in (TT.NEWLINE, TT.EOF)]


def values_of(source):
    return [t.value for t in lex(source) if t.type not in (TT.NEWLINE, TT.EOF)]


class TestCurrency:
    def test_pound_integer(self):
        toks = lex("£5")
        assert toks[0].type == TT.CURRENCY
        assert toks[0].value == "£5"

    def test_dollar_decimal(self):
        toks = lex("$0.10")
        assert toks[0].type == TT.CURRENCY
        assert toks[0].value == "$0.10"

    def test_euro(self):
        toks = lex("€100")
        assert toks[0].value == "€100"

    def test_currency_without_number_raises(self):
        with pytest.raises(LexError):
            lex("£")


class TestDuration:
    @pytest.mark.parametrize("source,value", [
        ("30s", "30s"),
        ("5m", "5m"),
        ("2h", "2h"),
        ("1d", "1d"),
    ])
    def test_each_unit(self, source, value):
        toks = lex(source)
        assert toks[0].type == TT.DURATION
        assert toks[0].value == value

    def test_number_followed_by_word_is_not_duration(self):
        # "5 minutes" should be a NUMBER followed by an IDENT, not a DURATION
        toks = types_of("5 minutes")
        assert toks == [TT.NUMBER, TT.IDENT]


class TestStrings:
    def test_simple_string(self):
        toks = lex('"hello"')
        assert toks[0].type == TT.STRING
        assert toks[0].value == "hello"

    def test_interpolation_marker_preserved(self):
        # Lexer doesn't parse {expr} — that's codegen's job.
        toks = lex('"Hello, {name}!"')
        assert toks[0].value == "Hello, {name}!"

    def test_triple_quoted_multiline(self):
        src = '"""line one\nline two\nline three"""'
        toks = lex(src)
        assert toks[0].type == TT.STRING
        assert "line one" in toks[0].value
        assert "line three" in toks[0].value

    def test_unterminated_string_raises(self):
        with pytest.raises(LexError):
            lex('"oops')


class TestComments:
    def test_line_comment_emitted(self):
        toks = lex("-- comment\nagent X {}")
        # Comment is emitted as a TT.COMMENT token (preserved for tooling).
        assert any(t.type == TT.COMMENT and t.value == "-- comment" for t in toks)
        # Parsing still works because the parser filters comments out.
        assert TT.IDENT in {t.type for t in toks}

    def test_block_comment_emitted(self):
        toks = lex("{- multi\nline\ncomment -}agent")
        comments = [t for t in toks if t.type == TT.COMMENT]
        assert len(comments) == 1
        assert "multi" in comments[0].value
        assert any(t.type == TT.IDENT and t.value == "agent" for t in toks)

    def test_nested_block_comment(self):
        toks = lex("{- a {- b -} c -}agent")
        comments = [t for t in toks if t.type == TT.COMMENT]
        assert len(comments) == 1
        assert comments[0].value == "{- a {- b -} c -}"
        assert any(t.type == TT.IDENT and t.value == "agent" for t in toks)


class TestIdentifiers:
    def test_pascal_case_is_type_ident(self):
        toks = lex("FitScore")
        assert toks[0].type == TT.TYPE_IDENT

    def test_snake_case_is_ident(self):
        toks = lex("fit_score")
        assert toks[0].type == TT.IDENT

    def test_underscore_prefix_is_ident(self):
        toks = lex("_private")
        assert toks[0].type == TT.IDENT

    def test_bool_literals(self):
        assert lex("true")[0].type == TT.BOOL
        assert lex("false")[0].type == TT.BOOL


class TestOperators:
    @pytest.mark.parametrize("source,expected", [
        ("->", TT.ARROW),
        ("=>", TT.FAT_ARROW),
        ("|>", TT.PIPE_ARROW),
        (">=", TT.GTE),
        ("<=", TT.LTE),
        ("==", TT.EQEQ),
        ("!=", TT.NEQ),
    ])
    def test_two_char_ops(self, source, expected):
        assert lex(source)[0].type == expected

    def test_two_char_takes_precedence_over_single(self):
        # "->" must not be MINUS then RANGLE
        toks = types_of("->")
        assert toks == [TT.ARROW]


class TestNewlineCollapsing:
    def test_multiple_newlines_collapse(self):
        toks = lex("a\n\n\nb")
        types = [t.type for t in toks]
        # IDENT, NEWLINE (only one), IDENT, EOF
        assert types == [TT.IDENT, TT.NEWLINE, TT.IDENT, TT.EOF]


class TestRegressions:
    """Tests for bugs I'd worry about."""

    def test_currency_in_string_is_string(self):
        # "£5" inside a string should not become a CURRENCY token
        toks = lex('"price is £5"')
        assert toks[0].type == TT.STRING
        assert toks[0].value == "price is £5"

    def test_dash_dash_inside_string_is_not_comment(self):
        toks = lex('"-- not a comment"')
        assert toks[0].type == TT.STRING
        assert toks[0].value == "-- not a comment"

    def test_comment_at_eof_without_newline(self):
        # No trailing newline after the comment
        toks = lex("agent X -- trailing")
        types = [t.type for t in toks]
        assert TT.IDENT in types
        assert TT.TYPE_IDENT in types
