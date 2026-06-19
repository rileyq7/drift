"""Adversarial tests — push the parser and codegen into corners.

These tests document where the spec is underspecified, where the parser is
fragile, and where syntactic ambiguity bites. Some of them XFAIL intentionally
to mark known gaps without breaking CI.
"""
import pytest

from drift.lexer import lex, LexError
from drift.parser import Parser, ParseError


def parse(source: str):
    return Parser(lex(source)).parse()


def transpile_str(source: str) -> str:
    from drift.codegen import CodeGenerator
    return CodeGenerator().generate(parse(source))


# ─── Lexer adversarial ──────────────────────────────────────────────

class TestLexerEdges:
    def test_only_whitespace(self):
        toks = lex("   \n   \n  ")
        # Should produce at most some NEWLINEs + EOF, never error
        assert toks[-1].type.name == "EOF"

    def test_only_comments(self):
        toks = lex("-- a\n-- b\n{- c -}")
        assert toks[-1].type.name == "EOF"

    def test_empty_string(self):
        assert lex("") [0].type.name == "EOF"

    def test_escape_in_string(self):
        toks = lex(r'"line1\nline2"')
        # The lexer treats \ as escape-next-char (so \n becomes a literal 'n')
        # Document the current behavior, even if it's not what most users expect.
        assert toks[0].value == "line1nline2"

    def test_block_comment_unterminated(self):
        # The lexer currently does NOT raise — it walks to EOF silently.
        # That's a fragility worth flagging.
        toks = lex("{- forever")
        assert toks[-1].type.name == "EOF"


# ─── Parser adversarial ─────────────────────────────────────────────

class TestParserEdges:
    def test_empty_agent_body(self):
        # An agent with no steps — does it parse?
        d = parse("agent A { }").declarations[0]
        assert d.name == "A"
        assert len(d.steps) == 0

    def test_empty_step_body(self):
        # A step with no statements.
        d = parse("agent A { step f() { } }").declarations[0]
        assert len(d.steps[0].body) == 0

    def test_empty_schema(self):
        d = parse("schema X { }").declarations[0]
        assert len(d.fields) == 0

    def test_trailing_newlines(self):
        d = parse('agent A { step f() { respond "x" } }\n\n\n').declarations[0]
        assert d.name == "A"

    def test_multiple_declarations(self):
        src = (
            'config { name: "x" }\n'
            'schema S { a: string }\n'
            'agent A { step f() { respond "x" } }\n'
        )
        p = parse(src)
        assert len(p.declarations) == 3

    def test_intent_verb_at_statement_position(self):
        # `extract a, b from doc` as a statement (not a let-binding) — does it parse?
        d = parse('agent A { step f() { classify x as Y } }').declarations[0]
        assert len(d.steps[0].body) == 1


class TestIntentExpressionAmbiguity:
    """The §15 question: 'where is the line for natural language?'"""

    def test_classify_with_unknown_input_word(self):
        # `classify mystery as Y` — `mystery` isn't bound. Should parse, will
        # fail at runtime as a NameError. That's the right call (parser stays
        # cheap, runtime catches it), but worth pinning behavior.
        d = parse('agent A { step f() { let x = classify mystery as Y } }').declarations[0]
        let = d.steps[0].body[0]
        assert let.value.verb == "classify"

    def test_extract_without_from(self):
        # `extract a, b as X` — no `from` clause. The grammar allows this.
        d = parse('agent A { step f() { let x = extract a, b as X } }').declarations[0]
        intent = d.steps[0].body[0].value
        assert intent.verb == "extract"
        assert "as" in intent.clauses
        assert "from" not in intent.clauses

    def test_summarize_without_count(self):
        d = parse('agent A { step f() { let x = summarize doc } }').declarations[0]
        intent = d.steps[0].body[0].value
        assert intent.verb == "summarize"

    def test_multi_word_description_input(self):
        # "summarize the latest report" — three words before the clause.
        # The parser collects them as a string literal, then 'in 3 sentences'
        # parses as the count clause. Surprised this works robustly — pinned
        # here so future parser changes don't break it.
        d = parse(
            'agent A { step f() { let x = summarize the latest report in 3 sentences } }'
        ).declarations[0]
        intent = d.steps[0].body[0].value
        assert intent.verb == "summarize"
        assert intent.clauses["in"]["count"] == "3"


class TestMatchKeywordConflict:
    """§6.1 lists `match` as an intent verb, but `match` is also the keyword
    for the pattern-matching statement. The parser routes `match` to the
    statement parser first, making the intent form unreachable.
    """

    def test_match_as_statement_works(self):
        d = parse(
            'agent A { step f() { '
            '  match x { "a" -> respond "alpha" '
            '            any other -> respond "?" } '
            '} }'
        ).declarations[0]
        assert d.steps[0].body[0].__class__.__name__ == "MatchStmt"

    def test_match_as_intent_verb(self):
        # Resolved: parser looks ahead for `against` to distinguish the
        # intent form from the statement form.
        d = parse(
            'agent A { step f() { let x = match input against criteria as Result } }'
        ).declarations[0]
        intent = d.steps[0].body[0].value
        assert intent.verb == "match"
        assert "against" in intent.clauses
        assert intent.clauses["as"].name == "Result"


class TestSpecGaps:
    """Sections of the spec that aren't implemented yet — pin them so we
    don't accidentally claim they work."""

    # §2.4 tool declarations ARE now parsed — see test_tool.py

    # §2.5 pipeline declarations ARE now parsed — see test_pipeline.py

    # §8 attempt/recover IS now parsed — see tests/unit/test_attempt_recover.py

    # §9 memory block IS now parsed — see test_memory.py

    # §6.2 define verb IS now parsed — see test_define_verb.py

    # §4 model block form IS now parsed — see test_model_block.py

    # §9 state block contents ARE now preserved — see test_state.py


class TestSchemaConstraintEdgeCases:
    def test_between_with_negative(self):
        # Spec doesn't say negative numbers are allowed in `between`. They are
        # in fact lexed as MINUS + NUMBER, so parse_between_constraint can't
        # handle them. Document.
        with pytest.raises(ParseError):
            parse("schema X { delta: number between -1 and 1 }")

    def test_between_with_float(self):
        d = parse("schema X { p: number between 0.0 and 1.0 }").declarations[0]
        c = d.fields[0].constraints[0]
        assert c.low == 0.0
        assert c.high == 1.0

    def test_optional_with_constraint(self):
        d = parse(
            "schema X { score: number between 0 and 100 optional }"
        ).declarations[0]
        f = d.fields[0]
        assert f.optional is True
        assert len(f.constraints) == 1


class TestNestedTypes:
    def test_list_of_list(self):
        d = parse("schema X { grid: list<list<string>> }").declarations[0]
        t = d.fields[0].type_expr
        assert t.__class__.__name__ == "ListType"
        assert t.element_type.__class__.__name__ == "ListType"

    def test_map_of_string_to_list(self):
        d = parse("schema X { idx: map<string, list<string>> }").declarations[0]
        t = d.fields[0].type_expr
        assert t.__class__.__name__ == "MapType"

    def test_confident_with_nested(self):
        d = parse("schema X { r: confident<list<string>> }").declarations[0]
        t = d.fields[0].type_expr
        assert t.__class__.__name__ == "ConfidentType"
        assert t.inner_type.__class__.__name__ == "ListType"


class TestCodegenEdges:
    def test_step_with_no_return_type_no_checkpoint_after_intent(self):
        # If a step has no return type, the last expression shouldn't be
        # wrapped in `_result = ...; return _result`.
        out = transpile_str(
            'agent A { step f() { let x = classify doc as Y } }'
        )
        # The let binds, no return at the end.
        assert "let x" not in out  # `let` keyword shouldn't leak
        assert "x = await self.intent" in out

    def test_currency_in_step_body(self):
        # Currency literals as expressions inside step bodies.
        out = transpile_str(
            'agent A { step f() { let cost = $5 } }'
        )
        # Currency value should appear; symbol may not.
        assert "cost = 5" in out or "cost = 5.0" in out
