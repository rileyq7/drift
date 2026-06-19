"""Parser unit tests — AST-shape level."""
import pytest

from drift import ast_nodes as ast
from drift.lexer import lex
from drift.parser import Parser, ParseError


def parse(source: str) -> ast.Program:
    return Parser(lex(source)).parse()


def single_decl(source: str):
    p = parse(source)
    assert len(p.declarations) == 1, f"Expected 1 decl, got {len(p.declarations)}"
    return p.declarations[0]


class TestConfig:
    def test_basic_config(self):
        d = single_decl('config { name: "x" version: "1.0" }')
        assert isinstance(d, ast.ConfigDecl)
        assert d.entries == {"name": "x", "version": "1.0"}


class TestSchema:
    def test_minimal_schema(self):
        d = single_decl("schema X { a: string }")
        assert isinstance(d, ast.SchemaDecl)
        assert d.name == "X"
        assert len(d.fields) == 1
        assert d.fields[0].name == "a"
        assert d.fields[0].type_expr.name == "string"

    def test_list_of_strings(self):
        d = single_decl("schema X { items: list<string> }")
        f = d.fields[0]
        assert isinstance(f.type_expr, ast.ListType)
        assert f.type_expr.element_type.name == "string"

    def test_optional_field(self):
        d = single_decl("schema X { deadline: string optional }")
        assert d.fields[0].optional is True

    def test_between_constraint(self):
        d = single_decl("schema X { score: number between 0 and 100 }")
        c = d.fields[0].constraints[0]
        assert isinstance(c, ast.BetweenConstraint)
        assert c.low == 0.0
        assert c.high == 100.0

    def test_enum_type(self):
        d = single_decl('schema X { fit: one of "a", "b", "c" }')
        t = d.fields[0].type_expr
        assert isinstance(t, ast.EnumType)
        assert t.values == ["a", "b", "c"]

    def test_confident_type(self):
        d = single_decl("schema X { result: confident<string> }")
        t = d.fields[0].type_expr
        assert isinstance(t, ast.ConfidentType)
        assert t.inner_type.name == "string"

    def test_nested_schema_reference(self):
        d = single_decl("schema X { profile: CompanyProfile }")
        assert d.fields[0].type_expr.name == "CompanyProfile"


class TestAgent:
    def test_minimal_agent(self):
        src = """
        agent G {
          step hello(name: string) -> string {
            respond "hi"
          }
        }
        """
        d = single_decl(src)
        assert isinstance(d, ast.AgentDecl)
        assert d.name == "G"
        assert len(d.steps) == 1
        assert d.steps[0].name == "hello"
        assert d.steps[0].params[0].name == "name"
        assert d.steps[0].return_type.name == "string"

    def test_model_simple(self):
        src = 'agent X { model: "claude-sonnet" step f() { respond "x" } }'
        d = single_decl(src)
        assert d.model_config.default == "claude-sonnet"

    def test_model_prefer_fallback(self):
        src = 'agent X { model: prefer "claude-sonnet" fallback "gpt-4o" step f() { respond "x" } }'
        d = single_decl(src)
        assert d.model_config.prefer == "claude-sonnet"
        assert d.model_config.fallback == "gpt-4o"

    def test_budget_currency_per_run(self):
        src = 'agent X { budget: £5 per run step f() { respond "x" } }'
        d = single_decl(src)
        assert d.budget_config.value == 5.0
        assert d.budget_config.currency_sym == "£"
        assert d.budget_config.per == "run"

    def test_quality_min_confidence(self):
        src = 'agent X { quality: 0.9 minimum confidence step f() { respond "x" } }'
        d = single_decl(src)
        assert d.quality_config.min_confidence == 0.9


class TestStepModifiers:
    @pytest.mark.parametrize("mod", ["cached", "parallel", "manual", "silent"])
    def test_each_modifier(self, mod):
        src = f'agent X {{ {mod} step f() {{ respond "x" }} }}'
        d = single_decl(src)
        assert d.steps[0].modifier == mod


class TestIntentExpressions:
    def _intent_in(self, source: str) -> ast.IntentExpr:
        agent_src = f'agent A {{ step s() {{ let x = {source} }} }}'
        d = single_decl(agent_src)
        let_stmt = d.steps[0].body[0]
        assert isinstance(let_stmt, ast.LetStmt)
        return let_stmt.value

    def test_classify_as_type(self):
        i = self._intent_in("classify doc as Category")
        assert isinstance(i, ast.IntentExpr)
        assert i.verb == "classify"
        assert "as" in i.clauses
        assert i.clauses["as"].name == "Category"

    def test_extract_from(self):
        i = self._intent_in("extract names, dates from doc as Contact")
        assert i.verb == "extract"
        # extract collects field names into a ListLit
        assert isinstance(i.input_expr, ast.ListLit)
        assert "from" in i.clauses
        assert "as" in i.clauses

    def test_summarize_in_n_sentences(self):
        i = self._intent_in("summarize doc in 3 sentences")
        assert i.verb == "summarize"
        assert i.clauses["in"] == {"count": "3", "unit": "sentences"}

    def test_rate_against_criteria_as_type(self):
        i = self._intent_in("rate company against criteria as FitScore")
        assert i.verb == "rate"
        assert "against" in i.clauses
        assert i.clauses["as"].name == "FitScore"

    def test_translate_to_language(self):
        # translate is now a real intent verb.
        agent_src = 'agent A { step s() { let x = translate doc to "French" } }'
        d = parse(agent_src).declarations[0]
        intent = d.steps[0].body[0].value
        assert intent.verb == "translate"
        assert intent.clauses["to"].value == "French"


class TestControlFlow:
    def _body(self, source: str):
        agent_src = f'agent A {{ step s() {{ {source} }} }}'
        return single_decl(agent_src).steps[0].body

    def test_if_otherwise(self):
        body = self._body('if x > 5 { respond "big" } otherwise { respond "small" }')
        stmt = body[0]
        assert isinstance(stmt, ast.IfStmt)
        assert len(stmt.body) == 1
        assert len(stmt.otherwise_body) == 1

    def test_if_otherwise_if_chain(self):
        body = self._body(
            'if x > 70 { respond "a" } '
            'otherwise if x > 40 { respond "b" } '
            'otherwise { respond "c" }'
        )
        stmt = body[0]
        assert stmt.otherwise_if is not None
        assert stmt.otherwise_if.otherwise_body  # the final otherwise lives here

    def test_for_each_sequential(self):
        body = self._body('for each item in items { respond "x" }')
        assert isinstance(body[0], ast.ForEachStmt)
        assert body[0].parallel is False

    def test_for_each_parallel(self):
        body = self._body('for each item in items parallel { respond "x" }')
        assert body[0].parallel is True

    def test_match_with_default(self):
        body = self._body(
            'match x { '
            '  "a" -> respond "alpha" '
            '  "b" -> respond "beta" '
            '  any other -> respond "?" '
            '}'
        )
        stmt = body[0]
        assert isinstance(stmt, ast.MatchStmt)
        assert len(stmt.arms) == 3
        assert stmt.arms[-1].is_default


class TestSchemaConstructor:
    def test_basic_constructor(self):
        agent_src = (
            'agent A { step s() -> Result { '
            '  return Result { name: "x", score: 90 } '
            '} }'
        )
        d = single_decl(agent_src)
        ret = d.steps[0].body[0]
        assert isinstance(ret, ast.ReturnStmt)
        assert isinstance(ret.value, ast.SchemaConstructor)
        assert ret.value.type_name == "Result"
        assert set(ret.value.fields) == {"name", "score"}


class TestErrors:
    def test_unknown_top_level_raises(self):
        with pytest.raises(ParseError):
            parse("garbage X {}")

    def test_pipeline_keyword_now_parses(self):
        # Was xfail — now parses.
        p = parse("pipeline P { discover -> analyze }")
        assert p.declarations[0].name == "P"
        assert len(p.declarations[0].edges) == 1

    def test_tool_keyword_now_parses(self):
        # Was xfail — now parses.
        p = parse('tool t from mcp "x"')
        assert p.declarations[0].kind == "mcp"
