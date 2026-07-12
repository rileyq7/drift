"""Codegen unit tests — check the generated Python contains the right pieces.

We don't pin exact line-by-line output here (that's what golden tests do).
We check that *important* code shapes appear, so refactors that change
formatting don't fail this layer.
"""
import pytest


class TestSchemaCodegen:
    def test_dataclass_decorator(self, transpile):
        out = transpile("schema X { a: string }")
        assert "@dataclass" in out
        assert "class X:" in out
        assert "a: str" in out

    def test_list_type(self, transpile):
        out = transpile("schema X { tags: list<string> }")
        assert "tags: list[str]" in out

    def test_optional_field(self, transpile):
        out = transpile("schema X { tag: string optional }")
        assert "Optional[str]" in out
        assert "= None" in out

    def test_enum_becomes_literal(self, transpile):
        out = transpile('schema X { fit: one of "a", "b" }')
        assert 'Literal["a", "b"]' in out

    def test_between_constraint_generates_validate(self, transpile):
        out = transpile("schema X { score: number between 0 and 100 }")
        assert "def validate(self):" in out
        assert "0.0 <= self.score <= 100.0" in out

    def test_between_violation_raises_schema_violation_not_assertion_error(self, transpile):
        # step_decorator's retry loop only catches SchemaViolation to retry
        # with a stricter prompt (see drift/runtime/core.py). An
        # AssertionError from a bare `assert` used to crash the step outright
        # on the first out-of-range value instead of getting that retry.
        out = transpile("schema X { score: number between 0 and 100 }")
        assert "raise SchemaViolation(" in out
        assert "assert " not in out

    def test_one_of_constraint_generates_validate(self, transpile):
        # `one of` used to become ONLY a Literal[...] type hint with nothing
        # checking it at runtime — an LLM returning an out-of-enum value
        # passed validation silently. Must now generate a real check too.
        out = transpile('schema X { fit: one of "a", "b" }')
        assert "def validate(self):" in out
        assert "self.fit not in ('a', 'b')" in out
        assert "raise SchemaViolation(" in out

    def test_confident_becomes_runtime_class(self, transpile):
        # confident<T> compiles to the Confident runtime class. T is doc only —
        # the runtime stores Any in .value.
        out = transpile("schema X { r: confident<string> }")
        assert "r: Confident" in out
        assert "tuple[" not in out


class TestAgentCodegen:
    BASIC_AGENT = (
        'agent A { model: "claude-sonnet" budget: $1 per run '
        'step f(x: string) -> string { respond "hi" } }'
    )

    def test_class_inherits_from_agent(self, transpile):
        out = transpile(self.BASIC_AGENT)
        assert "class A(Agent):" in out

    def test_init_creates_model_router(self, transpile):
        out = transpile(self.BASIC_AGENT)
        assert 'ModelRouter(default="claude-sonnet")' in out

    def test_init_creates_budget(self, transpile):
        out = transpile(self.BASIC_AGENT)
        assert "Budget(max_per_run=1.0" in out
        assert 'currency="USD"' in out

    def test_pound_currency_maps_to_gbp(self, transpile):
        out = transpile(
            'agent A { budget: £5 per run step f() { respond "x" } }'
        )
        assert 'currency="GBP"' in out

    def test_step_is_async_with_decorator(self, transpile):
        out = transpile(self.BASIC_AGENT)
        assert "@step_decorator(output=str)" in out
        assert "async def f(self, x: str) -> str:" in out

    def test_step_has_budget_precheck(self, transpile):
        out = transpile(self.BASIC_AGENT)
        assert "self.cost_tracker.pre_check()" in out


class TestIntentCodegen:
    def _step_body(self, transpile, intent_src: str) -> str:
        out = transpile(
            f'agent A {{ step s() {{ let x = {intent_src} }} }}'
        )
        return out

    def test_classify_translation(self, transpile):
        out = self._step_body(transpile, "classify doc as MySchema")
        assert 'verb="classify"' in out
        assert "input_data=doc" in out
        assert "output_schema=MySchema" in out

    def test_extract_with_fields_and_source(self, transpile):
        out = self._step_body(transpile, "extract a, b from doc as X")
        assert 'verb="extract"' in out
        assert 'input_data=["a", "b"]' in out
        assert "source=doc" in out
        assert "output_schema=X" in out

    def test_summarize_with_count(self, transpile):
        out = self._step_body(transpile, "summarize doc in 3 sentences")
        assert 'verb="summarize"' in out
        assert "count=3" in out
        assert 'unit="sentences"' in out

    def test_rate_with_against(self, transpile):
        out = self._step_body(transpile, "rate company against criteria as FitScore")
        assert 'verb="rate"' in out
        assert "criteria=criteria" in out

    def test_hyphenated_word_in_multiword_description_stays_intact(self, transpile):
        # Regression: a hyphenated word in a free-form intent-input
        # description (e.g. "one-sentence") lexes as IDENT MINUS IDENT —
        # indistinguishable from subtraction at the token level. Rejoining
        # every token with a single space used to turn "one-sentence" into
        # "one - sentence" in the actual LLM prompt text sent at runtime,
        # silently changing what the model was asked for.
        out = self._step_body(
            transpile, "generate a warm one-sentence greeting as string"
        )
        assert 'input_data="a warm one-sentence greeting"' in out
        assert " - " not in out

    def test_multiple_hyphens_in_description_stay_intact(self, transpile):
        out = self._step_body(
            transpile, "generate a state-of-the-art summary as string"
        )
        assert 'input_data="a state-of-the-art summary"' in out

    def test_with_clause_is_wired_up(self, transpile):
        # Regression: `with` was a recognized clause keyword (parser
        # consumed it into intent.clauses['with']) but codegen never read
        # that key at all — the clause value was silently dropped, never
        # reaching self.intent() or the actual LLM prompt.
        out = self._step_body(
            transpile, "summarize doc with formatting_notes as string"
        )
        assert "with_=formatting_notes" in out

    def test_using_clause_accepts_comma_separated_list(self, transpile):
        # LLM.md: "Their values can be variables, expressions, or
        # comma-separated lists" for every clause — this used to be true
        # only for `considering`; `using a, b` was a ParseError.
        out = self._step_body(
            transpile, "generate a reply using ctx1, ctx2 as string"
        )
        assert "context=[ctx1, ctx2]" in out

    def test_single_value_from_clause_stays_unwrapped(self, transpile):
        # A single value must NOT become a 1-item list — keeps the common
        # case (`from doc`) passing a plain value, matching prior behavior
        # and build_intent_prompt's existing single-value formatting.
        out = self._step_body(transpile, 'answer "q" from doc as string')
        assert "source=doc" in out
        assert "source=[doc]" not in out

    def test_missing_as_clause_does_not_swallow_following_return(self, parse_ast):
        # Regression: the free-form description word-collection loop only
        # stopped at clause keywords (as/from/in/against/to/using/
        # considering/with) or newline/brace — not at other statement-
        # starting keywords. `generate a reply return m` (a missing `as`
        # clause followed by a legitimate `return` statement on the same
        # line) used to swallow "return m" as more description text
        # instead of parsing it as its own ReturnStmt, silently dropping
        # the return.
        src = 'agent A { step f() -> string { let m = classify x return m } }'
        p = parse_ast(src)
        agent = p.declarations[0]
        body = agent.steps[0].body
        assert len(body) == 2
        assert body[1].__class__.__name__ == "ReturnStmt"
        assert body[1].value.name == "m"


class TestControlFlowCodegen:
    def test_if_otherwise_chain(self, transpile):
        src = (
            'agent A { step s(x: number) { '
            '  if x > 70 { respond "a" } '
            '  otherwise if x > 40 { respond "b" } '
            '  otherwise { respond "c" } '
            '} }'
        )
        out = transpile(src)
        assert "if (x > 70):" in out
        assert "elif (x > 40):" in out
        assert "else:" in out

    @pytest.mark.asyncio
    async def test_or_of_equalities_evaluates_correctly_at_runtime(self, transpile, tmp_path):
        # End-to-end proof (not just AST/text shape) that `x == "a" or
        # x == "b"` actually evaluates as boolean-logic-correct — this is
        # the exact pattern that used to silently miscompile to
        # `((x == "a") or x) == "b"`.
        src = (
            'agent A { step classify_status(status: string) -> string { '
            '  if status == "strong fit" or status == "possible fit" { '
            '    return "qualified" '
            '  } otherwise { '
            '    return "not qualified" '
            '  } '
            '} }'
        )
        py = transpile(src).replace("Source: <drift_file>", "Source: inline")
        path = tmp_path / "gen_precedence.py"
        path.write_text(py)
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("gen_precedence", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["gen_precedence"] = mod
        spec.loader.exec_module(mod)
        agent = mod.A()

        assert await agent.classify_status(status="strong fit") == "qualified"
        assert await agent.classify_status(status="possible fit") == "qualified"
        assert await agent.classify_status(status="weak fit") == "not qualified"
        assert await agent.classify_status(status="no fit") == "not qualified"

    def test_for_each_sequential(self, transpile):
        src = (
            'agent A { step s(xs: list<string>) { '
            '  for each x in xs { respond "hi" } '
            '} }'
        )
        out = transpile(src)
        assert "for x in xs:" in out
        # gather_or_cancel is always imported (part of the fixed runtime
        # import list), so check it's not actually CALLED.
        assert "gather_or_cancel(" not in out

    def test_for_each_parallel_uses_gather(self, transpile):
        src = (
            'agent A { step s(xs: list<string>) { '
            '  for each x in xs parallel { respond "hi" } '
            '} }'
        )
        out = transpile(src)
        # gather_or_cancel wraps asyncio.gather with cancel-on-failure
        # cleanup for still-in-flight siblings (drift/runtime/core.py).
        assert "gather_or_cancel(" in out


class TestBareCrossAgentCallStatement:
    """Regression: LLM.md only documents `let result = OtherAgent.step(...)`
    (§10), but a bare cross-agent call with no `let` binding — a
    completely reasonable side-effect-only call, e.g. paging an on-call
    notifier agent and discarding its return value — never parsed at all.
    parse_statement's dispatch only checked `t.type == TT.IDENT`; a
    statement starting with TT.TYPE_IDENT (any PascalCase agent name)
    fell straight through to a bare "Expected statement" ParseError."""

    def test_bare_cross_agent_call_parses(self, transpile):
        src = (
            'agent Notifier { model: "claude-haiku" '
            '  step ping(msg: string) -> string { return msg } } '
            'agent A { model: "claude-haiku" '
            '  step f(msg: string) { Notifier.ping(msg) } }'
        )
        out = transpile(src)
        assert "await Notifier().ping(msg)" in out

    @pytest.mark.asyncio
    async def test_bare_cross_agent_call_actually_executes(self, transpile, tmp_path, monkeypatch):
        src = (
            'agent Notifier { model: "claude-haiku" '
            '  step ping(msg: string) -> string { return msg } } '
            'agent A { model: "claude-haiku" '
            '  step f(msg: string) -> string { Notifier.ping(msg) return "done" } }'
        )
        py = transpile(src).replace("Source: <drift_file>", "Source: inline")
        path = tmp_path / "gen_bare_cross_agent.py"
        path.write_text(py)
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("gen_bare_cross_agent", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["gen_bare_cross_agent"] = mod
        spec.loader.exec_module(mod)

        # Prove Notifier.ping actually ran (not just parsed/awaited-away as
        # a no-op) by monkeypatching its method to record the call.
        calls = []
        orig_ping = mod.Notifier.ping
        async def spy_ping(self, msg):
            calls.append(msg)
            return await orig_ping(self, msg)
        monkeypatch.setattr(mod.Notifier, "ping", spy_ping)

        agent = mod.A()
        result = await agent.f(msg="hello")
        assert result == "done"
        assert calls == ["hello"]


class TestRespond:
    def test_respond_with_interpolation(self, transpile):
        src = 'agent A { step s(name: string) { respond "Hi, {name}!" } }'
        out = transpile(src)
        # f-string emitted for {expr} interpolation
        assert 'f"Hi, {name}!"' in out
        assert "self.output(" in out

    def test_doubled_braces_are_literal_not_interpolation(self, transpile):
        # LLM.md documents `{{`/`}}` as the escape for a literal brace —
        # pinning the actual behavior here since it was previously
        # undocumented (the mechanism existed in codegen but no LLM.md
        # section named it as a supported, intentional escape).
        src = 'agent A { step s() { respond "set notation: {{1, 2, 3}}" } }'
        out = transpile(src)
        assert 'f"set notation: {{1, 2, 3}}"' in out


class TestImports:
    def test_runtime_imports_present(self, transpile):
        out = transpile("schema X { a: string }")
        assert "from drift.runtime import" in out
        assert "Agent" in out
        assert "step_decorator" in out
        assert "ModelRouter" in out


class TestStringEscapingIntoGeneratedSource:
    """config values and model names are user-written STRING literals
    interpolated into generated Python string literals. `gen_config` and
    `gen_model_init` used to embed them with bare `"{value}"` quoting — a
    `"` or `\\` in the value broke out of the generated literal, producing
    a SyntaxError only `drift run` (not `drift check`) would ever surface,
    mislabeled as a "Runtime error" even though nothing had executed.
    """
    import ast as _py_ast

    def test_config_value_with_embedded_quote_produces_valid_python(self, transpile):
        out = transpile('config { name: "my \\"agent\\"" version: "1.0" }')
        self._py_ast.parse(out)  # raises SyntaxError if codegen is broken
        assert 'my \\"agent\\"' in out

    def test_config_value_with_backslash_produces_valid_python(self, transpile):
        out = transpile('config { name: "C:\\\\Users\\\\x" version: "1.0" }')
        self._py_ast.parse(out)

    def test_model_default_with_embedded_quote_produces_valid_python(self, transpile):
        src = 'agent A { model: "claude\\"haiku" step f() { respond "x" } }'
        out = transpile(src)
        self._py_ast.parse(out)

    def test_model_fallback_with_embedded_quote_produces_valid_python(self, transpile):
        src = (
            'agent A { model { default: "claude-haiku" '
            'fallback: "gpt\\"4o" } step f() { respond "x" } }'
        )
        out = transpile(src)
        self._py_ast.parse(out)

    def test_model_upgrade_target_with_embedded_quote_produces_valid_python(self, transpile):
        src = (
            'agent A { model { default: "claude-haiku" '
            'upgrade to "claude\\"opus" when { confidence < 0.5 } } '
            'step f() { respond "x" } }'
        )
        out = transpile(src)
        self._py_ast.parse(out)

    def test_generate_catches_a_codegen_bug_that_produces_invalid_python(self, monkeypatch):
        # Safety net: exercise the REAL generate() end-to-end, with a single
        # gen_config call patched to reintroduce the old unescaped-quoting
        # bug, to prove generate()'s own validation (not a reimplementation
        # of it) catches broken output as a CodegenError — instead of
        # `drift check` reporting "syntax OK" on source that only fails at
        # `drift run`.
        from drift.lexer import lex
        from drift.parser import Parser
        from drift.codegen import CodeGenerator, CodegenError

        def broken_gen_config(self, config):
            self.emit_line("DRIFT_CONFIG = {")
            self.indent()
            for k, v in config.entries.items():
                self.emit_line(f'"{k}": "{v}",')  # the old, unescaped bug
            self.dedent()
            self.emit_line("}")

        monkeypatch.setattr(CodeGenerator, "gen_config", broken_gen_config)

        program = Parser(lex('config { name: "my \\"agent\\"" }')).parse()
        with pytest.raises(CodegenError, match="codegen produced invalid Python"):
            CodeGenerator().generate(program)
