"""
Drift Parser — Turns a token stream into an Abstract Syntax Tree.

Pipeline: source → tokens → [PARSER] → AST → codegen → Python

This is a recursive descent parser. Each grammar rule becomes a method.
The parser consumes tokens left-to-right and builds AST nodes.
"""

from .lexer import Token, TT, LexError
from . import ast_nodes as ast


INTENT_VERBS = {
    'classify', 'extract', 'summarize', 'rate', 'generate',
    'rewrite', 'answer', 'compare', 'decide',
}

INTENT_CLAUSE_KEYWORDS = {
    'as', 'from', 'in', 'against', 'to', 'using', 'considering', 'with',
}

STEP_MODIFIERS = {'cached', 'parallel', 'manual', 'silent'}


class ParseError(Exception):
    def __init__(self, message: str, token: Token):
        self.token = token
        super().__init__(f"Line {token.line}: {message} (got {token.type.name} '{token.value}')")


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    # ─── Token Navigation ──────────────────────────────────────────

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def peek_ahead(self, n: int = 1) -> Token:
        idx = self.pos + n
        if idx < len(self.tokens):
            return self.tokens[idx]
        return self.tokens[-1]  # EOF

    def at_end(self) -> bool:
        return self.peek().type == TT.EOF

    def check(self, tt: TT, value: str = None) -> bool:
        t = self.peek()
        if t.type != tt:
            return False
        if value is not None and t.value != value:
            return False
        return True

    def check_ident(self, value: str) -> bool:
        return self.check(TT.IDENT, value)

    def eat(self, tt: TT = None, value: str = None) -> Token:
        t = self.peek()
        if tt is not None and t.type != tt:
            expected = f"{tt.name}"
            if value:
                expected += f" '{value}'"
            raise ParseError(f"Expected {expected}", t)
        if value is not None and t.value != value:
            raise ParseError(f"Expected '{value}'", t)
        self.pos += 1
        return t

    def eat_ident(self, value: str = None) -> Token:
        return self.eat(TT.IDENT, value)

    def skip_newlines(self):
        while not self.at_end() and self.peek().type == TT.NEWLINE:
            self.pos += 1

    def skip_to_next_statement(self):
        """Skip to next newline or closing brace — used for error recovery."""
        while not self.at_end():
            if self.peek().type in (TT.NEWLINE, TT.RBRACE):
                break
            self.pos += 1

    # ─── Top-Level ─────────────────────────────────────────────────

    def parse(self) -> ast.Program:
        prog = ast.Program()
        self.skip_newlines()

        while not self.at_end():
            decl = self.parse_declaration()
            if decl:
                prog.declarations.append(decl)
            self.skip_newlines()

        return prog

    def parse_declaration(self):
        t = self.peek()

        if t.type == TT.IDENT:
            if t.value == 'config':
                return self.parse_config()
            elif t.value == 'schema':
                return self.parse_schema()
            elif t.value == 'agent':
                return self.parse_agent()
            elif t.value == 'import':
                # Skip import lines for MVP
                self.skip_to_next_statement()
                return None

        raise ParseError("Expected 'agent', 'schema', or 'config'", t)

    # ─── Config ────────────────────────────────────────────────────

    def parse_config(self):
        self.eat_ident('config')
        self.skip_newlines()
        self.eat(TT.LBRACE)
        self.skip_newlines()

        config = ast.ConfigDecl()
        while not self.check(TT.RBRACE):
            key = self.eat(TT.IDENT).value
            self.eat(TT.COLON)
            val = self.eat(TT.STRING).value
            config.entries[key] = val
            self.skip_newlines()

        self.eat(TT.RBRACE)
        return config

    # ─── Schema ────────────────────────────────────────────────────

    def parse_schema(self):
        self.eat_ident('schema')
        name = self.eat(TT.TYPE_IDENT).value
        self.skip_newlines()
        self.eat(TT.LBRACE)
        self.skip_newlines()

        schema = ast.SchemaDecl(name=name)
        while not self.check(TT.RBRACE):
            f = self.parse_schema_field()
            schema.fields.append(f)
            self.skip_newlines()

        self.eat(TT.RBRACE)
        return schema

    def parse_schema_field(self):
        name = self.eat(TT.IDENT).value
        self.eat(TT.COLON)
        type_expr = self.parse_type_expr()

        f = ast.SchemaField(name=name, type_expr=type_expr)

        # Check for constraints
        while not self.at_end() and self.peek().type not in (TT.NEWLINE, TT.RBRACE):
            if self.check_ident('optional'):
                self.eat()
                f.optional = True
            elif self.check_ident('between'):
                c = self.parse_between_constraint()
                f.constraints.append(c)
            else:
                break

        return f

    def parse_type_expr(self) -> ast.TypeExpr:
        t = self.peek()

        # list<T>
        if t.type == TT.IDENT and t.value == 'list':
            self.eat()
            if self.check(TT.LANGLE):
                self.eat(TT.LANGLE)
                inner = self.parse_type_expr()
                self.eat(TT.RANGLE)
                return ast.ListType(name='list', element_type=inner)
            return ast.ListType(name='list')

        # map<K, V>
        if t.type == TT.IDENT and t.value == 'map':
            self.eat()
            if self.check(TT.LANGLE):
                self.eat(TT.LANGLE)
                key_type = self.parse_type_expr()
                self.eat(TT.COMMA)
                val_type = self.parse_type_expr()
                self.eat(TT.RANGLE)
                return ast.MapType(name='map', key_type=key_type, value_type=val_type)
            return ast.MapType(name='map')

        # confident<T>
        if t.type == TT.IDENT and t.value == 'confident':
            self.eat()
            if self.check(TT.LANGLE):
                self.eat(TT.LANGLE)
                inner = self.parse_type_expr()
                self.eat(TT.RANGLE)
                return ast.ConfidentType(name='confident', inner_type=inner)
            return ast.ConfidentType(name='confident')

        # one of "a", "b", "c"
        if t.type == TT.IDENT and t.value == 'one':
            self.eat()
            self.eat_ident('of')
            values = [self.eat(TT.STRING).value]
            while self.check(TT.COMMA):
                self.eat(TT.COMMA)
                values.append(self.eat(TT.STRING).value)
            return ast.EnumType(name='enum', values=values)

        # Simple type: string, number, bool, document, or PascalCase schema ref
        if t.type == TT.TYPE_IDENT:
            return ast.TypeExpr(name=self.eat().value)
        if t.type == TT.IDENT:
            return ast.TypeExpr(name=self.eat().value)

        raise ParseError("Expected type expression", t)

    def parse_between_constraint(self):
        self.eat_ident('between')
        low = float(self.eat(TT.NUMBER).value)
        self.eat_ident('and')
        high = float(self.eat(TT.NUMBER).value)
        return ast.BetweenConstraint(low=low, high=high)

    # ─── Agent ─────────────────────────────────────────────────────

    def parse_agent(self):
        self.eat_ident('agent')
        name = self.eat(TT.TYPE_IDENT).value
        self.skip_newlines()
        self.eat(TT.LBRACE)
        self.skip_newlines()

        agent = ast.AgentDecl(name=name)

        while not self.check(TT.RBRACE):
            t = self.peek()

            if t.type == TT.IDENT and t.value == 'model':
                agent.model_config = self.parse_model_config()
            elif t.type == TT.IDENT and t.value == 'budget':
                agent.budget_config = self.parse_budget_config()
            elif t.type == TT.IDENT and t.value == 'quality':
                agent.quality_config = self.parse_quality_config()
            elif t.type == TT.IDENT and t.value == 'step':
                agent.steps.append(self.parse_step())
            elif t.type == TT.IDENT and t.value in STEP_MODIFIERS:
                # cached step, parallel step, etc.
                agent.steps.append(self.parse_step())
            elif t.type == TT.IDENT and t.value == 'state':
                self.parse_state_block(agent)
            else:
                raise ParseError(f"Unexpected token in agent body", t)

            self.skip_newlines()

        self.eat(TT.RBRACE)
        return agent

    def parse_model_config(self):
        self.eat_ident('model')
        self.eat(TT.COLON)
        config = ast.ModelConfig()

        t = self.peek()

        # Simple: model: "claude-sonnet"
        if t.type == TT.STRING:
            config.default = self.eat(TT.STRING).value
            return config

        # Prefer/fallback: model: prefer "x" fallback "y"
        if t.type == TT.IDENT and t.value == 'prefer':
            self.eat()
            config.prefer = self.eat(TT.STRING).value
            if self.check_ident('fallback'):
                self.eat()
                config.fallback = self.eat(TT.STRING).value
            config.default = config.prefer
            return config

        # Block form: model { ... } — skip for MVP, treat as simple
        raise ParseError("Expected model string or 'prefer'", t)

    def parse_budget_config(self):
        self.eat_ident('budget')
        self.eat(TT.COLON)

        config = ast.BudgetConfig()
        currency_tok = self.eat(TT.CURRENCY)
        config.amount = currency_tok.value
        config.currency_sym = currency_tok.value[0]
        config.value = float(currency_tok.value[1:])

        # per run / per day / per company
        if self.check_ident('per'):
            self.eat()
            config.per = self.eat(TT.IDENT).value
        # max
        elif self.check_ident('max'):
            self.eat()
            config.per = 'run'

        return config

    def parse_quality_config(self):
        self.eat_ident('quality')
        self.eat(TT.COLON)
        value = float(self.eat(TT.NUMBER).value)
        # Consume the literal phrase "minimum confidence" if present — but
        # only those exact words, so we don't accidentally eat the next decl.
        if self.check_ident('minimum'):
            self.eat()
            if self.check_ident('confidence'):
                self.eat()
        return ast.QualityConfig(min_confidence=value)

    def parse_state_block(self, agent):
        """state { name: type [= default] ... } — agent-instance state.

        Fields persist across steps within a single run. They become regular
        Python instance attributes initialized in the agent's __init__.
        """
        self.eat_ident('state')
        self.skip_newlines()
        self.eat(TT.LBRACE)
        self.skip_newlines()

        while not self.check(TT.RBRACE):
            f = ast.StateField()
            f.name = self.eat(TT.IDENT).value
            self.eat(TT.COLON)
            f.type_expr = self.parse_type_expr()
            if self.check(TT.EQUALS):
                self.eat(TT.EQUALS)
                f.default = self.parse_expression()
            agent.state_block.append(f)
            self.skip_newlines()

        self.eat(TT.RBRACE)

    # ─── Step ──────────────────────────────────────────────────────

    def parse_step(self):
        step = ast.StepDecl()

        # Check for modifier
        if self.peek().type == TT.IDENT and self.peek().value in STEP_MODIFIERS:
            step.modifier = self.eat().value

        self.eat_ident('step')
        step.name = self.eat(TT.IDENT).value

        # Parameters
        self.eat(TT.LPAREN)
        step.params = self.parse_params()
        self.eat(TT.RPAREN)

        # Return type
        if self.check(TT.ARROW):
            self.eat(TT.ARROW)
            step.return_type = self.parse_type_expr()

        self.skip_newlines()
        self.eat(TT.LBRACE)
        self.skip_newlines()

        # Body
        while not self.check(TT.RBRACE):
            stmt = self.parse_statement()
            if stmt:
                step.body.append(stmt)
            self.skip_newlines()

        self.eat(TT.RBRACE)
        return step

    def parse_params(self) -> list:
        params = []
        if self.check(TT.RPAREN):
            return params

        params.append(self.parse_param())
        while self.check(TT.COMMA):
            self.eat(TT.COMMA)
            params.append(self.parse_param())
        return params

    def parse_param(self):
        name = self.eat(TT.IDENT).value
        self.eat(TT.COLON)
        type_expr = self.parse_type_expr()
        return ast.Param(name=name, type_expr=type_expr)

    # ─── Statements ────────────────────────────────────────────────

    def parse_statement(self):
        t = self.peek()

        if t.type == TT.IDENT:
            if t.value == 'let':
                return self.parse_let()
            elif t.value == 'return':
                return self.parse_return()
            elif t.value == 'respond':
                return self.parse_respond()
            elif t.value == 'if':
                return self.parse_if()
            elif t.value == 'for':
                return self.parse_for_each()
            elif t.value == 'match':
                return self.parse_match()
            elif t.value == 'attempt':
                return self.parse_attempt()
            elif t.value == 'retry':
                self.eat()
                return ast.RetryStmt()
            elif t.value == 'fail':
                return self.parse_fail()
            elif t.value in INTENT_VERBS:
                expr = self.parse_intent_expr()
                return ast.ExprStmt(expr=expr)
            else:
                # Could be a function call or expression
                expr = self.parse_expression()
                return ast.ExprStmt(expr=expr)

        raise ParseError("Expected statement", t)

    def parse_attempt(self):
        self.eat_ident('attempt')
        self.skip_newlines()
        self.eat(TT.LBRACE)
        self.skip_newlines()

        node = ast.AttemptStmt()
        while not self.check(TT.RBRACE):
            node.body.append(self.parse_statement())
            self.skip_newlines()
        self.eat(TT.RBRACE)

        self.skip_newlines()
        self.eat_ident('recover')
        self.eat_ident('from')
        self.skip_newlines()
        self.eat(TT.LBRACE)
        self.skip_newlines()

        while not self.check(TT.RBRACE):
            arm = ast.RecoverArm()
            t = self.peek()
            # `any error` default arm
            if t.type == TT.IDENT and t.value == 'any':
                self.eat()
                if self.check_ident('error'):
                    self.eat()
                arm.is_default = True
                arm.error_type = 'any'
            elif t.type == TT.TYPE_IDENT:
                arm.error_type = self.eat(TT.TYPE_IDENT).value
            else:
                raise ParseError(
                    "Expected an exception type or `any error` in recover arm", t
                )

            self.eat(TT.ARROW)
            # Body: single statement, or `{ ... }`
            if self.check(TT.LBRACE):
                self.eat(TT.LBRACE)
                self.skip_newlines()
                while not self.check(TT.RBRACE):
                    arm.body.append(self.parse_statement())
                    self.skip_newlines()
                self.eat(TT.RBRACE)
            else:
                arm.body.append(self.parse_statement())
            node.arms.append(arm)
            self.skip_newlines()

        self.eat(TT.RBRACE)
        return node

    def parse_fail(self):
        self.eat_ident('fail')
        # `fail with "<message>"` — `with` is optional sugar
        if self.check_ident('with'):
            self.eat()
        msg = self.parse_expression()
        return ast.FailStmt(message=msg)

    def parse_let(self):
        self.eat_ident('let')
        name = self.eat(TT.IDENT).value
        self.eat(TT.EQUALS)
        value = self.parse_expression()
        return ast.LetStmt(name=name, value=value)

    def parse_return(self):
        self.eat_ident('return')
        if self.peek().type in (TT.NEWLINE, TT.RBRACE, TT.EOF):
            return ast.ReturnStmt(value=None)
        value = self.parse_expression()
        return ast.ReturnStmt(value=value)

    def parse_respond(self):
        self.eat_ident('respond')
        value = self.parse_expression()
        return ast.RespondStmt(value=value)

    def parse_if(self):
        self.eat_ident('if')
        condition = self.parse_expression()
        self.skip_newlines()
        self.eat(TT.LBRACE)
        self.skip_newlines()

        body = []
        while not self.check(TT.RBRACE):
            body.append(self.parse_statement())
            self.skip_newlines()
        self.eat(TT.RBRACE)

        node = ast.IfStmt(condition=condition, body=body)

        self.skip_newlines()
        if self.check_ident('otherwise'):
            self.eat()
            if self.check_ident('if'):
                node.otherwise_if = self.parse_if()
            else:
                self.skip_newlines()
                self.eat(TT.LBRACE)
                self.skip_newlines()
                while not self.check(TT.RBRACE):
                    node.otherwise_body.append(self.parse_statement())
                    self.skip_newlines()
                self.eat(TT.RBRACE)

        return node

    def parse_for_each(self):
        self.eat_ident('for')
        self.eat_ident('each')
        var_name = self.eat(TT.IDENT).value
        self.eat_ident('in')
        iterable = self.parse_expression()

        parallel = False
        if self.check_ident('parallel'):
            self.eat()
            parallel = True

        self.skip_newlines()
        self.eat(TT.LBRACE)
        self.skip_newlines()

        body = []
        while not self.check(TT.RBRACE):
            body.append(self.parse_statement())
            self.skip_newlines()
        self.eat(TT.RBRACE)

        return ast.ForEachStmt(
            var_name=var_name, iterable=iterable,
            body=body, parallel=parallel
        )

    def parse_match(self):
        self.eat_ident('match')
        target = self.parse_expression()
        self.skip_newlines()
        self.eat(TT.LBRACE)
        self.skip_newlines()

        arms = []
        while not self.check(TT.RBRACE):
            arm = ast.MatchArm()
            if self.check_ident('any'):
                self.eat()
                if self.check_ident('other'):
                    self.eat()
                arm.is_default = True
                arm.pattern = ast.Ident(name='_')
            elif self.peek().type == TT.STRING:
                arm.pattern = ast.StringLit(value=self.eat(TT.STRING).value)
            else:
                arm.pattern = self.parse_expression()

            self.eat(TT.ARROW)
            # Single-expression arm or block
            if self.check(TT.LBRACE):
                self.eat(TT.LBRACE)
                self.skip_newlines()
                while not self.check(TT.RBRACE):
                    arm.body.append(self.parse_statement())
                    self.skip_newlines()
                self.eat(TT.RBRACE)
            else:
                arm.body.append(self.parse_statement())

            arms.append(arm)
            self.skip_newlines()

        self.eat(TT.RBRACE)
        return ast.MatchStmt(target=target, arms=arms)

    # ─── Expressions ───────────────────────────────────────────────

    def parse_expression(self):
        """Parse an expression. Handles intent verbs, pipes, and binary ops."""
        t = self.peek()

        # Intent expression
        if t.type == TT.IDENT and t.value in INTENT_VERBS:
            return self.parse_intent_expr()

        return self.parse_pipe_expr()

    def parse_pipe_expr(self):
        """expr |> fn |> fn"""
        left = self.parse_comparison()
        if self.check(TT.PIPE_ARROW):
            ops = []
            while self.check(TT.PIPE_ARROW):
                self.eat(TT.PIPE_ARROW)
                ops.append(self.parse_comparison())
            return ast.PipeExpr(input_expr=left, operations=ops)
        return left

    def parse_comparison(self):
        left = self.parse_addition()

        while not self.at_end():
            t = self.peek()
            if t.type == TT.RANGLE:
                self.eat()
                right = self.parse_addition()
                left = ast.BinOp(op='>', left=left, right=right)
            elif t.type == TT.LANGLE:
                self.eat()
                right = self.parse_addition()
                left = ast.BinOp(op='<', left=left, right=right)
            elif t.type == TT.GTE:
                self.eat()
                right = self.parse_addition()
                left = ast.BinOp(op='>=', left=left, right=right)
            elif t.type == TT.LTE:
                self.eat()
                right = self.parse_addition()
                left = ast.BinOp(op='<=', left=left, right=right)
            elif t.type == TT.EQEQ:
                self.eat()
                right = self.parse_addition()
                left = ast.BinOp(op='==', left=left, right=right)
            elif t.type == TT.NEQ:
                self.eat()
                right = self.parse_addition()
                left = ast.BinOp(op='!=', left=left, right=right)
            elif t.type == TT.IDENT and t.value in ('and', 'or'):
                op = self.eat().value
                right = self.parse_addition()
                left = ast.BinOp(op=op, left=left, right=right)
            elif t.type == TT.IDENT and t.value == 'is':
                self.eat()
                # "is confident", "is uncertain", "is between"
                right = self.parse_primary()
                left = ast.BinOp(op='is', left=left, right=right)
            elif t.type == TT.IDENT and t.value == 'not':
                self.eat()
                if self.check_ident('in'):
                    self.eat()
                    right = self.parse_addition()
                    left = ast.BinOp(op='not in', left=left, right=right)
                else:
                    right = self.parse_addition()
                    left = ast.BinOp(op='not', left=left, right=right)
            else:
                break

        return left

    def parse_addition(self):
        left = self.parse_multiplication()
        while self.check(TT.PLUS) or self.check(TT.MINUS):
            op = self.eat().value
            right = self.parse_multiplication()
            left = ast.BinOp(op=op, left=left, right=right)
        return left

    def parse_multiplication(self):
        left = self.parse_unary()
        while self.check(TT.STAR) or self.check(TT.SLASH):
            op = self.eat().value
            right = self.parse_unary()
            left = ast.BinOp(op=op, left=left, right=right)
        return left

    def parse_unary(self):
        if self.check(TT.MINUS):
            self.eat()
            return ast.UnaryOp(op='-', operand=self.parse_primary())
        if self.check_ident('not'):
            self.eat()
            return ast.UnaryOp(op='not', operand=self.parse_primary())
        return self.parse_postfix()

    def parse_postfix(self):
        """Handle field access (a.b) and method calls (a.b(c))"""
        expr = self.parse_primary()

        while True:
            if self.check(TT.DOT):
                self.eat(TT.DOT)
                field_name = self.eat(TT.IDENT).value
                if self.check(TT.LPAREN):
                    # Method call
                    self.eat(TT.LPAREN)
                    args = self.parse_args()
                    self.eat(TT.RPAREN)
                    expr = ast.FnCall(name=field_name, args=args, target=expr)
                else:
                    expr = ast.FieldAccess(target=expr, field_name=field_name)
            elif self.check(TT.LPAREN) and isinstance(expr, ast.Ident):
                # Function call
                self.eat(TT.LPAREN)
                args = self.parse_args()
                self.eat(TT.RPAREN)
                expr = ast.FnCall(name=expr.name, args=args)
            else:
                break

        return expr

    def parse_primary(self):
        t = self.peek()

        if t.type == TT.NUMBER:
            return ast.NumberLit(value=float(self.eat().value))

        if t.type == TT.STRING:
            val = self.eat().value
            has_interp = '{' in val and '}' in val
            return ast.StringLit(value=val, has_interpolation=has_interp)

        if t.type == TT.CURRENCY:
            tok = self.eat()
            return ast.CurrencyLit(symbol=tok.value[0], value=float(tok.value[1:]))

        if t.type == TT.DURATION:
            tok = self.eat()
            return ast.DurationLit(value=float(tok.value[:-1]), unit=tok.value[-1])

        if t.type == TT.BOOL:
            return ast.BoolLit(value=self.eat().value == 'true')

        if t.type == TT.TYPE_IDENT:
            name = self.eat().value
            # Schema constructor: TypeName { field: value }
            if self.check(TT.LBRACE):
                self.eat(TT.LBRACE)
                self.skip_newlines()
                fields = {}
                while not self.check(TT.RBRACE):
                    fname = self.eat(TT.IDENT).value
                    self.eat(TT.COLON)
                    fval = self.parse_expression()
                    fields[fname] = fval
                    if self.check(TT.COMMA):
                        self.eat(TT.COMMA)
                    self.skip_newlines()
                self.eat(TT.RBRACE)
                return ast.SchemaConstructor(type_name=name, fields=fields)
            return ast.Ident(name=name)

        if t.type == TT.IDENT:
            return ast.Ident(name=self.eat().value)

        if t.type == TT.LPAREN:
            self.eat(TT.LPAREN)
            expr = self.parse_expression()
            self.eat(TT.RPAREN)
            return expr

        if t.type == TT.LBRACKET:
            self.eat(TT.LBRACKET)
            elements = []
            if not self.check(TT.RBRACKET):
                elements.append(self.parse_expression())
                while self.check(TT.COMMA):
                    self.eat(TT.COMMA)
                    elements.append(self.parse_expression())
            self.eat(TT.RBRACKET)
            return ast.ListLit(elements=elements)

        raise ParseError("Expected expression", t)

    def parse_args(self) -> list:
        args = []
        if self.check(TT.RPAREN):
            return args

        # Support named args: key: value
        args.append(self.parse_arg())
        while self.check(TT.COMMA):
            self.eat(TT.COMMA)
            self.skip_newlines()
            args.append(self.parse_arg())
        return args

    def parse_arg(self):
        # Check for named arg: ident: expr
        if (self.peek().type == TT.IDENT and
                self.peek_ahead(1).type == TT.COLON):
            name = self.eat(TT.IDENT).value
            self.eat(TT.COLON)
            val = self.parse_expression()
            # Return as a tuple for named args
            return (name, val)
        return self.parse_expression()

    # ─── Intent Expressions ────────────────────────────────────────

    def parse_intent_expr(self):
        """
        classify doc as FitScore
        extract names, dates from doc as ContactInfo
        summarize doc in 3 sentences
        rate company against criteria as number between 0 and 100
        generate a warm welcome message as string
        """
        verb = self.eat(TT.IDENT).value
        intent = ast.IntentExpr(verb=verb)

        # Parse the input expression — but stop at clause keywords
        if not self.at_end() and self.peek().type not in (TT.NEWLINE, TT.RBRACE, TT.EOF):
            if self.peek().type == TT.IDENT and self.peek().value in INTENT_CLAUSE_KEYWORDS:
                # No input, go straight to clauses
                intent.input_expr = None
            else:
                # For "extract": might have comma-separated field list before "from"
                if verb == 'extract':
                    intent.input_expr = self.parse_extract_fields()
                else:
                    # Collect the input — could be a single variable or a multi-word description
                    # If first token is a known variable/field access, parse as expression
                    # Otherwise collect words until clause keyword as a description string
                    first = self.peek()
                    if first.type in (TT.STRING, TT.NUMBER, TT.CURRENCY, TT.BOOL):
                        intent.input_expr = self.parse_postfix()
                    elif first.type == TT.TYPE_IDENT:
                        intent.input_expr = self.parse_postfix()
                    elif first.type == TT.IDENT and first.value not in INTENT_CLAUSE_KEYWORDS:
                        # Check if this looks like a variable (next token is clause keyword, dot, or end)
                        next_tok = self.peek_ahead(1)
                        if (next_tok.type in (TT.DOT, TT.LPAREN) or
                            next_tok.type in (TT.NEWLINE, TT.RBRACE, TT.EOF) or
                            (next_tok.type == TT.IDENT and next_tok.value in INTENT_CLAUSE_KEYWORDS)):
                            # Single identifier — parse as variable
                            intent.input_expr = self.parse_postfix()
                        else:
                            # Multi-word description — collect until clause keyword
                            words = []
                            while (not self.at_end() and
                                   self.peek().type not in (TT.NEWLINE, TT.RBRACE, TT.EOF) and
                                   not (self.peek().type == TT.IDENT and
                                        self.peek().value in INTENT_CLAUSE_KEYWORDS)):
                                words.append(self.eat().value)
                            intent.input_expr = ast.StringLit(value=" ".join(words))
                    else:
                        intent.input_expr = self.parse_postfix()

        # Parse clauses: as Type, from source, in 3 sentences, etc.
        while (not self.at_end() and
               self.peek().type == TT.IDENT and
               self.peek().value in INTENT_CLAUSE_KEYWORDS):

            clause_key = self.eat(TT.IDENT).value

            if clause_key == 'as':
                intent.clauses['as'] = self.parse_type_expr()
            elif clause_key == 'in':
                # "in 3 sentences"
                if self.peek().type == TT.NUMBER:
                    count = self.eat(TT.NUMBER).value
                    unit = self.eat(TT.IDENT).value if self.peek().type == TT.IDENT else "items"
                    intent.clauses['in'] = {'count': count, 'unit': unit}
                else:
                    intent.clauses['in'] = self.parse_postfix()
            elif clause_key in ('from', 'against', 'to', 'using', 'with'):
                intent.clauses[clause_key] = self.parse_postfix()
            elif clause_key == 'considering':
                # considering factor1, factor2, factor3
                factors = [self.parse_postfix()]
                while self.check(TT.COMMA):
                    self.eat(TT.COMMA)
                    factors.append(self.parse_postfix())
                intent.clauses['considering'] = factors

        return intent

    def parse_extract_fields(self):
        """Parse comma-separated field names for extract verb."""
        fields = [self.eat(TT.IDENT).value]
        while self.check(TT.COMMA):
            self.eat(TT.COMMA)
            self.skip_newlines()  # allow multi-line field lists
            if self.peek().type == TT.IDENT and self.peek().value not in INTENT_CLAUSE_KEYWORDS:
                fields.append(self.eat(TT.IDENT).value)
            else:
                break
        return ast.ListLit(elements=[ast.StringLit(value=f) for f in fields])
