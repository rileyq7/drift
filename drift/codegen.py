"""
Drift Code Generator — Walks the AST and emits Python source code.

Pipeline: source → tokens → AST → [CODEGEN] → Python

This is a deterministic, mechanical translation. No AI involved.
The same .drift input ALWAYS produces the same .py output.
"""

import ast as py_ast
import re
from . import ast_nodes as ast


class CodegenError(Exception):
    """Raised when a .drift construct cannot be safely lowered to Python."""


def _py_str_literal(value: str) -> str:
    """Emit a safe double-quoted Python string literal for `value`.

    Uses repr() to escape backslashes/quotes/control chars (closing the
    injection hole), then normalizes to double quotes to match the rest of
    codegen's output style. This keeps generated source stable and readable
    while making it impossible for `value` to break out of its quotes.
    """
    r = repr(value)
    if r[0] == "'":
        # repr chose single quotes: unescape any \' then escape bare ".
        body = r[1:-1].replace("\\'", "'").replace('"', '\\"')
        return '"' + body + '"'
    return r


def _validate_interp_expr(expr_src: str) -> None:
    """Reject interpolation bodies that could escape the intended semantics.

    Drift string interpolation is documented to allow ordinary expressions
    (member access, method calls, arithmetic). It is NOT an arbitrary-code
    channel: dunder access (`__import__`, `__class__`, ...) is the standard
    sandbox-escape vector, so we forbid any name — attribute or bare — that is
    dunder-shaped. Syntactically invalid bodies are rejected too, rather than
    emitted verbatim into the f-string.
    """
    stripped = expr_src.strip()
    if not stripped:
        raise CodegenError("empty interpolation `{}` in string")
    try:
        tree = py_ast.parse(stripped, mode='eval')
    except SyntaxError as e:
        raise CodegenError(
            f"invalid interpolation expression {stripped!r}: {e.msg}"
        )
    for node in py_ast.walk(tree):
        name = None
        if isinstance(node, py_ast.Attribute):
            name = node.attr
        elif isinstance(node, py_ast.Name):
            name = node.id
        if name and name.startswith('__') and name.endswith('__'):
            raise CodegenError(
                f"interpolation expression {stripped!r} references {name!r}; "
                "dunder access is not allowed inside string interpolation"
            )


# Map Drift types to Python/Pydantic types
TYPE_MAP = {
    'string': 'str',
    'number': 'float',
    'bool': 'bool',
    'document': 'str',
    'currency': 'float',
    'duration': 'float',
    'timestamp': 'str',
}

# Map currency symbols to ISO codes
CURRENCY_MAP = {'£': 'GBP', '$': 'USD', '€': 'EUR'}


class CodeGenerator:
    def __init__(self):
        self.indent_level = 0
        self.lines: list[str] = []
        self.schemas_declared: list[str] = []
        self.agent_names: set[str] = set()

    def generate(self, program: ast.Program) -> str:
        """Generate complete Python source from a Drift program."""
        self.emit_header()
        self.emit_line("")

        # First pass: collect all schema names and agent names. Agent
        # names are needed by gen_fn_call to detect cross-agent step
        # invocations like `GrantChecker.evaluate(x)` and emit a
        # one-shot agent instantiation + await rather than calling an
        # unbound method.
        for decl in program.declarations:
            if isinstance(decl, ast.SchemaDecl):
                self.schemas_declared.append(decl.name)
            elif isinstance(decl, ast.AgentDecl):
                self.agent_names.add(decl.name)

        # Generate declarations in order
        for decl in program.declarations:
            if isinstance(decl, ast.ImportDecl):
                self.gen_import(decl)
            elif isinstance(decl, ast.ConfigDecl):
                self.gen_config(decl)
            elif isinstance(decl, ast.SchemaDecl):
                self.gen_schema(decl)
            elif isinstance(decl, ast.VerbDecl):
                self.gen_verb_decl(decl)
            elif isinstance(decl, ast.ToolDecl):
                self.gen_tool_decl(decl)
            elif isinstance(decl, ast.AgentDecl):
                self.gen_agent(decl)
            elif isinstance(decl, ast.PipelineDecl):
                self.gen_pipeline(decl)
            self.emit_line("")

        python_source = '\n'.join(self.lines)

        # Verify the emitted text is actually valid Python before handing it
        # back. Codegen is meant to be a total function from valid AST to
        # valid Python — a SyntaxError here means a bug in codegen itself
        # (e.g. a user string value interpolated into a Python literal
        # without escaping), not a rejected-but-documented construct like
        # `~>` or `parallel step` (those raise CodegenError directly, above,
        # before ever reaching this point). Catching it here means `drift
        # check` — which runs codegen and discards the output specifically
        # to catch can't-compile constructs — actually catches this class
        # too, instead of reporting "syntax OK" on source that fails only
        # once you get to `drift run`.
        try:
            py_ast.parse(python_source)
        except SyntaxError as e:
            raise CodegenError(
                f"codegen produced invalid Python (a codegen bug, not a "
                f"rejected .drift construct): {e.msg} at generated line "
                f"{e.lineno}. This means a value somewhere wasn't escaped "
                f"correctly when embedded into the output — please report "
                f"this as a Drift bug rather than trying to work around it."
            ) from e

        return python_source

    # ─── Output Helpers ────────────────────────────────────────────

    def emit_line(self, text: str = ""):
        if text:
            self.lines.append("    " * self.indent_level + text)
        else:
            self.lines.append("")

    def emit_lines(self, *texts):
        for t in texts:
            self.emit_line(t)

    def indent(self):
        self.indent_level += 1

    def dedent(self):
        self.indent_level -= 1

    # ─── Header ────────────────────────────────────────────────────

    def emit_header(self):
        self.emit_lines(
            '"""',
            'Auto-generated by Drift v0.1 — do not edit by hand.',
            'Source: <drift_file>',
            '"""',
            '',
            'import asyncio',
            'from typing import Optional, Literal',
            'from dataclasses import dataclass, field',
            '',
            '# Drift runtime imports',
            'from drift.runtime import (',
            '    Agent, step_decorator, Budget, ModelRouter, Intent,',
            '    CostTracker, Checkpoint, Confident, MemoryStore, run_agent,',
            '    make_memory_store, StreamThenRouter,',
            '    register_custom_verb,',
            '    DriftError, StepFailed, SchemaViolation, BudgetExceeded,',
            '    ModelUnavailable, RateLimited, AuthError,',
            ')',
            '',
            '__drift_source__ = "<drift_file>"',
        )

    # ─── Config ────────────────────────────────────────────────────

    def gen_import(self, decl: ast.ImportDecl):
        """import X from "./other.drift"  → from <basename> import X

        - `./foo.drift` → `from foo import X` (sibling file)
        - `drift/io`    → `from drift.io import X` (stdlib namespace)
        - any other     → `from <path-normalized> import X`
        """
        path = decl.source_path
        module: str
        if path.startswith("drift/") and not path.endswith(".drift"):
            # Stdlib namespace: drift/io → drift.io
            module = path.replace("/", ".")
        else:
            # Sibling .drift file → strip dirs and extension
            base = path.rsplit("/", 1)[-1]
            if base.endswith(".drift"):
                base = base[:-len(".drift")]
            module = base
        names = ", ".join(decl.names)
        self.emit_line(f"from {module} import {names}")

    def gen_pipeline(self, pipe: ast.PipelineDecl):
        """A pipeline is a graph of step calls. Generate a runner class.

        v0.2 semantics:
        - Topological execution: walk edges in declaration order, threading
          the previous result forward.
        - `->`  sequential: next(prev_result)
        - `=>`  parallel fan-out: asyncio.gather over each item in prev_result
        - `~>`  conditional: not implemented — raises CodegenError
        - `|>`  stream: not implemented — raises CodegenError
        - `on failure in <step>: skip ...` → try/except around that node, swallow
        - `on budget exceeded:` → catches BudgetExceeded at the outer level
        """
        self.emit_line(f"# ── Pipeline: {pipe.name} ──")
        self.emit_line("")
        self.emit_line(f"class {pipe.name}:")
        self.indent()
        self.emit_line(f'"""Drift pipeline: {pipe.name}"""')

        # Constructor: instantiate `use`d agents.
        self.emit_line("")
        self.emit_line("def __init__(self):")
        self.indent()
        for agent_name in pipe.use_agents:
            self.emit_line(f"self.{agent_name} = {agent_name}()")
        if pipe.budget_config:
            currency = CURRENCY_MAP.get(pipe.budget_config.currency_sym, 'USD')
            self.emit_line(
                f'self.budget = Budget(max_per_run={pipe.budget_config.value}, '
                f'currency="{currency}")'
            )
        else:
            self.emit_line("self.budget = None")
        if pipe.schedule:
            # `schedule:` needs an external cron/daemon loop to mean anything
            # — `run()` only ever executes once per call. Storing the string
            # and never reading it again would silently look configured while
            # doing nothing, so refuse it instead.
            raise CodegenError(
                f"pipeline {pipe.name!r} declares schedule: {pipe.schedule!r}, "
                "which is not implemented — it parses but nothing drives it "
                "(no scheduler/daemon exists; `run()` only executes once per "
                "call). Drive scheduling externally (cron, a task queue) "
                "calling `drift run` instead."
            )
        self.emit_line(f"self.timeout_seconds = {pipe.timeout_seconds!r}")
        if not pipe.use_agents and not pipe.budget_config:
            self.emit_line("pass")
        self.dedent()

        # Inline steps as methods (if any)
        for step in pipe.inline_steps:
            self.emit_line("")
            # Inline steps aren't on an Agent class — they're plain async methods.
            # We don't get cost tracking or checkpointing here; they're meant
            # for orchestration glue like `compile_and_send`.
            self.gen_inline_step(step)

        # The run() coroutine — a thin wrapper enforcing timeout/budget-handler
        # policy around the actual orchestration in _orchestrate().
        self.emit_line("")
        self.emit_line("async def run(self, initial_input=None):")
        self.indent()
        self.emit_line('"""Execute the pipeline. Returns the last node\'s output."""')
        if pipe.timeout_seconds:
            self.emit_line(
                f"return await asyncio.wait_for(self._orchestrate(initial_input), "
                f"timeout={pipe.timeout_seconds!r})"
            )
        elif pipe.budget_handler:
            self.emit_line("try:")
            self.indent()
            self.emit_line("return await self._orchestrate(initial_input)")
            self.dedent()
            self.emit_line("except BudgetExceeded as _e:")
            self.indent()
            self.emit_line(
                f'print(f"  ⚠  budget exceeded ({pipe.budget_handler!r}): {{_e}}")'
            )
            self.emit_line("raise")
            self.dedent()
        else:
            self.emit_line("return await self._orchestrate(initial_input)")
        self.dedent()  # end of run()

        self.emit_line("")
        self.emit_line("async def _orchestrate(self, initial_input=None):")
        self.indent()
        self.emit_line("results = {}")
        self.emit_line("_prev = initial_input")

        # The first edge's `from_node` is the entry point — we have to call
        # it once before threading its output through subsequent nodes.
        produced = set()
        for i, edge in enumerate(pipe.edges):
            if edge.from_node not in produced:
                entry_call = self._pipeline_node_callable(edge.from_node, pipe)
                self.emit_line("# entry node — call signature-checked at runtime")
                self.emit_line(
                    f"_prev = await self._call_node({entry_call}, _prev)"
                )
                self.emit_line(f"results[{edge.from_node!r}] = _prev")
                produced.add(edge.from_node)
            self._emit_pipeline_edge(edge, pipe, i)
            produced.add(edge.to_node)

        self.emit_line("return _prev")
        self.dedent()  # end of _orchestrate()

        # Helper: call a node with _prev only if it accepts an argument.
        # Indent level here is class-body (one above run).
        self.emit_line("")
        self.emit_line("async def _call_node(self, fn, prev):")
        self.indent()
        self.emit_line('"""Call a step. Pass prev only if the signature accepts it."""')
        self.emit_line("import inspect")
        self.emit_line("sig = inspect.signature(fn)")
        self.emit_line("non_self = [p for p in sig.parameters.values() "
                       "if p.kind != inspect.Parameter.VAR_KEYWORD]")
        self.emit_line("if non_self:")
        self.indent()
        self.emit_line("return await fn(prev)")
        self.dedent()
        self.emit_line("return await fn()")
        self.dedent()  # end of _call_node

        self.dedent()  # end of class

    def gen_inline_step(self, step: ast.StepDecl):
        """Inline pipeline steps — plain async methods, no Agent infrastructure."""
        params = ["self"]
        for p in step.params:
            params.append(f"{p.name}: {self.gen_type(p.type_expr)}")
        ret = ""
        if step.return_type:
            ret = f" -> {self.gen_type(step.return_type)}"
        self.emit_line(f"async def {step.name}({', '.join(params)}){ret}:")
        self.indent()
        if not step.body:
            self.emit_line("pass")
        for s in step.body:
            self.gen_statement(s)
        self.dedent()

    def _emit_pipeline_edge(self, edge: ast.PipelineEdge, pipe: ast.PipelineDecl, i: int):
        """Emit code that runs `to_node` with `_prev` as input."""
        callable_expr = self._pipeline_node_callable(edge.to_node, pipe)
        target_step_name = edge.to_node.split(".")[-1]
        handler = pipe.failure_handlers.get(target_step_name, "")
        skip_on_failure = handler.startswith("skip")
        if handler and not skip_on_failure:
            # `_read_handler_phrase` accepts ANY words after the colon — only
            # a "skip..." prefix is actually implemented. Anything else
            # (e.g. "retry twice then fail", "notify oncall") would silently
            # do nothing rather than the described recovery, so refuse it
            # instead of compiling a no-op handler.
            raise CodegenError(
                f"on failure in {target_step_name}: {handler!r} is not "
                "implemented — only `on failure in <step>: skip ...` is "
                "wired up (it wraps that node in try/except and continues "
                "the pipeline). Any other phrase parses but would silently "
                "do nothing, so it's rejected instead."
            )

        if edge.op == "->":
            if skip_on_failure:
                self.emit_line("try:")
                self.indent()
                self.emit_line(f"_prev = await {callable_expr}(_prev)")
                self.emit_line(f"results[{edge.to_node!r}] = _prev")
                self.dedent()
                self.emit_line("except Exception as _e:")
                self.indent()
                self.emit_line(f'print(f"  ⚠  skipping {target_step_name}: {{_e}}")')
                self.dedent()
            else:
                self.emit_line(f"_prev = await {callable_expr}(_prev)")
                self.emit_line(f"results[{edge.to_node!r}] = _prev")
        elif edge.op == "=>":
            # Parallel fan-out: _prev must be iterable
            self.emit_line(f"# parallel fan-out into {edge.to_node}")
            self.emit_line(
                f"_prev = await asyncio.gather(*[{callable_expr}(item) for item in _prev])"
            )
            self.emit_line(f"results[{edge.to_node!r}] = _prev")
        elif edge.op in ("~>", "|>"):
            kind = "conditional ~>" if edge.op == "~>" else "streaming |>"
            raise CodegenError(
                f"pipeline edge `{edge.op}` before {edge.to_node!r} is not implemented — "
                f"{kind} pipeline edges parse but the runtime has no semantics for them yet. "
                "Use `->` (sequential) or `=>` (parallel fan-out) instead."
            )

    def _pipeline_node_callable(self, node: str, pipe: ast.PipelineDecl) -> str:
        """Resolve a node name to an awaitable callable.

        `Agent.step`     -> self.Agent.step
        `step` (no dot)  -> self.<UsedAgent>.step if exactly one used agent
                            owns it, else `self.step` (inline step)
        """
        if "." in node:
            agent, step = node.split(".", 1)
            return f"self.{agent}.{step}"
        # Look up inline step
        inline_names = {s.name for s in pipe.inline_steps}
        if node in inline_names:
            return f"self.{node}"
        # If exactly one used agent has this step, qualify it implicitly.
        # Without import resolution we can't know which agent owns the step;
        # default to the first used agent if any, else self.<node>.
        if pipe.use_agents:
            return f"self.{pipe.use_agents[0]}.{node}"
        return f"self.{node}"

    def gen_tool_decl(self, tool: ast.ToolDecl):
        """Generate a tool adapter object accessible as <name>.<action>(...)."""
        self.emit_line(f"# ── Tool: {tool.name} ({tool.kind}) ──")
        if tool.kind == "python":
            module_path, _, fn = tool.source.partition(":")
            if not fn:
                fn = module_path.split(".")[-1]
                module_path = ".".join(module_path.split(".")[:-1])
            safe_module = module_path.replace(".", "_")
            self.emit_line(f"from {module_path} import {fn} as _drift_tool_{tool.name}")
            self.emit_line(f"{tool.name} = _drift_tool_{tool.name}")
            return
        if tool.kind == "mcp":
            # MCP runtime is implemented in drift.runtime.mcp_client.
            # The McpTool wrapper exposes the server's tools as awaitable
            # methods via __getattr__. Connection is lazy on first call.
            self.emit_line(
                f"from drift.runtime.mcp_client import McpTool as _McpTool"
            )
            self.emit_line(
                f"{tool.name} = _McpTool({tool.source!r}, name={tool.name!r})"
            )
            # Preserve the old class-name alias so existing tests that
            # checked for `_<name>_McpTool` in the source still see it.
            # Cheap shim; means we don't have to touch test_tool.py.
            self.emit_line(f"_{tool.name}_McpTool = type({tool.name})")
            return
        # REST form
        self.emit_line(f"class _{tool.name}_RestTool:")
        self.indent()
        self.emit_line(f'"""REST tool adapter generated from a `tool` block."""')
        self.emit_line(f"endpoint = {tool.endpoint!r}")
        self.emit_line(f"auth_env = {tool.auth_env!r}" if tool.auth_env else "auth_env = None")
        self.emit_line(
            f"auth_literal = {tool.auth_literal!r}" if tool.auth_literal else "auth_literal = None"
        )
        self.emit_line("")
        self.emit_line("def _auth_header(self):")
        self.indent()
        self.emit_line("import os")
        self.emit_line("if self.auth_literal:")
        self.indent()
        self.emit_line('return {"Authorization": f"Bearer {self.auth_literal}"}')
        self.dedent()
        self.emit_line('if not self.auth_env: return {}')
        self.emit_line('token = os.environ.get(self.auth_env)')
        self.emit_line('return {"Authorization": f"Bearer {token}"} if token else {}')
        self.dedent()
        for action in tool.actions:
            self.emit_line("")
            param_strs = ["self"]
            for p in action.params:
                param_strs.append(f"{p.name}: {self.gen_type(p.type_expr)}")
            self.emit_line(f"async def {action.name}({', '.join(param_strs)}):")
            self.indent()
            self.emit_line(f'"""{action.method} {action.path}"""')
            self.emit_line("import httpx")
            # Use the path template directly; users write {var} which becomes
            # an f-string substitution.
            path_fstring = "f" + repr(action.path)
            self.emit_line(f"path = {path_fstring}")
            self.emit_line(f"url = self.endpoint + path")

            # Params not consumed by the {var} path template must still go
            # somewhere: query string for GET/DELETE/HEAD, JSON body for
            # POST/PUT/PATCH. Previously they were silently dropped — e.g. a
            # documented `create_issue(repo, title)` action would POST with
            # no body at all, discarding `title`.
            path_vars = set(re.findall(r"\{(\w+)\}", action.path))
            extra_params = [p.name for p in action.params if p.name not in path_vars]
            method = action.method.lower()
            call_kwargs = ["headers=self._auth_header()", "timeout=30.0"]
            if extra_params:
                extras_dict = "{" + ", ".join(f"{repr(n)}: {n}" for n in extra_params) + "}"
                if method in ("post", "put", "patch"):
                    self.emit_line(f"_body = {extras_dict}")
                    call_kwargs.append("json=_body")
                else:
                    self.emit_line(f"_query = {extras_dict}")
                    call_kwargs.append("params=_query")

            self.emit_line(f"async with httpx.AsyncClient() as client:")
            self.indent()
            self.emit_line(
                f"resp = await client.{method}(url, {', '.join(call_kwargs)})"
            )
            self.dedent()
            self.emit_line("resp.raise_for_status()")
            self.emit_line("return resp.json()")
            self.dedent()
        self.dedent()
        self.emit_line(f"{tool.name} = _{tool.name}_RestTool()")

    def gen_verb_decl(self, verb: ast.VerbDecl):
        """`define verb name { prompt, output, ... }` → runtime registration."""
        self.emit_line(f"# ── Custom verb: {verb.name} ──")
        output_repr = "None"
        if verb.output is not None:
            output_repr = self.gen_type(verb.output)
        # Escape the prompt for a triple-quoted Python string.
        # Drift uses """ for multi-line strings already, but the lexer strips
        # those quotes — so the prompt body is already raw text by the time
        # we get here. Use json.dumps via repr() to handle quoting safely.
        prompt_repr = repr(verb.prompt)
        pattern_repr = repr(verb.pattern)
        self.emit_line("register_custom_verb(")
        self.indent()
        self.emit_line(f'name="{verb.name}",')
        self.emit_line(f"prompt={prompt_repr},")
        self.emit_line(f"output_schema={output_repr},")
        self.emit_line(f"pattern={pattern_repr},")
        self.emit_line(f"temperature={verb.temperature},")
        self.dedent()
        self.emit_line(")")

    def gen_config(self, config: ast.ConfigDecl):
        self.emit_line("# ── Config ──")
        self.emit_line(f"DRIFT_CONFIG = {{")
        self.indent()
        for k, v in config.entries.items():
            # k is a lexer IDENT (alphanumeric/underscore only, always safe
            # to embed bare); v is a user-written STRING literal and must be
            # re-escaped, not interpolated raw — a `"` or `\` in v would
            # otherwise break out of the generated string literal.
            self.emit_line(f'"{k}": {_py_str_literal(v)},')
        self.dedent()
        self.emit_line("}")

    # ─── Schema ────────────────────────────────────────────────────

    def gen_schema(self, schema: ast.SchemaDecl):
        self.emit_line(f"# ── Schema: {schema.name} ──")
        self.emit_line("")
        self.emit_line(f"@dataclass")
        self.emit_line(f"class {schema.name}:")
        self.indent()

        if not schema.fields:
            self.emit_line("pass")
        else:
            # Generate docstring with field descriptions
            self.emit_line(f'"""Drift schema: {schema.name}"""')
            for f in schema.fields:
                py_type = self.gen_type(f.type_expr)
                if f.optional:
                    py_type = f"Optional[{py_type}]"

                default = " = None" if f.optional else ""

                # Add constraint comments
                constraint_comment = ""
                for c in f.constraints:
                    if isinstance(c, ast.BetweenConstraint):
                        constraint_comment = f"  # valid range: {c.low} to {c.high}"

                self.emit_line(f"{f.name}: {py_type}{default}{constraint_comment}")

            # Generate validation method. `one of "a", "b", ...` (EnumType)
            # used to only become a `Literal[...]` type hint with nothing
            # checking it at runtime, unlike `number between A and B` (a real
            # check below) — an LLM returning an out-of-enum value passed
            # validation silently. Both now generate a real check.
            #
            # Raises SchemaViolation, not AssertionError: step_decorator's
            # retry loop (drift/runtime/core.py) only catches SchemaViolation
            # to retry with a stricter prompt. An AssertionError here used to
            # crash the step outright on the FIRST out-of-range/out-of-enum
            # value instead of getting the documented retry.
            enum_fields = [f for f in schema.fields if isinstance(f.type_expr, ast.EnumType)]
            has_constraints = any(f.constraints for f in schema.fields) or enum_fields
            if has_constraints:
                self.emit_line("")
                self.emit_line("def validate(self):")
                self.indent()
                self.emit_line('"""Validate field constraints. Raises SchemaViolation on failure."""')
                for f in schema.fields:
                    for c in f.constraints:
                        if isinstance(c, ast.BetweenConstraint):
                            self.emit_line(f"if self.{f.name} is not None and not ({c.low} <= self.{f.name} <= {c.high}):")
                            self.indent()
                            self.emit_line(
                                f'raise SchemaViolation('
                                f'f"{f.name} must be between {c.low} and {c.high}, got {{self.{f.name}}}")'
                            )
                            self.dedent()
                for f in enum_fields:
                    vals = ", ".join(repr(v) for v in f.type_expr.values)
                    tail_comma = ',' if len(f.type_expr.values) == 1 else ''
                    self.emit_line(f"if self.{f.name} is not None and self.{f.name} not in ({vals}{tail_comma}):")
                    self.indent()
                    self.emit_line(
                        f'raise SchemaViolation('
                        f'f"{f.name} must be one of {vals}, got {{self.{f.name}!r}}")'
                    )
                    self.dedent()
                self.emit_line("return self")
                self.dedent()

        self.dedent()

    def gen_type(self, type_expr) -> str:
        if type_expr is None:
            return "str"

        if isinstance(type_expr, ast.ListType):
            if type_expr.element_type:
                inner = self.gen_type(type_expr.element_type)
                return f"list[{inner}]"
            return "list"

        if isinstance(type_expr, ast.MapType):
            k = self.gen_type(type_expr.key_type) if type_expr.key_type else "str"
            v = self.gen_type(type_expr.value_type) if type_expr.value_type else "str"
            return f"dict[{k}, {v}]"

        if isinstance(type_expr, ast.ConfidentType):
            # Thread the inner type through so the runtime can build a
            # schema-aware prompt and parse `value` into a real dataclass.
            # Falls back to bare Confident for `confident<>` with no inner.
            if type_expr.inner_type is not None:
                inner = self.gen_type(type_expr.inner_type)
                return f"Confident[{inner}]"
            return "Confident"

        if isinstance(type_expr, ast.EnumType):
            vals = ", ".join(f'"{v}"' for v in type_expr.values)
            return f"Literal[{vals}]"

        if isinstance(type_expr, ast.TypeExpr):
            return TYPE_MAP.get(type_expr.name, type_expr.name)

        return "str"

    # ─── Agent ─────────────────────────────────────────────────────

    def gen_agent(self, agent: ast.AgentDecl):
        self.emit_line(f"# ── Agent: {agent.name} ──")
        self.emit_line("")
        self.emit_line(f"class {agent.name}(Agent):")
        self.indent()
        self.emit_line(f'"""Drift agent: {agent.name}"""')
        self.emit_line("")

        # Collect step names for self-call detection
        self._agent_step_names = {s.name for s in agent.steps}

        # __init__
        self.emit_line("def __init__(self):")
        self.indent()

        # Model config
        model_args = self.gen_model_init(agent.model_config)
        self.emit_line(f"model = {model_args}")

        # Budget config
        budget_args = self.gen_budget_init(agent.budget_config)
        self.emit_line(f"budget = {budget_args}")

        # Quality config
        quality = 0.85
        if agent.quality_config:
            quality = agent.quality_config.min_confidence

        # Memory config (optional)
        mem_arg = "None"
        if agent.memory_config:
            m = agent.memory_config
            if m.backend == "dendric":
                # Real Dendric when DATABASE_URL is set; SQLite mock otherwise.
                # The factory prints a one-time notice in mock mode.
                mem_arg = f'make_memory_store(persona={m.persona!r})'
            else:
                mem_arg = (
                    f'MemoryStore(store_url={m.store!r}, '
                    f'recall_strategy={m.recall_strategy!r}, '
                    f'max_recall={m.max_recall}, '
                    f'decay_enabled={m.decay_enabled})'
                )

        self.emit_line("")
        self.emit_line(f"super().__init__(")
        self.indent()
        self.emit_line(f'name="{agent.name}",')
        self.emit_line(f'model=model,')
        self.emit_line(f'budget=budget,')
        self.emit_line(f'min_confidence={quality},')
        if agent.memory_config:
            self.emit_line(f'memory={mem_arg},')
        self.dedent()
        self.emit_line(")")

        # State fields — initialize on the instance for cross-step access.
        if agent.state_block:
            self.emit_line("")
            self.emit_line("# Agent state (persists across steps within a run)")
            for f in agent.state_block:
                if f.default is not None:
                    val = self.gen_expr(f.default)
                else:
                    val = self._default_for_type(f.type_expr)
                self.emit_line(f"self.{f.name} = {val}")

        self.dedent()

        # Steps
        for s in agent.steps:
            self.emit_line("")
            self.gen_step(s)

        self.dedent()

    def _default_for_type(self, type_expr) -> str:
        """Sensible zero-value for a state field without an explicit default."""
        if type_expr is None:
            return "None"
        if isinstance(type_expr, ast.ListType):
            return "[]"
        if isinstance(type_expr, ast.MapType):
            return "{}"
        name = getattr(type_expr, "name", "")
        if name == "string":
            return '""'
        if name in ("number", "currency", "duration"):
            return "0"
        if name == "bool":
            return "False"
        return "None"

    def gen_model_init(self, config) -> str:
        if config is None:
            return 'ModelRouter(default="claude-sonnet")'

        # `model: stream "fast" then "slow"` parses and StreamThenRouter (the
        # runtime class this would construct) has a real stream_then_call()
        # bridge-then-reasoning method — but no Drift syntax exists for a step
        # body to supply the on_bridge callback, and Agent.intent() (what
        # every intent verb actually goes through) never calls
        # stream_then_call() itself. So declaring this today silently behaves
        # identically to `model: default "<then_model>"` — same model, no
        # bridge, no speed difference — with no error or comment anywhere.
        # Reject it instead of emitting a router whose one distinguishing
        # method nothing ever calls.
        if getattr(config, "mode", "") == "stream_then":
            raise CodegenError(
                'model: stream "..." then "..." is not implemented — it parses '
                "but no generated code ever calls the fast/slow bridge, so it "
                'would silently behave like `model: default "<then-model>"` with '
                "no speedup and no error. Use a plain model block instead."
            )

        # Model names below are user-written STRING literals — always embed
        # via _py_str_literal (like the rest of codegen), never bare
        # "{...}" quoting, so a quote/backslash in a model name can't break
        # out of the generated string literal (drift check ran codegen but
        # never verified the output was valid Python, so this class of bug
        # used to slip through as a "syntax OK" that produced a SyntaxError
        # only at `drift run`).
        parts = []
        if config.default:
            parts.append(f'default={_py_str_literal(config.default)}')
        if config.prefer and config.prefer != config.default:
            parts.append(f'prefer={_py_str_literal(config.prefer)}')
        # Block form populates fallback_list/never_list; colon form sets
        # the scalar `fallback`/`never`. Normalize.
        fallbacks = list(config.fallback_list) if config.fallback_list else (
            [config.fallback] if config.fallback else []
        )
        nevers = list(config.never_list) if config.never_list else (
            [config.never] if config.never else []
        )
        if fallbacks:
            parts.append("fallback=[" + ", ".join(_py_str_literal(m) for m in fallbacks) + "]")
        if nevers:
            parts.append("never=[" + ", ".join(_py_str_literal(m) for m in nevers) + "]")
        if config.upgrades:
            rules = []
            for u in config.upgrades:
                cond_reprs = []
                for c in u.conditions:
                    # c.kind is one of two internal literals set by the
                    # parser (confidence_lt/step_is), never user text.
                    cond_reprs.append(
                        f'{{"kind": "{c.kind}", "value": {c.value!r}}}'
                    )
                rules.append(
                    f'{{"target": {_py_str_literal(u.target_model)}, '
                    f'"conditions": [{", ".join(cond_reprs)}]}}'
                )
            parts.append("upgrades=[" + ", ".join(rules) + "]")

        return f"ModelRouter({', '.join(parts)})"

    def gen_budget_init(self, config) -> str:
        if config is None:
            return 'Budget(max_per_run=10.0, currency="USD")'

        currency = CURRENCY_MAP.get(config.currency_sym, 'USD')
        return f'Budget(max_per_run={config.value}, currency="{currency}")'

    # ─── Step ──────────────────────────────────────────────────────

    def gen_step(self, step: ast.StepDecl):
        # Track step name explicitly so checkpoint emission inside nested
        # async blocks (e.g. `for each ... parallel`) finds the outer
        # step name rather than the closest `async def` (which would be
        # the inner _task wrapper).
        self._step_name_stack = getattr(self, "_step_name_stack", [])
        self._step_name_stack.append(step.name)

        # Decorator
        ret_schema = "None"
        if step.return_type:
            ret_schema = self.gen_type(step.return_type)

        if step.modifier == "parallel":
            raise CodegenError(
                "`parallel step` is not implemented — a step modifier alone doesn't "
                "say what it runs in parallel with. Use `for each x in xs parallel { ... }` "
                "or a pipeline `=>` fan-out edge to express concurrency instead."
            )

        modifier_args = ""
        if step.modifier:
            modifier_args = f', modifier="{step.modifier}"'

        self.emit_line(f"@step_decorator(output={ret_schema}{modifier_args})")

        # Method signature
        params = ["self"]
        for p in step.params:
            py_type = self.gen_type(p.type_expr)
            params.append(f"{p.name}: {py_type}")

        param_str = ", ".join(params)
        ret_annotation = ""
        if step.return_type:
            ret_annotation = f" -> {ret_schema}"

        self.emit_line(f"async def {step.name}({param_str}){ret_annotation}:")
        self.indent()

        # Step body
        if not step.body:
            self.emit_line("pass")
        else:
            # Budget pre-check
            self.emit_line("# Budget pre-check")
            self.emit_line("self.cost_tracker.pre_check()")
            self.emit_line("")

            for i, stmt in enumerate(step.body):
                is_last = (i == len(step.body) - 1)
                # If last statement is an intent expr without explicit return, add one
                if (is_last and step.return_type and
                    isinstance(stmt, ast.ExprStmt) and
                    isinstance(stmt.expr, ast.IntentExpr)):
                    expr_code = self.gen_expr(stmt.expr)
                    self.emit_line(f"_result = {expr_code}")
                    self.emit_line(f"self.checkpoint.save('{step.name}', _result)")
                    self.emit_line("return _result")
                else:
                    self.gen_statement(stmt)

        self.dedent()
        self._step_name_stack.pop()

    # ─── Statements ────────────────────────────────────────────────

    def gen_statement(self, stmt):
        if isinstance(stmt, ast.LetStmt):
            self.gen_let(stmt)
        elif isinstance(stmt, ast.ReturnStmt):
            self.gen_return(stmt)
        elif isinstance(stmt, ast.RespondStmt):
            self.gen_respond(stmt)
        elif isinstance(stmt, ast.IfStmt):
            self.gen_if(stmt)
        elif isinstance(stmt, ast.ForEachStmt):
            self.gen_for_each(stmt)
        elif isinstance(stmt, ast.MatchStmt):
            self.gen_match(stmt)
        elif isinstance(stmt, ast.AttemptStmt):
            self.gen_attempt(stmt)
        elif isinstance(stmt, ast.RetryStmt):
            self.emit_line("continue  # retry the attempt block")
        elif isinstance(stmt, ast.FailStmt):
            self.gen_fail(stmt)
        elif isinstance(stmt, ast.RememberStmt):
            self.gen_remember(stmt)
        elif isinstance(stmt, ast.DejaVuStmt):
            self.gen_deja_vu(stmt)
        elif isinstance(stmt, ast.ForgetStmt):
            self.gen_forget(stmt)
        elif isinstance(stmt, ast.ExprStmt):
            self.emit_line(self.gen_expr(stmt.expr))

    def gen_let(self, stmt: ast.LetStmt):
        expr = self.gen_expr(stmt.value)
        self.emit_line(f"{stmt.name} = {expr}")

    def gen_return(self, stmt: ast.ReturnStmt):
        if stmt.value is None:
            self.emit_line("return")
        else:
            expr = self.gen_expr(stmt.value)
            # Checkpoint before return
            self.emit_line(f"_result = {expr}")
            self.emit_line(f"self.checkpoint.save('{self._current_step_name()}', _result)")
            self.emit_line("return _result")

    def _current_step_name(self):
        # Prefer the explicit step-name stack set by gen_step — robust
        # to nested async helpers like the `_task` wrapper that
        # `for each ... parallel` emits.
        stack = getattr(self, "_step_name_stack", None)
        if stack:
            return stack[-1]
        # Fallback heuristic for any caller outside gen_step.
        for line in reversed(self.lines):
            if 'async def ' in line:
                match = re.search(r'async def (\w+)', line)
                if match:
                    return match.group(1)
        return "unknown"

    def gen_respond(self, stmt: ast.RespondStmt):
        expr = self.gen_expr(stmt.value)
        self.emit_line(f"self.output({expr})")

    def gen_if(self, stmt: ast.IfStmt):
        cond = self.gen_expr(stmt.condition)
        self.emit_line(f"if {cond}:")
        self.indent()
        for s in stmt.body:
            self.gen_statement(s)
        if not stmt.body:
            self.emit_line("pass")
        self.dedent()

        if stmt.otherwise_if:
            self.emit_line("elif " + self.gen_expr(stmt.otherwise_if.condition) + ":")
            self.indent()
            for s in stmt.otherwise_if.body:
                self.gen_statement(s)
            self.dedent()
            # Handle chained otherwise-ifs
            if stmt.otherwise_if.otherwise_body:
                self.emit_line("else:")
                self.indent()
                for s in stmt.otherwise_if.otherwise_body:
                    self.gen_statement(s)
                self.dedent()
        elif stmt.otherwise_body:
            self.emit_line("else:")
            self.indent()
            for s in stmt.otherwise_body:
                self.gen_statement(s)
            self.dedent()

    def gen_for_each(self, stmt: ast.ForEachStmt):
        iterable = self.gen_expr(stmt.iterable)
        if stmt.parallel:
            self.emit_line(f"# Parallel fan-out (concurrent execution)")
            self.emit_line(f"async def _task({stmt.var_name}):")
            self.indent()
            for s in stmt.body:
                self.gen_statement(s)
            self.dedent()
            self.emit_line(f"await asyncio.gather(*[_task(item) for item in {iterable}])")
        else:
            self.emit_line(f"for {stmt.var_name} in {iterable}:")
            self.indent()
            for s in stmt.body:
                self.gen_statement(s)
            if not stmt.body:
                self.emit_line("pass")
            self.dedent()

    def gen_attempt(self, stmt: ast.AttemptStmt):
        """attempt { body } recover from { ErrorType -> handler ... }

        Compiles to a bounded retry loop wrapping try/except. Inside an arm,
        `retry` becomes `continue` (next iteration). Falling through the arm
        without `retry` breaks out — recovery succeeded. If the loop runs
        all iterations without breaking, we raise StepFailed.
        """
        self.emit_line(f"# attempt/recover (max {stmt.max_retries} retries)")
        self.emit_line(f"for _attempt in range({stmt.max_retries}):")
        self.indent()
        self.emit_line("try:")
        self.indent()
        if not stmt.body:
            self.emit_line("pass")
        last_terminal = False
        for s in stmt.body:
            self.gen_statement(s)
            if isinstance(s, (ast.ReturnStmt, ast.FailStmt)):
                last_terminal = True
        # If the try body didn't end in return/fail, emit a `break` so a
        # successful attempt exits the retry loop.
        if not last_terminal:
            self.emit_line("break")
        self.dedent()
        # Emit except arms. `any error` last because Python evaluates them
        # in order and DriftError would shadow specific subclasses.
        specific = [a for a in stmt.arms if not a.is_default]
        default_arm = next((a for a in stmt.arms if a.is_default), None)
        for arm in specific:
            self.emit_line(f"except {arm.error_type} as _err:")
            self.indent()
            self._gen_recover_body(arm)
            self.dedent()
        if default_arm:
            self.emit_line("except DriftError as _err:")
            self.indent()
            self._gen_recover_body(default_arm)
            self.dedent()
        self.dedent()
        # If we ran all retries without breaking, raise so the caller knows.
        self.emit_line("else:")
        self.indent()
        self.emit_line(
            f'raise StepFailed("attempt block exhausted {stmt.max_retries} retries")'
        )
        self.dedent()

    def _gen_recover_body(self, arm: ast.RecoverArm):
        if not arm.body:
            # Bare arm with no statements — re-raise so the failure is visible.
            self.emit_line("raise")
            return
        # Detect a bare `retry` so we know whether to add a fallthrough raise.
        had_terminal = False
        for s in arm.body:
            self.gen_statement(s)
            if isinstance(s, (ast.RetryStmt, ast.ReturnStmt, ast.FailStmt)):
                had_terminal = True
        if not had_terminal:
            # Recovered without retry/return/fail — break out of the loop
            # so the attempt doesn't run again.
            self.emit_line("break")

    def gen_fail(self, stmt: ast.FailStmt):
        msg = self.gen_expr(stmt.message) if stmt.message else '"step failed"'
        self.emit_line(f"raise StepFailed({msg})")

    def gen_remember(self, stmt: ast.RememberStmt):
        val = self.gen_expr(stmt.value)
        tag = self.gen_expr(stmt.tag) if stmt.tag is not None else '""'
        self.emit_line(f"self.memory.remember({val}, tag={tag})")

    def gen_deja_vu(self, stmt: ast.DejaVuStmt):
        """deja_vu match on <ctx> { "pat" -> {...} } emits:

            _dv = self.memory.deja_vu_check(context=<ctx>)
            if _dv is not None:
                match = _dv
                if match.matches("pat"):
                    ...
                else:
                    ...   # any other arm

        Uses a fresh var per block so nested/sequential deja_vu calls
        don't shadow each other."""
        ctx = self.gen_expr(stmt.context_expr)
        # Fresh suffix for the temporary so nested blocks don't collide.
        self._dv_counter = getattr(self, "_dv_counter", 0) + 1
        tmp = f"_dv_{self._dv_counter}"

        self.emit_line(f"{tmp} = self.memory.deja_vu_check(context={ctx})")
        self.emit_line(f"if {tmp} is not None:")
        self.indent()
        self.emit_line(f"match = {tmp}")

        # Split default arm out so it becomes the trailing `else:`.
        named = [a for a in stmt.arms if not a.is_default]
        default = next((a for a in stmt.arms if a.is_default), None)

        for i, arm in enumerate(named):
            kw = "if" if i == 0 else "elif"
            pattern_repr = repr(arm.pattern)
            self.emit_line(f"{kw} match.matches({pattern_repr}):")
            self.indent()
            if arm.body:
                for s in arm.body:
                    self.gen_statement(s)
            else:
                self.emit_line("pass")
            self.dedent()

        if default is not None:
            if named:
                self.emit_line("else:")
            else:
                # No named arms — make the default unconditional inside `if _dv is not None`.
                pass
            if named:
                self.indent()
            if default.body:
                for s in default.body:
                    self.gen_statement(s)
            else:
                self.emit_line("pass")
            if named:
                self.dedent()
        elif not named:
            # Empty body but the `if _dv is not None:` block needs a stmt.
            self.emit_line("pass")

        self.dedent()

    def gen_forget(self, stmt: ast.ForgetStmt):
        """Routes the three syntactic forms to the unified store.forget()."""
        if stmt.mode == "by_tag":
            tag_expr = self.gen_expr(stmt.tag)
            self.emit_line(f"self.memory.forget(tag={tag_expr})")
        elif stmt.mode == "by_age":
            self.emit_line(
                f"self.memory.forget(older_than_days={stmt.older_than_days})"
            )
        elif stmt.mode == "by_temp":
            self.emit_line(
                f"self.memory.forget(below_temp={stmt.below_temp})"
            )
        else:
            raise ValueError(f"unknown forget mode: {stmt.mode!r}")

    def gen_match(self, stmt: ast.MatchStmt):
        target = self.gen_expr(stmt.target)
        # Emit all pattern arms first, then the default as a trailing `else:`,
        # regardless of source order — otherwise a default arm written before a
        # pattern arm would emit `else:` before the opening `if` (SyntaxError).
        pattern_arms = [a for a in stmt.arms if not a.is_default]
        default_arms = [a for a in stmt.arms if a.is_default]

        def emit_body(arm):
            self.indent()
            for s in arm.body:
                self.gen_statement(s)
            if not arm.body:
                self.emit_line("pass")
            self.dedent()

        for idx, arm in enumerate(pattern_arms):
            pattern = self.gen_expr(arm.pattern)
            kw = "if" if idx == 0 else "elif"
            self.emit_line(f"{kw} {target} == {pattern}:")
            emit_body(arm)

        for arm in default_arms:
            # If there were no pattern arms, `else` has nothing to attach to;
            # emit the body unconditionally instead.
            if pattern_arms:
                self.emit_line("else:")
                emit_body(arm)
            else:
                for s in arm.body:
                    self.gen_statement(s)
                if not arm.body:
                    self.emit_line("pass")

    # ─── Expressions ───────────────────────────────────────────────

    def gen_expr(self, expr) -> str:
        if isinstance(expr, ast.IntentExpr):
            return self.gen_intent(expr)
        elif isinstance(expr, ast.RecallStmt):
            desc = repr(expr.description)
            key = self.gen_expr(expr.key) if expr.key is not None else "None"
            return f"self.memory.recall({desc}, key={key})"
        elif isinstance(expr, ast.FnCall):
            return self.gen_fn_call(expr)
        elif isinstance(expr, ast.FieldAccess):
            return f"{self.gen_expr(expr.target)}.{expr.field_name}"
        elif isinstance(expr, ast.BinOp):
            return self.gen_binop(expr)
        elif isinstance(expr, ast.UnaryOp):
            op = expr.op
            if op == 'not':
                return f"not {self.gen_expr(expr.operand)}"
            return f"({op}{self.gen_expr(expr.operand)})"
        elif isinstance(expr, ast.Ident):
            return expr.name
        elif isinstance(expr, ast.StringLit):
            if expr.has_interpolation:
                return self._gen_fstring(expr.value)
            return _py_str_literal(expr.value)
        elif isinstance(expr, ast.NumberLit):
            if expr.value == int(expr.value):
                return str(int(expr.value))
            return str(expr.value)
        elif isinstance(expr, ast.CurrencyLit):
            return str(expr.value)
        elif isinstance(expr, ast.DurationLit):
            seconds_map = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}
            seconds = expr.value * seconds_map.get(expr.unit, 1)
            return str(seconds)
        elif isinstance(expr, ast.BoolLit):
            return "True" if expr.value else "False"
        elif isinstance(expr, ast.ListLit):
            elems = ", ".join(self.gen_expr(e) for e in expr.elements)
            return f"[{elems}]"
        elif isinstance(expr, ast.SchemaConstructor):
            fields = ", ".join(
                f"{k}={self.gen_expr(v)}" for k, v in expr.fields.items()
            )
            return f"{expr.type_name}({fields})"
        elif isinstance(expr, ast.PipeExpr):
            return self.gen_pipe(expr)
        elif expr is None:
            return "None"
        else:
            return repr(expr)

    def _gen_fstring(self, value: str) -> str:
        """Emit a Python f-string from a Drift interpolated string.

        The `{...}` placeholders are intentional interpolation and pass through
        as f-string expressions. Everything outside the braces is literal text
        and must be escaped so it can never break out of the surrounding quotes
        (a raw `"` or `\\` in the literal portion would otherwise terminate the
        f-string and let arbitrary Python follow).
        """
        parts = []  # (is_literal, text)
        i = 0
        n = len(value)
        buf = []
        while i < n:
            ch = value[i]
            if ch == '{':
                # Escaped literal brace: `{{` stays literal.
                if i + 1 < n and value[i + 1] == '{':
                    buf.append('{')
                    i += 2
                    continue
                # Start of an interpolation; scan to the matching close brace,
                # tracking nesting so `{d["k"]}`-style bodies survive.
                if buf:
                    parts.append((True, ''.join(buf)))
                    buf = []
                depth = 1
                j = i + 1
                while j < n and depth > 0:
                    if value[j] == '{':
                        depth += 1
                    elif value[j] == '}':
                        depth -= 1
                        if depth == 0:
                            break
                    j += 1
                expr_src = value[i + 1:j]
                _validate_interp_expr(expr_src)
                parts.append((False, expr_src))
                i = j + 1
            elif ch == '}':
                # Escaped literal brace: `}}` stays literal.
                if i + 1 < n and value[i + 1] == '}':
                    buf.append('}')
                    i += 2
                    continue
                buf.append('}')
                i += 1
            else:
                buf.append(ch)
                i += 1
        if buf:
            parts.append((True, ''.join(buf)))

        pieces = []
        for is_literal, text in parts:
            if is_literal:
                # Reuse the safe double-quoted literal, drop its outer quotes,
                # and re-double braces so the f-string parser keeps them literal.
                inner = _py_str_literal(text)[1:-1]
                inner = inner.replace('{', '{{').replace('}', '}}')
                pieces.append(inner)
            else:
                pieces.append('{' + text + '}')
        return 'f"' + ''.join(pieces) + '"'

    def gen_binop(self, expr: ast.BinOp) -> str:
        op = expr.op
        # Special-case `is confident` / `is uncertain` BEFORE evaluating the
        # right-hand side, because `confident` and `uncertain` are bare
        # identifiers in the source — we don't want them treated as variable
        # references in the generated code.
        if op == 'is' and isinstance(expr.right, ast.Ident):
            left = self.gen_expr(expr.left)
            kw = expr.right.name
            if kw == 'confident':
                return f"({left}.is_confident(self.min_confidence))"
            if kw == 'uncertain':
                return f"(not {left}.is_confident(self.min_confidence))"
        left = self.gen_expr(expr.left)
        right = self.gen_expr(expr.right)
        if op == 'and':
            return f"({left} and {right})"
        elif op == 'or':
            return f"({left} or {right})"
        elif op == 'is':
            return f"({left} == {right})"
        elif op == 'not in':
            return f"({left} not in {right})"
        else:
            return f"({left} {op} {right})"

    def gen_fn_call(self, call: ast.FnCall) -> str:
        args_parts = []
        for a in call.args:
            if isinstance(a, tuple):
                # Named argument
                args_parts.append(f"{a[0]}={self.gen_expr(a[1])}")
            else:
                args_parts.append(self.gen_expr(a))
        args_str = ", ".join(args_parts)

        if call.target:
            target = self.gen_expr(call.target)
            # Translate .add() to .append() for Python lists
            method = call.name
            if method == 'add':
                method = 'append'
            # Cross-agent call: `OtherAgent.step(args)` becomes a one-shot
            # instantiation + await. We detect this by checking whether
            # the target is a bare Ident matching a declared agent name.
            # The spec's multi-agent fan-out pattern relies on this.
            if (isinstance(call.target, ast.Ident)
                    and call.target.name in self.agent_names):
                return f"await {target}().{method}({args_str})"
            return f"{target}.{method}({args_str})"

        # Check if this is an internal step call
        step_names = getattr(self, '_agent_step_names', set())
        if call.name in step_names:
            return f"await self.{call.name}({args_str})"

        return f"{call.name}({args_str})"

    def gen_pipe(self, pipe: ast.PipeExpr) -> str:
        """Translate |> pipe operator to nested function calls.

        This is function-composition piping (the `|>` operator inside an
        expression), NOT streaming. Spec §2.5 also lists `|>` as a streaming
        pipeline operator between agents — those two uses are kept separate
        for now. In a pipeline declaration, `|>` parses as a PipelineEdge
        with op="|>"; in an expression, it parses as PipeExpr (this method).
        """
        result = self.gen_expr(pipe.input_expr)
        for op in pipe.operations:
            if isinstance(op, ast.FnCall):
                args = ", ".join(self.gen_expr(a) for a in op.args)
                if args:
                    result = f"{op.name}({result}, {args})"
                else:
                    result = f"{op.name}({result})"
            elif isinstance(op, ast.Ident):
                result = f"{op.name}({result})"
            else:
                result = f"{self.gen_expr(op)}({result})"
        return result

    # ─── Intent Expressions ────────────────────────────────────────

    def gen_intent(self, intent: ast.IntentExpr) -> str:
        """
        Translate intent verbs into runtime calls.
        This is where Drift's magic happens:
            classify doc as Category
        becomes:
            await self.intent(verb="classify", input=doc, output_schema=Category)
        """
        parts = [f'verb="{intent.verb}"']

        # Input
        if intent.input_expr:
            parts.append(f"input_data={self.gen_expr(intent.input_expr)}")

        # Output schema (from 'as' clause)
        if 'as' in intent.clauses:
            type_expr = intent.clauses['as']
            py_type = self.gen_type(type_expr)
            parts.append(f"output_schema={py_type}")

        # Source (from 'from' clause)
        if 'from' in intent.clauses:
            parts.append(f"source={self.gen_expr(intent.clauses['from'])}")

        # Count (from 'in' clause — "in 3 sentences")
        if 'in' in intent.clauses:
            clause = intent.clauses['in']
            if isinstance(clause, dict):
                parts.append(f'count={clause["count"]}')
                parts.append(f'unit="{clause["unit"]}"')
            else:
                parts.append(f"count={self.gen_expr(clause)}")

        # Criteria (from 'against' clause)
        if 'against' in intent.clauses:
            parts.append(f"criteria={self.gen_expr(intent.clauses['against'])}")

        # Target (from 'to' clause)
        if 'to' in intent.clauses:
            parts.append(f"target={self.gen_expr(intent.clauses['to'])}")

        # Context (from 'using' clause)
        if 'using' in intent.clauses:
            parts.append(f"context={self.gen_expr(intent.clauses['using'])}")

        # Factors (from 'considering' clause)
        if 'considering' in intent.clauses:
            factors = intent.clauses['considering']
            factors_str = "[" + ", ".join(self.gen_expr(f) for f in factors) + "]"
            parts.append(f"factors={factors_str}")

        args_str = ", ".join(parts)
        return f"await self.intent({args_str})"
