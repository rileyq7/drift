"""
Drift MCP server — exposes Drift's transpile/check/run via the Model Context Protocol.

Other coding agents (Claude Code, Cursor, anything MCP-aware) can register this
server and gain four tools:

  - drift_check(source)   → "OK" or parse-error message
  - drift_transpile(source) → generated Python
  - drift_schema(source, name=None) → JSON Schema for the program's schema block(s)
  - drift_run(source, input=None, pipeline=None, agent=None, step=None) → cost-tracked run, returns result + cost

Launch with `drift mcp` (stdio). Wire it into Claude Code via .mcp.json:

    {
      "mcpServers": {
        "drift": { "command": "drift", "args": ["mcp"] }
      }
    }
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from drift.lexer import lex, LexError
from drift.parser import Parser, ParseError
from drift.codegen import CodeGenerator, CodegenError
from drift.runtime.core import build_run_outcome, _to_jsonable


def _transpile(source: str) -> str:
    tokens = lex(source)
    program = Parser(tokens).parse()
    return CodeGenerator().generate(program).replace(
        "Source: <drift_file>", "Source: <mcp_call>"
    )


def _transpile_result(source: str) -> dict:
    """The drift_transpile tool's logic, standalone so it's unit-testable
    without spinning up the stdio server. Must catch CodegenError alongside
    LexError/ParseError — a construct that parses but can't compile (e.g.
    `~>`/`parallel step`/`schedule:`) would otherwise propagate uncaught out
    of call_tool instead of the same clean {ok: false, ...} shape
    drift_check/drift_run give it."""
    try:
        py = _transpile(source)
        return {"ok": True, "python": py}
    except (LexError, ParseError, CodegenError) as e:
        return {"ok": False, "error": str(e)}


def _check(source: str) -> dict:
    """Validate syntax AND that it lowers to Python (codegen runs, output
    discarded), so constructs that parse but can't be compiled — e.g. `~>`/
    `|>` pipeline edges, `parallel step` — are caught here too."""
    try:
        tokens = lex(source)
        program = Parser(tokens).parse()
        CodeGenerator().generate(program)
    except LexError as e:
        # `error`, not `message` — matches drift_run/drift_transpile, so a
        # caller can use one `if not ok: report(result["error"])` handler
        # across all three tools instead of needing per-tool field lookups.
        return {"ok": False, "kind": "lex", "error": str(e), "line": e.line, "col": e.col}
    except ParseError as e:
        tok = e.token
        return {
            "ok": False, "kind": "parse",
            "error": str(e),
            "line": tok.line, "col": tok.col,
        }
    except CodegenError as e:
        return {"ok": False, "kind": "codegen", "error": str(e)}
    return {"ok": True}


def _schema_result(source: str, name: str | None = None) -> dict:
    """The drift_schema tool's logic (and `drift schema`'s underlying
    implementation) — standalone so it's unit-testable without spinning up
    the stdio server, same pattern as _check/_transpile_result.

    Transpiles `source`, execs the generated module, and renders each
    top-level `schema` block's generated dataclass to JSON Schema via
    `dataclass_to_json_schema` (drift/runtime/core.py) — the same
    converter the runtime already uses internally to build provider
    strict-mode schemas, just exposed here directly instead of requiring
    a caller to transpile + exec + introspect a module by hand. There was
    previously no way to get a program's schema shape without running an
    agent (which spends budget) or hand-parsing the generated Python.

    Without `name`, returns every schema declared in `source` (in source
    order) as {ok: true, schemas: {name: json_schema, ...}}. With `name`,
    returns just that one as {ok: true, schema: json_schema}. Schemas
    aren't "discovered" from a module the way agents/pipelines are (no
    existing first-declared-wins convention makes sense here — a caller
    asking for a program's schemas almost always wants all of them, e.g.
    to build several tool input/output schemas from one shared file), so
    the default is "all", not "first".
    """
    try:
        python_source = _transpile(source)
    except LexError as e:
        return {"ok": False, "stage": "lex", "error": str(e)}
    except ParseError as e:
        return {"ok": False, "stage": "parse", "error": str(e)}
    except CodegenError as e:
        return {"ok": False, "stage": "codegen", "error": str(e)}

    tokens = lex(source)
    program = Parser(tokens).parse()
    codegen = CodeGenerator()
    codegen.generate(program)
    schema_names = list(codegen.schemas_declared)

    if not schema_names:
        return {"ok": False, "stage": "discover", "error": "No schemas in program"}

    if name and name not in schema_names:
        avail = ', '.join(schema_names)
        return {
            "ok": False, "stage": "discover",
            "error": f"Schema {name!r} not found. Available: {avail}",
        }

    with tempfile.TemporaryDirectory() as td:
        py_path = Path(td) / "drift_schema_program.py"
        py_path.write_text(python_source)

        import importlib.util
        module_name = "drift_mcp_schema_program"
        prev_module = sys.modules.get(module_name)
        spec = importlib.util.spec_from_file_location(module_name, py_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            try:
                spec.loader.exec_module(module)
            except Exception as e:
                return {"ok": False, "stage": "import", "error": f"{type(e).__name__}: {e}"}

            from drift.runtime.core import dataclass_to_json_schema
            rendered = {}
            for schema_name in ([name] if name else schema_names):
                cls = getattr(module, schema_name, None)
                json_schema = dataclass_to_json_schema(cls) if cls is not None else None
                if json_schema is None:
                    return {
                        "ok": False, "stage": "import",
                        "error": f"Schema {schema_name!r} did not produce a dataclass",
                    }
                rendered[schema_name] = json_schema
        finally:
            if prev_module is not None:
                sys.modules[module_name] = prev_module
            else:
                sys.modules.pop(module_name, None)

    if name:
        return {"ok": True, "schema": rendered[name]}
    return {"ok": True, "schemas": rendered}


async def _run(source: str, input_json: str | None = None,
                pipeline: str | None = None, agent: str | None = None,
                step: str | None = None) -> dict:
    """Transpile + exec a Drift program against the runtime. Returns
    {ok, result?, cost?, outputs?, error?, stage?, kind?} — `cost` is a
    {total_cost, budget, currency, calls} snapshot and `outputs` is the
    list of `respond`-statement lines the agent printed, both present on
    success and on a "run"-stage failure (a run can spend real money and
    produce partial output before failing).

    `pipeline`, if given, names a `pipeline` declaration to run instead of
    an agent — mirrors `drift run --pipeline <name>`. Without it, a
    program containing ONLY pipelines (no agents) auto-runs its
    first-declared pipeline; a program with agents always prefers agent
    execution unless `pipeline` is explicitly given, matching this
    module's own prior (agent-only) behavior for anyone not using
    pipelines. Previously this function never looked for pipelines at
    all — it silently ran an agent's step directly even when the source
    declared a pipeline, or crashed with a misleading `kind: "bug"` if
    pipeline-shaped (list) input was passed, both undocumented gaps
    against LLM.md's own claim that drift_run's only limitation is
    cross-file imports.

    `agent`/`step`, if given, select a specific agent/step by name —
    mirrors `drift run --agent`/`--step`. Without `agent`, a program with
    more than one agent used to always silently pick the first-declared
    one with no way for an MCP caller to target another, unlike the CLI
    (which already had --agent/--step). Ignored when `pipeline` is given.

    Async because it awaits the agent directly: the MCP server calls this from
    an already-running event loop, so `asyncio.run()` here would raise
    "cannot be called from a running event loop".
    """
    try:
        python_source = _transpile(source)
    except LexError as e:
        return {"ok": False, "stage": "lex", "error": str(e)}
    except ParseError as e:
        return {"ok": False, "stage": "parse", "error": str(e)}

    try:
        parsed_input = json.loads(input_json) if input_json else None
    except json.JSONDecodeError as e:
        # Previously unguarded — a malformed `input` string raised
        # straight out of this function instead of returning the
        # documented {ok: false, ...} envelope every other failure mode
        # here uses, breaking the "uniform shape across all three tools"
        # contract described in the drift_run tool description.
        return {"ok": False, "stage": "input", "error": f"invalid input JSON: {e}"}

    with tempfile.TemporaryDirectory() as td:
        py_path = Path(td) / "drift_program.py"
        py_path.write_text(python_source)

        import importlib.util
        module_name = "drift_mcp_program"
        prev_module = sys.modules.get(module_name)
        spec = importlib.util.spec_from_file_location(module_name, py_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            try:
                spec.loader.exec_module(module)
            except ModuleNotFoundError as e:
                # Two distinct causes collapse into the same Python
                # exception: (a) a cross-file `import { X } from
                # "./other.drift"` — drift_run can't resolve it, since it
                # only ever receives raw source text, not a file path, so
                # there's no directory for a relative import to resolve
                # against (unlike `drift run <file>` via the CLI, which
                # auto-transpiles the dependency and fixes up the import
                # path); or (b) a `tool ... from python "module:fn"`
                # referencing a real missing/unimportable Python module,
                # unrelated to Drift's own `import`. Only show the
                # cross-file-import hint when the source actually
                # contains a `.drift`-suffixed import — a cheap text
                # check on the original source, not a full re-parse.
                if re.search(r'import\s.*from\s+"[^"]*\.drift"', source):
                    hint = (
                        " — likely cause: drift_run can't resolve cross-file "
                        "`import` (it receives raw source text, not a file "
                        "path, so relative imports have nothing to resolve "
                        "against). Inline the dependency into one source "
                        "string, or write the files to disk and use `drift "
                        "run <file>` instead."
                    )
                else:
                    hint = ""
                return {
                    "ok": False, "stage": "import",
                    "error": f"{type(e).__name__}: {e}{hint}",
                }
            except Exception as e:
                return {"ok": False, "stage": "import", "error": f"{type(e).__name__}: {e}"}

            from drift.runtime.core import Agent, run_agent, first_declared
            agents = {n: getattr(module, n) for n in dir(module)
                      if isinstance(getattr(module, n), type)
                      and issubclass(getattr(module, n), Agent)
                      and getattr(module, n) is not Agent}
            # Pipelines are plain classes (not Agent subclasses) with an
            # async `run` method — same shape-based detection cli.py's
            # _run_once already uses. Previously this function never
            # looked for pipelines at all: a program declaring ONLY a
            # `pipeline { ... }` (no agent) silently ran an agent step
            # anyway (whichever agent a nested `use`d/referenced class
            # happened to expose), or — if pipeline-shaped (list) input
            # was passed — crashed with a misleading `kind: "bug"`
            # (AttributeError: 'list' object has no attribute 'items'
            # from _coerce_inputs' dict-oriented code path), implying a
            # defect in the user's program rather than an unsupported
            # tool-side gap.
            pipelines = {}
            for n in dir(module):
                if n.startswith('_'):
                    continue
                obj = getattr(module, n)
                if not isinstance(obj, type) or obj is Agent or n in agents:
                    continue
                run_attr = getattr(obj, 'run', None)
                if run_attr is not None and asyncio.iscoroutinefunction(run_attr):
                    pipelines[n] = obj

            if pipeline:
                pipe_cls = pipelines.get(pipeline)
                if not pipe_cls:
                    avail = ', '.join(pipelines) if pipelines else '(none)'
                    return {
                        "ok": False, "stage": "discover",
                        "error": f"Pipeline {pipeline!r} not found. Available: {avail}",
                    }
                old_out = sys.stdout
                buf = io.StringIO()
                sys.stdout = buf
                try:
                    result = await pipe_cls().run(initial_input=parsed_input)
                except Exception as e:
                    sys.stdout = old_out
                    return {
                        "ok": False, "stage": "run", "kind": "bug",
                        "error": f"{type(e).__name__}: {e}",
                    }
                finally:
                    sys.stdout = old_out
                return {"ok": True, "result": _to_jsonable(result)}

            if not agents:
                if pipelines:
                    avail = ', '.join(pipelines)
                    return {
                        "ok": False, "stage": "discover",
                        "error": (
                            f"No agents found, but pipelines are: {avail}. "
                            "Pass pipeline=<name> to run one."
                        ),
                    }
                return {"ok": False, "stage": "discover", "error": "No agents in program"}

            if agent:
                agent_cls = agents.get(agent)
                if not agent_cls:
                    avail = ', '.join(agents) if agents else '(none)'
                    return {
                        "ok": False, "stage": "discover",
                        "error": f"Agent {agent!r} not found. Available: {avail}",
                    }
            else:
                # dir(module) returns names ALPHABETICALLY — next(iter(...))
                # used to silently run whichever agent's class name sorted
                # first, not the first one actually declared in the source,
                # contradicting LLM.md's documented "runs the first agent's
                # first step". See first_declared's docstring.
                agent_cls = first_declared(agents.values())
            inputs = parsed_input if parsed_input is not None else {}

            # Capture stdout so the run banner (box-drawing header, printed
            # cost report, etc.) doesn't pollute the MCP channel. Its
            # content is otherwise fully redundant with the structured
            # `cost`/`result`/`outputs` fields below, so it's discarded
            # rather than returned — a calling agent shouldn't have to pay
            # context tokens re-parsing box-drawing on every call.
            old_out = sys.stdout
            buf = io.StringIO()
            sys.stdout = buf
            cost: dict = {}
            try:
                result = await run_agent(agent_cls, step_name=step, inputs=inputs, cost_out=cost)
            except Exception as e:
                sys.stdout = old_out
                # BudgetExceeded/StepFailed/AuthError are the agent's own
                # business-logic outcomes (bad LLM output, exhausted retries,
                # bad credentials) — distinct from an infra/codegen bug, so a
                # calling agent can tell "your program has a bug" apart from
                # "the run failed for a reason your program already handles".
                # Cost is attached by run_agent (see `_drift_cost`) even on
                # failure, since a run can spend real money before failing.
                # build_run_outcome derives `kind` from the exception type
                # and folds in `agent`/`step` from the _drift_agent/
                # _drift_step tags step_decorator already attaches.
                return build_run_outcome(
                    error=e, cost=getattr(e, "_drift_cost", cost or None)
                )
            finally:
                sys.stdout = old_out

            return build_run_outcome(result=result, cost=cost)
        finally:
            # Restore/clear the module slot we hijacked so concurrent or
            # subsequent runs don't see a stale program module.
            if prev_module is not None:
                sys.modules[module_name] = prev_module
            else:
                sys.modules.pop(module_name, None)


# ─── MCP server entry point ─────────────────────────────────────────────

def serve_stdio():
    """Run as an MCP stdio server. Blocks until stdin closes."""
    try:
        from mcp.server import Server, NotificationOptions
        from mcp.server.stdio import stdio_server
        import mcp.types as types
    except ImportError:
        sys.stderr.write(
            "  ✗ drift mcp requires the `mcp` package.\n"
            "    Install with: pip install 'drift-lang[mcp]'\n"
        )
        sys.exit(1)

    server = Server("drift")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="drift_check",
                description=(
                    "Validate Drift source code — free (no LLM calls), catches syntax and "
                    "codegen errors before spending run budget. Prefer this over drift_run "
                    "as a first pass when iterating on a program. Returns {ok: true} on "
                    "success, or {ok: false, kind, error, line, col} on failure, where kind "
                    "is one of lex/parse/codegen."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {"source": {"type": "string", "description": "Drift program source"}},
                    "required": ["source"],
                },
            ),
            types.Tool(
                name="drift_transpile",
                description="Transpile Drift source to async Python. Returns the generated Python as a string.",
                inputSchema={
                    "type": "object",
                    "properties": {"source": {"type": "string"}},
                    "required": ["source"],
                },
            ),
            types.Tool(
                name="drift_schema",
                description=(
                    "Render a Drift program's `schema` block(s) as JSON Schema — free (no LLM "
                    "calls; the module is imported but no agent runs). Useful for deriving an "
                    "external tool's input/output schema from a Drift program without running "
                    "it. Without `name`, returns every schema declared in `source`: {ok: true, "
                    "schemas: {name: json_schema, ...}}. With `name`, returns just that one: "
                    "{ok: true, schema: json_schema}. Returns {ok: false, stage, error} on "
                    "failure, where stage is one of lex/parse/codegen/discover/import."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "name": {
                            "type": "string",
                            "description": "Name of a specific schema to render (default: all)",
                        },
                    },
                    "required": ["source"],
                },
            ),
            types.Tool(
                name="drift_run",
                description=(
                    "Transpile and execute a Drift program (this spends real LLM budget — "
                    "run drift_check first to catch syntax/codegen errors for free). Optional "
                    "`input` is a JSON-encoded string — an object mapping step parameter names "
                    "to values for an agent run, or the pipeline's initial input (any JSON "
                    "value) when `pipeline` is given. Optional `pipeline` names a `pipeline` "
                    "declaration to run instead of an agent (required if the source declares "
                    "only pipelines, no agents — mirrors `drift run --pipeline <name>`). "
                    "Optional `agent`/`step` select a specific agent/step by name when a "
                    "program declares more than one (mirrors `drift run --agent`/`--step`); "
                    "without `agent`, the first-declared agent is used. Ignored when `pipeline` "
                    "is given. Returns {ok, result, cost, outputs} on success — cost is "
                    "{total_cost, budget, currency, calls}, outputs is the list of the agent's "
                    "`respond`-statement lines (agent runs only; a pipeline's own cost/outputs "
                    "aren't separately tracked) — or {ok: false, stage, kind, error, agent, "
                    "step, cost, outputs} on failure, where kind is one of "
                    "budget/auth/business-logic/bug and cost/outputs reflect spend and output "
                    "produced before the failure. Note: cross-file `import` doesn't resolve "
                    "here (raw source text, no file path)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "input": {"type": "string", "description": "JSON value as a string"},
                        "pipeline": {
                            "type": "string",
                            "description": "Name of a `pipeline` declaration to run instead of an agent",
                        },
                        "agent": {
                            "type": "string",
                            "description": "Name of a specific agent to run (default: first-declared)",
                        },
                        "step": {
                            "type": "string",
                            "description": "Name of a specific step to run on the selected agent",
                        },
                    },
                    "required": ["source"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        if name == "drift_check":
            result = _check(arguments["source"])
        elif name == "drift_transpile":
            result = _transpile_result(arguments["source"])
        elif name == "drift_schema":
            result = _schema_result(arguments["source"], arguments.get("name"))
        elif name == "drift_run":
            result = await _run(
                arguments["source"], arguments.get("input"), arguments.get("pipeline"),
                arguments.get("agent"), arguments.get("step"),
            )
        else:
            result = {"ok": False, "error": f"unknown tool {name!r}"}

        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(
                    notification_options=NotificationOptions(),
                ),
            )

    asyncio.run(main())


if __name__ == "__main__":
    serve_stdio()
