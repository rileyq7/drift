"""
Drift MCP server — exposes Drift's transpile/check/run via the Model Context Protocol.

Other coding agents (Claude Code, Cursor, anything MCP-aware) can register this
server and gain three tools:

  - drift_check(source)   → "OK" or parse-error message
  - drift_transpile(source) → generated Python
  - drift_run(source, input=None, pipeline=None) → cost-tracked run, returns result + cost

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
from drift.runtime.core import BudgetExceeded, StepFailed, AuthError


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


def _split_cost_and_outputs(snapshot: dict | None) -> tuple[dict | None, list]:
    """run_agent's cost_out bundles cost numbers and `respond`-statement
    output into one dict (see run_agent's docstring) — split them into the
    two separate top-level response fields drift_run actually returns."""
    if not snapshot:
        return snapshot, []
    outputs = snapshot.pop('outputs', [])
    return snapshot, outputs


async def _run(source: str, input_json: str | None = None,
                pipeline: str | None = None) -> dict:
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
                result = await run_agent(agent_cls, inputs=inputs, cost_out=cost)
            except Exception as e:
                sys.stdout = old_out
                # BudgetExceeded/StepFailed/AuthError are the agent's own
                # business-logic outcomes (bad LLM output, exhausted retries,
                # bad credentials) — distinct from an infra/codegen bug, so a
                # calling agent can tell "your program has a bug" apart from
                # "the run failed for a reason your program already handles".
                # Cost is attached by run_agent (see `_drift_cost`) even on
                # failure, since a run can spend real money before failing.
                if isinstance(e, BudgetExceeded):
                    kind = "budget"
                elif isinstance(e, AuthError):
                    kind = "auth"
                elif isinstance(e, StepFailed):
                    kind = "business-logic"
                else:
                    kind = "bug"
                cost_snapshot, outputs = _split_cost_and_outputs(
                    getattr(e, "_drift_cost", cost or None)
                )
                return {
                    "ok": False,
                    "stage": "run",
                    "kind": kind,
                    "error": f"{type(e).__name__}: {e}",
                    "cost": cost_snapshot,
                    "outputs": outputs,
                }
            finally:
                sys.stdout = old_out

            cost_snapshot, outputs = _split_cost_and_outputs(cost)
            return {
                "ok": True,
                "result": _to_jsonable(result),
                "cost": cost_snapshot,
                "outputs": outputs,
            }
        finally:
            # Restore/clear the module slot we hijacked so concurrent or
            # subsequent runs don't see a stale program module.
            if prev_module is not None:
                sys.modules[module_name] = prev_module
            else:
                sys.modules.pop(module_name, None)


def _to_jsonable(obj: Any) -> Any:
    import dataclasses
    if dataclasses.is_dataclass(obj):
        return {f.name: _to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


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
                name="drift_run",
                description=(
                    "Transpile and execute a Drift program (this spends real LLM budget — "
                    "run drift_check first to catch syntax/codegen errors for free). Optional "
                    "`input` is a JSON-encoded string — an object mapping step parameter names "
                    "to values for an agent run, or the pipeline's initial input (any JSON "
                    "value) when `pipeline` is given. Optional `pipeline` names a `pipeline` "
                    "declaration to run instead of an agent (required if the source declares "
                    "only pipelines, no agents — mirrors `drift run --pipeline <name>`). "
                    "Returns {ok, result, cost, outputs} on success — cost is {total_cost, "
                    "budget, currency, calls}, outputs is the list of the agent's `respond`-"
                    "statement lines (agent runs only; a pipeline's own cost/outputs aren't "
                    "separately tracked) — or {ok: false, stage, kind, error, cost, outputs} on "
                    "failure, where kind is one of budget/auth/business-logic/bug and "
                    "cost/outputs reflect spend and output produced before the failure. Note: "
                    "cross-file `import` doesn't resolve here (raw source text, no file path)."
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
        elif name == "drift_run":
            result = await _run(
                arguments["source"], arguments.get("input"), arguments.get("pipeline")
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
