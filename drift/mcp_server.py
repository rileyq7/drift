"""
Drift MCP server — exposes Drift's transpile/check/run via the Model Context Protocol.

Other coding agents (Claude Code, Cursor, anything MCP-aware) can register this
server and gain three tools:

  - drift_check(source)   → "OK" or parse-error message
  - drift_transpile(source) → generated Python
  - drift_run(source, input=None) → cost-tracked run, returns result + cost

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


async def _run(source: str, input_json: str | None = None) -> dict:
    """Transpile + exec a Drift program against the runtime. Returns
    {ok, result?, cost?, outputs?, error?, stage?, kind?} — `cost` is a
    {total_cost, budget, currency, calls} snapshot and `outputs` is the
    list of `respond`-statement lines the agent printed, both present on
    success and on a "run"-stage failure (a run can spend real money and
    produce partial output before failing).

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
            except Exception as e:
                return {"ok": False, "stage": "import", "error": f"{type(e).__name__}: {e}"}

            from drift.runtime.core import Agent, run_agent
            agents = {n: getattr(module, n) for n in dir(module)
                      if isinstance(getattr(module, n), type)
                      and issubclass(getattr(module, n), Agent)
                      and getattr(module, n) is not Agent}
            if not agents:
                return {"ok": False, "stage": "discover", "error": "No agents in program"}

            agent_cls = next(iter(agents.values()))
            inputs = json.loads(input_json) if input_json else {}

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
                    "`input` is a JSON object mapping step parameter names to values, passed "
                    "as a JSON-encoded string. Returns {ok, result, cost, outputs} on success "
                    "— cost is {total_cost, budget, currency, calls}, outputs is the list of "
                    "the agent's `respond`-statement lines — or {ok: false, stage, kind, error, "
                    "cost, outputs} on failure, where kind is one of budget/auth/business-logic/"
                    "bug and cost/outputs reflect spend and output produced before the failure."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "input": {"type": "string", "description": "JSON object as a string"},
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
            result = await _run(arguments["source"], arguments.get("input"))
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
