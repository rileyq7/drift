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
from drift.codegen import CodeGenerator


def _transpile(source: str) -> str:
    tokens = lex(source)
    program = Parser(tokens).parse()
    return CodeGenerator().generate(program).replace(
        "Source: <drift_file>", "Source: <mcp_call>"
    )


def _check(source: str) -> dict:
    try:
        tokens = lex(source)
        Parser(tokens).parse()
    except LexError as e:
        return {"ok": False, "kind": "lex", "message": str(e), "line": e.line, "col": e.col}
    except ParseError as e:
        tok = e.token
        return {
            "ok": False, "kind": "parse",
            "message": str(e),
            "line": tok.line, "col": tok.col,
        }
    return {"ok": True}


async def _run(source: str, input_json: str | None = None) -> dict:
    """Transpile + exec a Drift program against the runtime. Returns
    {ok, result?, cost?, calls?, error?, traceback?}.

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
                return {"ok": False, "stage": "import", "error": str(e)}

            from drift.runtime.core import Agent, run_agent
            agents = {n: getattr(module, n) for n in dir(module)
                      if isinstance(getattr(module, n), type)
                      and issubclass(getattr(module, n), Agent)
                      and getattr(module, n) is not Agent}
            if not agents:
                return {"ok": False, "stage": "discover", "error": "No agents in program"}

            agent_cls = next(iter(agents.values()))
            inputs = json.loads(input_json) if input_json else {}

            # Capture stdout so the run banner doesn't pollute the MCP channel.
            old_out = sys.stdout
            buf = io.StringIO()
            sys.stdout = buf
            try:
                result = await run_agent(agent_cls, inputs=inputs)
            except Exception as e:
                return {"ok": False, "stage": "run", "error": f"{type(e).__name__}: {e}"}
            finally:
                sys.stdout = old_out

            banner = buf.getvalue()
            return {
                "ok": True,
                "result": _to_jsonable(result),
                "banner": banner,
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
                description="Validate Drift source code. Returns {ok, kind, message, line, col} on failure.",
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
                    "Transpile and execute a Drift program. Optional `input` is a JSON object "
                    "mapping step parameter names to values. Returns the step's result plus the "
                    "captured run banner (cost report etc.)."
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
            try:
                py = _transpile(arguments["source"])
                result = {"ok": True, "python": py}
            except (LexError, ParseError) as e:
                result = {"ok": False, "error": str(e)}
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
