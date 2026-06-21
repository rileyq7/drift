"""MCP client runtime — backs Drift's `tool name from mcp "url"` declarations.

Drift's codegen emits a thin wrapper class for each MCP tool. The
wrapper holds a connection (lazy, on first call) to the named MCP
server and exposes the server's tools as Python methods via
__getattr__. Each method translates to an `await
session.call_tool(name, arguments=kwargs)` round trip.

URL forms understood:
  mcp://command [arg [arg ...]]     stdio — splits on whitespace, treats
                                   the first token as the command,
                                   remainder as argv. Server speaks
                                   newline-delimited JSON-RPC on stdio.
  mcp+http://host[:port]/...        streamable HTTP — the "Streamable
                                   HTTP" transport from the MCP spec.
  mcp+sse://...  (alias for http)   same as above for compatibility.
  https://... / http://...          treated as streamable HTTP.

A MockMCPClient is also provided. Tests use it via
`drift.runtime.mcp_client.use_mock(...)` so they don't need a real
server. Generated code is shape-identical in both modes.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
from typing import Any, Optional
from urllib.parse import urlparse


# ── Mock client (test path) ────────────────────────────────────────────


class _MockMcpSession:
    """Minimal stand-in for mcp.ClientSession used by tests.

    Records each call_tool invocation and returns whatever the test set
    up via set_response(). No subprocess, no HTTP."""

    def __init__(self, responses: Optional[dict] = None):
        self.responses = responses or {}
        self.calls: list[tuple[str, dict]] = []

    async def initialize(self):
        pass

    async def list_tools(self):
        # Return whatever names the test handed us as response keys —
        # enough for the codegen wrapper to validate or introspect.
        from types import SimpleNamespace
        return SimpleNamespace(tools=[
            SimpleNamespace(name=n, description="(mock)", inputSchema={})
            for n in self.responses
        ])

    async def call_tool(self, name: str, arguments: Optional[dict] = None):
        self.calls.append((name, arguments or {}))
        return self.responses.get(name, {"echo": {"name": name, "args": arguments}})


_MOCK_SESSION: Optional[_MockMcpSession] = None


def use_mock(responses: Optional[dict] = None) -> _MockMcpSession:
    """Wire a mock session into the next MCPClient created in this process.
    Returns the mock so tests can assert on its `.calls` list.

    Tests call this in a fixture; production code never touches it."""
    global _MOCK_SESSION
    _MOCK_SESSION = _MockMcpSession(responses)
    return _MOCK_SESSION


def _take_mock() -> Optional[_MockMcpSession]:
    global _MOCK_SESSION
    s = _MOCK_SESSION
    _MOCK_SESSION = None
    return s


# ── Real client ────────────────────────────────────────────────────────


class MCPClient:
    """Connects to an MCP server and exposes its tools as awaitable
    methods. Lazy connection — the subprocess (or HTTP session) opens
    on the first tool call. Connection persists for the client's
    lifetime; closed via close().

    Concurrency: all access goes through a single ClientSession, which
    serializes requests internally. Multiple Drift agents sharing the
    same MCPClient instance is safe; the SDK handles request
    multiplexing."""

    def __init__(self, url: str, *, name: str = ""):
        self.url = url
        self.name = name or _derive_name(url)
        self._session = None
        self._task_group = None        # AsyncExitStack equivalent
        self._mock = _take_mock()       # set by use_mock() before init
        self._tools_cache: Optional[set[str]] = None

    async def _ensure_connected(self):
        if self._session is not None:
            return

        if self._mock is not None:
            self._session = self._mock
            await self._session.initialize()
            return

        # Real session — open via the appropriate transport.
        from contextlib import AsyncExitStack
        from mcp import ClientSession, StdioServerParameters

        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        scheme = self.url.split("://", 1)[0] if "://" in self.url else ""

        if scheme in ("http", "https", "mcp+http", "mcp+sse"):
            from mcp.client.streamable_http import streamablehttp_client
            http_url = self.url
            if scheme.startswith("mcp+"):
                http_url = "http" + self.url[len(scheme):]
            read, write, _ = await self._exit_stack.enter_async_context(
                streamablehttp_client(http_url),
            )
        else:
            # stdio: strip mcp:// and shell-split the remainder
            from mcp.client.stdio import stdio_client
            cmd_str = self.url[len("mcp://"):] if scheme == "mcp" else self.url
            parts = shlex.split(cmd_str)
            if not parts:
                raise ValueError(
                    f"mcp:// URL {self.url!r} has no command after the scheme"
                )
            params = StdioServerParameters(command=parts[0], args=parts[1:])
            read, write = await self._exit_stack.enter_async_context(
                stdio_client(params),
            )

        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read, write),
        )
        await self._session.initialize()

    async def call(self, method_name: str, **kwargs) -> Any:
        """Invoke an MCP tool by name. Drift codegen routes
        `tool_name.method_name(arg=val)` here."""
        await self._ensure_connected()
        result = await self._session.call_tool(method_name, arguments=kwargs)
        return _coerce_tool_result(result)

    async def list_tools(self) -> list[str]:
        await self._ensure_connected()
        listing = await self._session.list_tools()
        return [t.name for t in listing.tools]

    async def close(self):
        if hasattr(self, "_exit_stack"):
            await self._exit_stack.__aexit__(None, None, None)
            del self._exit_stack
        self._session = None


def _derive_name(url: str) -> str:
    try:
        parsed = urlparse(url)
        return parsed.netloc or parsed.path or url
    except Exception:
        return url


def _coerce_tool_result(result: Any) -> Any:
    """SDK returns a CallToolResult with .content (a list of content blocks).
    Most tools return one text block whose payload is JSON. We unwrap that
    common case so generated code can treat the return value as a dict."""
    if isinstance(result, dict):
        return result
    content = getattr(result, "content", None)
    if not content:
        return None
    if len(content) == 1:
        block = content[0]
        text = getattr(block, "text", None)
        if text is not None:
            try:
                return json.loads(text)
            except (json.JSONDecodeError, ValueError):
                return text
    # Multi-block or non-text — return raw content list
    return content


# ── McpTool wrapper — what codegen emits one of per tool decl ──────────


class McpTool:
    """Generated code creates one of these per `tool name from mcp ...`
    declaration. Method calls become MCP tool invocations:

        await slack.send_message(channel="...", text="...")
                              ↓
        await McpClient.call("send_message", channel="...", text="...")
    """

    def __init__(self, url: str, name: str = ""):
        self._client = MCPClient(url, name=name)

    def __getattr__(self, method_name: str):
        # Return an awaitable bound function so Drift's
        # `slack.send_message(...)` reads naturally in generated code.
        async def _invoke(**kwargs):
            return await self._client.call(method_name, **kwargs)
        return _invoke

    async def list_tools(self) -> list[str]:
        return await self._client.list_tools()

    async def close(self):
        await self._client.close()
