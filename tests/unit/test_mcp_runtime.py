"""Tests for the MCP runtime — the MCPClient / McpTool pair that backs
generated code for `tool name from mcp "..."` declarations.

Uses use_mock() to install a mock ClientSession so tests don't need a
real MCP server."""
import pytest

from drift.runtime.mcp_client import McpTool, use_mock, MCPClient


class TestMcpToolViaMock:
    @pytest.mark.asyncio
    async def test_method_call_routes_to_call_tool(self):
        mock = use_mock(responses={
            "send_message": {"ok": True, "ts": "123"},
        })
        slack = McpTool("mcp://fake", name="slack")
        result = await slack.send_message(channel="general", text="hi")
        assert result == {"ok": True, "ts": "123"}
        # Mock recorded the call with the kwargs Drift passed through.
        assert mock.calls == [("send_message", {"channel": "general", "text": "hi"})]

    @pytest.mark.asyncio
    async def test_list_tools_reflects_mock_responses(self):
        use_mock(responses={"a": {}, "b": {}, "c": {}})
        client = MCPClient("mcp://x")
        tools = await client.list_tools()
        assert set(tools) == {"a", "b", "c"}

    @pytest.mark.asyncio
    async def test_unknown_method_returns_echo(self):
        """When the mock has no canned response for a method, it echoes
        the call back. This makes tests robust to partial wiring — the
        call shape still flows through the wrapper without exploding."""
        use_mock(responses={})
        t = McpTool("mcp://x")
        result = await t.never_seen(arg=1)
        assert result == {"echo": {"name": "never_seen", "args": {"arg": 1}}}


class TestMcpToolGeneratedShape:
    """Confirm that generated code for an MCP declaration produces a
    runtime object whose method-call shape matches the spec example:

        slack.send_message(channel: "...", text: "...")
    """

    @pytest.mark.asyncio
    async def test_codegen_then_invoke(self, transpile, tmp_path,
                                       monkeypatch):
        py = transpile(
            'tool slack from mcp "mcp://slack-mock"'
        )
        # The emitted code creates `slack = _McpTool(...)` at module load.
        mod_path = tmp_path / "tool_under_test.py"
        mod_path.write_text(py)
        monkeypatch.syspath_prepend(str(tmp_path))

        use_mock(responses={"send_message": {"delivered": True}})

        import importlib, sys
        if "tool_under_test" in sys.modules:
            del sys.modules["tool_under_test"]
        mod = importlib.import_module("tool_under_test")

        result = await mod.slack.send_message(channel="ops", text="hello")
        assert result == {"delivered": True}
