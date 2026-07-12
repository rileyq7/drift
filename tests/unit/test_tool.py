"""Tests for §2.4 tool declarations — three forms: mcp, python, rest."""
import pytest

from drift import ast_nodes as ast


class TestPythonForm:
    def test_module_colon_fn(self, parse_ast):
        p = parse_ast('tool calc from python "utils.finance:calculate_runway"')
        t = p.declarations[0]
        assert t.kind == "python"
        assert t.source == "utils.finance:calculate_runway"

    def test_codegen_imports_function(self, transpile):
        out = transpile('tool calc from python "utils.finance:calculate_runway"')
        assert "from utils.finance import calculate_runway as _drift_tool_calc" in out
        assert "calc = _drift_tool_calc" in out


class TestMcpForm:
    def test_parse(self, parse_ast):
        p = parse_ast('tool grants from mcp "https://grants.api/mcp"')
        t = p.declarations[0]
        assert t.kind == "mcp"
        assert t.source == "https://grants.api/mcp"

    def test_codegen_emits_mcp_tool(self, transpile):
        """MCP runtime now real — codegen emits McpTool wrapping the URL.
        The legacy `_<name>_McpTool` symbol is kept as an alias so older
        downstream tooling that grepped for it still works."""
        out = transpile('tool grants from mcp "https://grants.api/mcp"')
        assert "from drift.runtime.mcp_client import McpTool" in out
        assert "grants = _McpTool('https://grants.api/mcp'" in out
        assert "_grants_McpTool" in out  # legacy alias preserved


class TestRestForm:
    REST = (
        'tool companies_house { '
        '  endpoint: "https://api.example.com" '
        '  auth: env("CH_KEY") '
        '  action lookup(company_number: string) -> string { '
        '    GET "/company/{company_number}" '
        '  } '
        '  action search(q: string) -> string { '
        '    GET "/search?q={q}" '
        '  } '
        '}'
    )

    def test_parses_rest_kind(self, parse_ast):
        t = parse_ast(self.REST).declarations[0]
        assert t.kind == "rest"
        assert t.endpoint == "https://api.example.com"
        assert t.auth_env == "CH_KEY"

    def test_parses_actions(self, parse_ast):
        t = parse_ast(self.REST).declarations[0]
        assert len(t.actions) == 2
        assert t.actions[0].name == "lookup"
        assert t.actions[0].method == "GET"
        assert t.actions[0].path == "/company/{company_number}"
        assert t.actions[1].name == "search"

    def test_codegen_emits_async_methods(self, transpile):
        out = transpile(self.REST)
        assert "async def lookup(self, company_number: str)" in out
        assert "async def search(self, q: str)" in out
        # Path interpolation as f-string
        assert "f'/company/{company_number}'" in out
        # Auth header builder
        assert "_auth_header" in out
        # Tool instance at module level
        assert "companies_house = _companies_house_RestTool()" in out

    def test_bare_string_auth_works(self, parse_ast):
        # `auth: "static-token"` should parse too (dev-only mode) — and, unlike
        # `auth: env(...)`, must be stored as the literal value, not an env
        # var *name* to look up. These used to share one field (auth_env),
        # so a literal token silently became `os.environ.get("<the token>")`
        # → None → every request went out with no Authorization header.
        t = parse_ast(
            'tool t { endpoint: "x" auth: "abc" '
            'action ping() -> string { GET "/" } }'
        ).declarations[0]
        assert t.auth_literal == "abc"
        assert t.auth_env == ""

    def test_bare_string_auth_sends_literal_not_env_lookup(self, transpile):
        out = transpile(
            'tool t { endpoint: "x" auth: "abc" '
            'action ping() -> string { GET "/" } }'
        )
        assert "auth_literal = 'abc'" in out
        assert 'return {"Authorization": f"Bearer {self.auth_literal}"}' in out

    def test_env_auth_still_looks_up_env_var(self, transpile):
        out = transpile(self.REST)
        assert "auth_env = 'CH_KEY'" in out
        assert "auth_literal = None" in out

    def test_post_param_not_in_path_goes_in_json_body(self, transpile):
        # LLM.md's own documented example: `title` isn't in the path
        # template, so it used to be silently dropped — POST went out with
        # no body at all. It must now be sent as a JSON body field.
        out = transpile(
            'tool gh { endpoint: "https://api.github.com" '
            'action create_issue(repo: string, title: string) -> string { '
            '  POST "/repos/{repo}/issues" '
            '} }'
        )
        assert "_body = {'title': title}" in out
        assert "json=_body" in out
        # `repo` is consumed by the path template, not duplicated into the body.
        assert "'repo': repo" not in out

    def test_get_param_not_in_path_goes_in_query_string(self, transpile):
        out = transpile(
            'tool gh { endpoint: "https://api.github.com" '
            'action search(q: string, per_page: int) -> string { '
            '  GET "/search" '
            '} }'
        )
        assert "_query = {'q': q, 'per_page': per_page}" in out
        assert "params=_query" in out

    def test_all_params_in_path_emits_no_extra_body_or_query(self, transpile):
        out = transpile(self.REST)  # lookup(company_number) -> GET "/company/{company_number}"
        assert "_body" not in out
        assert "_query" not in out

    def test_all_http_methods_accepted(self, parse_ast):
        for method in ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"]:
            src = (
                f'tool t {{ endpoint: "x" '
                f'action a(x: string) -> string {{ {method} "/x" }} }}'
            )
            t = parse_ast(src).declarations[0]
            assert t.actions[0].method == method


class TestToolErrors:
    def test_unknown_from_kind_fails(self, parse_ast):
        from drift.parser import ParseError
        with pytest.raises(ParseError):
            parse_ast('tool x from invalidkind "y"')

    def test_unknown_method_fails(self, parse_ast):
        from drift.parser import ParseError
        with pytest.raises(ParseError):
            parse_ast(
                'tool t { endpoint: "x" '
                'action a() -> string { INVALID "/" } }'
            )


class TestToolCallsAreAwaited:
    """Regression: gen_fn_call's fallback branch (any `target.method(...)`
    that isn't a cross-agent step call) never emitted `await`, even though
    every mcp/rest tool's generated method is `async def`. Calling an
    async function without awaiting it doesn't error or run the function
    body at all — Python just hands back an unawaited coroutine object —
    so every documented tool-call example in LLM.md (`weather.get_forecast
    (...)`, `helpdesk.post_outcome(...)`) silently never made the real
    call. No existing test caught this because none exercised a tool call
    from inside a step body through the real codegen path — they only
    checked generated text for unrelated substrings, or called the
    module-level tool object directly from Python instead of via a step.
    """

    MCP_SRC = (
        'tool weather from mcp "https://x.com" '
        'agent A { model: "claude-haiku" '
        '  step f(city: string) -> string { '
        '    let forecast = weather.get_forecast(city: city) '
        '    return forecast '
        '  } '
        '}'
    )

    REST_SRC = (
        'tool helpdesk { '
        '  endpoint: "https://x.com" '
        '  auth: env("TOKEN") '
        '  action post_outcome(id: string) -> dict { POST "/x/{id}" } '
        '} '
        'agent A { model: "claude-haiku" '
        '  step f(id: string) -> dict { '
        '    let result = helpdesk.post_outcome(id: id) '
        '    return result '
        '  } '
        '}'
    )

    BARE_STMT_SRC = (
        'tool helpdesk { '
        '  endpoint: "https://x.com" '
        '  auth: env("TOKEN") '
        '  action post_outcome(id: string) -> dict { POST "/x/{id}" } '
        '} '
        'agent A { model: "claude-haiku" '
        '  step f(id: string) -> string { '
        '    helpdesk.post_outcome(id: id) '
        '    return "done" '
        '  } '
        '}'
    )

    def test_mcp_call_assigned_to_let_is_awaited(self, transpile):
        out = transpile(self.MCP_SRC)
        assert "forecast = await weather.get_forecast(city=city)" in out

    def test_rest_action_call_assigned_to_let_is_awaited(self, transpile):
        out = transpile(self.REST_SRC)
        assert "result = await helpdesk.post_outcome(id=id)" in out

    def test_bare_statement_tool_call_is_awaited(self, transpile):
        out = transpile(self.BARE_STMT_SRC)
        assert "await helpdesk.post_outcome(id=id)" in out

    @pytest.mark.asyncio
    async def test_mcp_call_actually_reaches_the_mock_session(self, transpile, tmp_path, monkeypatch):
        # End-to-end: run a real step body through the real runtime and
        # confirm the mock MCP session actually recorded the call — this
        # is the test class of coverage that was missing entirely; every
        # prior MCP test either checked codegen text or called the tool
        # object directly from Python, never through a step body.
        from drift.runtime.mcp_client import use_mock
        from drift.runtime import run_agent

        py = transpile(self.MCP_SRC)
        mod_path = tmp_path / "tool_call_under_test.py"
        mod_path.write_text(py)
        monkeypatch.syspath_prepend(str(tmp_path))

        mock = use_mock(responses={"get_forecast": {"forecast": "sunny"}})

        import importlib, sys
        if "tool_call_under_test" in sys.modules:
            del sys.modules["tool_call_under_test"]
        mod = importlib.import_module("tool_call_under_test")

        result = await run_agent(mod.A, inputs={"city": "Boston"})
        assert result == {"forecast": "sunny"}
        assert mock.calls == [("get_forecast", {"city": "Boston"})]
