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
