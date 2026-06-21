# Drift as an MCP server

Other coding agents (Claude Code, Cursor, anything MCP-aware) can use Drift as a tool — they get `drift_check`, `drift_transpile`, and `drift_run`. Useful for letting an outer agent compose a Drift program, validate it, and run it without leaving its host environment.

## Install

```bash
pip install "drift-lang[mcp]"
```

## Register with Claude Code

Drop into your project's `.mcp.json` (or `~/.claude/mcp.json` for global):

```json
{
  "mcpServers": {
    "drift": {
      "command": "drift",
      "args": ["mcp"]
    }
  }
}
```

Restart Claude Code. The three Drift tools appear under the `drift` namespace.

## Register with Cursor

In your project's `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "drift": {
      "command": "drift",
      "args": ["mcp"],
      "env": {
        "OPENAI_API_KEY": "${env:OPENAI_API_KEY}"
      }
    }
  }
}
```

The optional `env` block lets the inner agent's runs reach real LLMs.

## Tools

| Tool | Args | Returns |
|---|---|---|
| `drift_check` | `source` (string) | `{ok: bool, message?, line?, col?}` |
| `drift_transpile` | `source` (string) | `{ok: bool, python: string}` or `{ok: false, error: string}` |
| `drift_run` | `source` (string), `input` (JSON-encoded string, optional) | `{ok: bool, result, banner}` |

`drift_run` will use whichever LLM provider is configured via the server's environment. Pipe through `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY` via the `env` block. With no key, the runtime falls back to the deterministic mock provider.

## When this matters

- A coding agent can iterate on a Drift program: write → check → fix → run, all without escaping to a shell.
- An outer agent can use Drift as a sub-language for cost-tracked LLM orchestration. The outer agent's host doesn't need to know Drift exists; it just sees three tools.
