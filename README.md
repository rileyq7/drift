# Drift

[![PyPI](https://img.shields.io/pypi/v/drift-lang.svg)](https://pypi.org/project/drift-lang/)
[![Python](https://img.shields.io/pypi/pyversions/drift-lang.svg)](https://pypi.org/project/drift-lang/)
[![VS Code](https://img.shields.io/visual-studio-marketplace/v/rileyq7.drift-lang?label=VS%20Code&color=blue)](https://marketplace.visualstudio.com/items?itemName=rileyq7.drift-lang)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Tests](https://github.com/rileyq7/drift/actions/workflows/test.yml/badge.svg)](https://github.com/rileyq7/drift/actions/workflows/test.yml)

**An intent-based language for agentic systems.** Write your agent in English-shaped blocks, run it as async Python.

```drift
schema EmailAnalysis {
  subject: string
  priority: one of "urgent", "normal", "low"
  category: one of "billing", "support", "sales", "spam", "personal"
  summary: string
  suggested_action: string
}

agent InboxTriage {
  model: "gpt-5.4-nano"
  budget: $0.10 per run

  step triage(emails: list<string>) -> list<EmailAnalysis> {
    let analyses = []
    for each email in emails parallel {
      let analysis = classify email as EmailAnalysis
      analyses.add(analysis)
    }
    for each a in analyses {
      if a.priority == "urgent" {
        respond "URGENT {a.subject}: {a.summary}"
      }
    }
    return analyses
  }
}
```

A full agent: model choice, budget, parallel fan-out, structured classification, conditional output. The transpiler emits async Python that runs on Drift's thin runtime. In one example run against OpenAI, 5 emails classified in ~1.8s for under a cent, returning 5 typed dataclasses — a single anecdotal measurement, not a benchmark; your latency and cost will vary by model and input.

## Install

```bash
pip install drift-lang
```

Optional extras:

```bash
pip install "drift-lang[mcp]"      # MCP tool support
pip install "drift-lang[dendric]"  # Dendric memory backend
pip install "drift-lang[all]"
```

## 30 seconds to your first agent

```bash
drift new hello
cd hello
drift run hello.drift --input '{"name":"Riley"}'
```

No API key required. Drift falls back to a mock provider so you see something work immediately. Drop an `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` into `.env` to use a real model.

## CLI

```
drift new <name>          Scaffold a starter project
drift run <file.drift>    Transpile and execute
drift check <file.drift>  Validate syntax
drift fmt <file.drift>    Format in place (--check for CI, --stdout to preview)
drift transpile <file>    Emit Python (use -o to write to a file)
drift mcp                 Run as an MCP stdio server (drift_check / transpile / run)
drift lex / parse         Debug tooling
```

## What's in the language

- **`agent`**: top-level unit. Has `model`, `budget`, `state`, `memory`, and `step`s.
- **`step`**: typed sub-procedure. Body is a sequence of declarative statements.
- **Intent verbs**: `summarize`, `extract`, `classify`, `translate`, `match`, `generate`, etc. Each one becomes a typed LLM call.
- **`confident<T>`**: confidence-gated branching. Run the cheap path when sure, `fail` or hand off to a stronger path when not. (There is no `escalate` keyword.)
- **`model { … }`**: multi-provider routing with `prefer`, `fallback`, and `upgrade when confidence < 0.7`. (`stream "fast" then "slow"` parses but is a compile error — no `.drift` syntax can drive the runtime's fast/slow bridge yet.)
- **`tool`**: declare external tools. `tool name from mcp "..."` or `tool name from python "mod:fn"`; REST tools use the inline block form (`tool name { endpoint: ... action ... }`) — there is no `from rest`. MCP runs against the official SDK.
- **`pipeline`**: composable flow. `->` is sequential and `=>` is parallel fan-out (`asyncio.gather` over items); `~>` (conditional) and `|>` (stream) parse but are compile errors — not implemented yet, and not silently downgraded to `->` either.
- **`for each x in xs parallel`**: `asyncio.gather` underneath.
- **`attempt / recover`**: structured error handling with retry, fail, and named arms.
- **`memory`**: short-term scratchpad or durable backend (Dendric). `remember`, `recall`, `deja_vu`, `forget`.
- **`define verb`**: extend the intent vocabulary with your own typed verbs.
- **Cross-agent calls**: `OtherAgent.step(args)` just works.

## Docs

| File | For |
|---|---|
| [`LLM.md`](./LLM.md) | Coding agents (Claude, Cursor, Copilot): complete reference for one-shot loading |
| [`docs/language.md`](./docs/language.md) | Humans learning Drift |
| [`docs/cookbook.md`](./docs/cookbook.md) | Copy-paste patterns |
| [`docs/gotchas.md`](./docs/gotchas.md) | Common mistakes |

## Examples

See [`examples/`](./examples) for working `.drift` programs and their generated Python:

- `hello.drift`: minimal agent
- `confident_demo.drift`: `confident<T>` branching
- `grant_checker.drift`: end-to-end intent + structured return
- `inbox_sorter.drift`: `for each … parallel` triage
- `inbox_triage_live.drift`: the canonical 30-line demo (runs against a real LLM; one example run did 5 emails in ~1.8s for under a cent — anecdotal, not a benchmark)
- `grant_checker_with_memory.drift`: Dendric-backed long-term memory
- `grant_checker_compare.drift`: citation-proof memory. Run 2's LLM reasoning cites Run 1 by name and makes side-by-side comparisons

## Status

Alpha. Language surface is stable, runtime works, 352/352 tests passing. OpenAI + Anthropic providers, MCP tools, Dendric memory, source-mapped runtime errors. Structured output uses provider-side strict JSON Schema on OpenAI; the Anthropic provider relies on schema-in-prompt plus validation-and-retry (it does not send a JSON Schema). Type system beyond `confident<T>` is on the roadmap.

## License

MIT
