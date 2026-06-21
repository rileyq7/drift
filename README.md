# Drift

**An intent-based language for agentic systems.** Write your agent in English-shaped blocks, run it as async Python.

```drift
agent GrantChecker {
  model: "claude-haiku"
  budget: $0.10 per run

  step assess(application: string) -> Decision {
    let summary = summarize the application as string
    let score = score eligibility from 1 to 10 as int

    if confident<score> {
      return Decision { approved: score > 7, summary: summary }
    } else {
      escalate to human review
    }
  }
}
```

That's a full agent — model choice, budget, an intent verb (`summarize`), confidence-gated branching, structured return. The transpiler turns it into async Python that runs on Drift's thin runtime.

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

No API key required — Drift falls back to a mock provider so you see something work immediately. Drop an `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` into `.env` to use a real model.

## CLI

```
drift new <name>          Scaffold a starter project
drift run <file.drift>    Transpile and execute
drift check <file.drift>  Validate syntax
drift transpile <file>    Emit Python (use -o to write to a file)
drift lex / parse         Debug tooling
```

## What's in the language

- **`agent`** — top-level unit. Has `model`, `budget`, `state`, `memory`, and `step`s.
- **`step`** — typed sub-procedure. Body is a sequence of declarative statements.
- **Intent verbs** — `summarize`, `extract`, `classify`, `translate`, `match`, `generate`, etc. Each one becomes a typed LLM call.
- **`confident<T>`** — confidence-gated branching. Run the cheap path when sure, escalate when not.
- **`model { … }`** — multi-provider routing with `prefer`, `fallback`, `upgrade when confidence < 0.7`, and `stream "fast" then "slow"`.
- **`tool name from python|mcp|rest`** — declare external tools. MCP runs against the official SDK.
- **`pipeline`** — composable flow with `->`, `=>`, `~>`, `|>` operators.
- **`for each x in xs parallel`** — `asyncio.gather` underneath.
- **`attempt / recover`** — structured error handling with retry, fail, and named arms.
- **`memory`** — short-term scratchpad or durable backend (Dendric). `remember`, `recall`, `deja_vu`, `forget`.
- **`define verb`** — extend the intent vocabulary with your own typed verbs.
- **Cross-agent calls** — `OtherAgent.step(args)` just works.

## Docs

| File | For |
|---|---|
| [`LLM.md`](./LLM.md) | Coding agents (Claude, Cursor, Copilot) — complete reference for one-shot loading |
| [`docs/language.md`](./docs/language.md) | Humans learning Drift |
| [`docs/cookbook.md`](./docs/cookbook.md) | Copy-paste patterns |
| [`docs/gotchas.md`](./docs/gotchas.md) | Common mistakes |

## Examples

See [`examples/`](./examples) for working `.drift` programs and their generated Python:

- `hello.drift` — minimal agent
- `confident_demo.drift` — `confident<T>` branching
- `grant_checker.drift` — end-to-end intent + structured return
- `inbox_sorter.drift` — `for each … parallel` triage
- `grant_checker_with_memory.drift` — Dendric-backed long-term memory

## Status

Alpha — language surface is stable, runtime works, 344/344 tests passing. Voice primitives parse but adapters aren't wired yet. Type system beyond `confident<T>` is on the roadmap.

## License

MIT
