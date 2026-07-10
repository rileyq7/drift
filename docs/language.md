# Drift Language Guide

This is the human-shaped reference. If you're a coding agent, read [`LLM.md`](../LLM.md) instead — it's the same material packed for one-shot consumption.

## What Drift is

Drift is an intent-based language for building LLM agents. You describe **what** an agent should do in declarative blocks; the transpiler emits async Python that runs on a thin runtime. The runtime handles model routing, cost tracking, retries, schema validation, and memory.

The bet: agent code is mostly orchestration, and orchestration reads better as a spec than as control flow.

## The shape of a Drift program

Every Drift program is a `.drift` file containing one or more top-level declarations. The most important one is `agent`:

```drift
agent GrantChecker {
  model: "claude-haiku"
  budget: $0.10 per run

  step assess(application: string) -> Decision {
    let scored = rate application against rubric as confident<Decision>
    if scored is confident {
      return scored.value
    }
    fail "low confidence — needs human review"
  }
}
```

Seven top-level forms:

| Declaration | Purpose |
|---|---|
| `config { ... }` | Project metadata (name, version) |
| `schema Name { ... }` | A structured type |
| `tool name from ...` | An external integration (MCP, Python, REST) |
| `agent Name { ... }` | The unit of execution |
| `pipeline name { ... }` | A composed flow across steps/agents |
| `define verb X { ... }` | A custom intent verb |
| `import { a, b } from "path"` | Pull names from another file |

## Steps and intent verbs

Inside an agent, you write `step`s. A step is a typed procedure with a body of declarative statements. The most important statements are **intent verbs** — verbs like `summarize`, `classify`, `extract`, `generate`, `answer`, `rate`. Each one becomes a typed LLM call.

```drift
step triage(email: string) -> EmailAnalysis {
  let analysis = classify email as EmailAnalysis
  if analysis.priority == "urgent" {
    respond "URGENT: {analysis.subject}"
  }
  return analysis
}
```

Every intent expression must end with `as <Type>`. The clauses in between (`from`, `against`, `using`, `considering`, `with`) attach context to the LLM call.

## Confidence-gated branching

The killer feature. Suffix an intent's output type with `confident<T>` to get a value plus a confidence score:

```drift
let scored = rate company against criteria as confident<FitScore>
if scored is confident {
  return scored.value
} otherwise {
  fail "low confidence — needs human review"
}
```

`is confident` tests against the agent's `min_confidence` threshold (default 0.85). Run the cheap model when sure, `fail` (raises `StepFailed`) or return a sentinel schema value when not. There is no `escalate` keyword — see [gotchas](./gotchas.md#there-is-no-escalate).

## Model routing

Three forms, simplest to richest:

```drift
model: "claude-haiku"
model: prefer "claude-sonnet" fallback "gpt-4o"
model: stream "claude-haiku" then "claude-sonnet"

model {
  default: "claude-haiku"
  fallback: "gpt-4o-mini"
  upgrade to "claude-sonnet" when {
    confidence < 0.7
    input_tokens > 8000
  }
}
```

The runtime picks a provider by model name: `gpt-*`/`o1`/`o3`/`o4` → OpenAI, `claude-*` → Anthropic. With no key set it uses a mock provider (deterministic stub data) so your agent always runs.

## Memory

Two backends.

**Dendric** (semantic, persistent, with neurotransmitter-modulated recall):
```drift
memory: dendric("user_persona_key")
```

**SQLite** (simpler, file-based):
```drift
memory {
  store: "sqlite"
  recall strategy: "semantic"
  max recall: 10 items
  decay: enabled
}
```

Memory operations:
```drift
remember answer tagged "advice", "user_123"
let context = recall question for "advice"
deja_vu match on situation {
  "first time" -> { ... }
  any other    -> { ... }
}
forget memories tagged "user_123"
forget memories older than 30d
```

## Parallel work

```drift
for each email in emails parallel {
  let result = classify email as EmailAnalysis
  results.add(result)
}
```

Compiles to `asyncio.gather`. Order of results matches input order.

## Error handling

```drift
attempt {
  let data = api.get_company()
  return rate data against criteria as FitScore
} recover from {
  RateLimited    -> retry
  BudgetExceeded -> fail "ran out of budget"
  any error      -> {
    respond "fallback path"
    return default_score
  }
}
```

Each arm is `<ErrorType> -> <body>` (or `any error -> <body>` for the catch-all). Error types are PascalCase and match runtime exception classes: `BudgetExceeded`, `RateLimited`, `AuthError`, `StepFailed`, `SchemaViolation`, `ModelUnavailable`, `DriftError`. Body is a single statement or `{ ... }`. Inside an arm: `retry`, `fail "<msg>"`, or any normal statements.

## Cross-agent calls

```drift
let summary = Summarizer.summarize(text)
```

Each agent has its own budget and model config. They compose naturally.

## Tools

Three integration paths:

```drift
-- MCP (model context protocol — works with Anthropic's official SDK)
tool weather from mcp "https://example.com/mcp"

-- Python function
tool slack from python "myproj.integrations:slack_post"

-- REST inline
tool github {
  endpoint: "https://api.github.com"
  auth: env("GITHUB_TOKEN")
  action list_issues(repo: string) -> list<dict> {
    GET "/repos/{repo}/issues"
  }
}
```

## Pipelines

For non-trivial multi-step flows, pipelines are clearer than nested steps. Pipeline names are PascalCase.

```drift
pipeline Triage {
  input -> Classifier.tag -> Router.route => Action.execute
}
```

Operators (current behavior): `->` sequential; `=>` **parallel fan-out** — the upstream node's output must be iterable and the downstream node runs concurrently over each item via `asyncio.gather`. `~>` (conditional) and `|>` (stream) are **parsed but not implemented** — using either as a pipeline edge is a compile error (`CodegenError`) at `drift check`/`transpile`/`run` time, not a silent fallback. Use `->` or `=>` instead.

## See also

- [`cookbook.md`](./cookbook.md) — copy-paste patterns
- [`gotchas.md`](./gotchas.md) — common mistakes
- [`../LLM.md`](../LLM.md) — full reference, dense
- [`../examples/`](../examples) — working programs
