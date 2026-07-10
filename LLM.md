# Drift — Reference for Coding Agents

You are writing `.drift` source code. Drift transpiles to async Python at run time. This file is the complete reference: read it once and you can write any Drift program. Optimized for correctness, not prose.

If a user asks for something not described here, refuse to invent syntax. Suggest the closest documented form instead.

---

## 1. Top-level structure

A `.drift` file is a sequence of declarations, separated by blank lines. Order doesn't matter for `agent`/`schema`/`config`/`tool`/`pipeline`/`define`, but every `define verb` must appear **before** any use of that verb.

Seven declaration forms:

```drift
config { ... }              -- project metadata
schema Name { fields }      -- structured types
tool name from <kind> ...   -- external integrations
agent Name { ... }          -- the unit of execution
pipeline name { ... }       -- composed flow across steps/agents
define verb X { ... }       -- custom intent verb
import { a, b } from "path" -- shared definitions across files
```

Only `agent` is required for a runnable program.

---

## 2. Comments

```drift
-- line comment to end of line
{- block comment, supports nesting {- like this -} -}
```

Comments may appear anywhere whitespace can. The lexer preserves them; `drift fmt` keeps them aligned to the current indent.

---

## 3. Literals

| Form | Example | Type |
|---|---|---|
| String | `"hello"` (supports `"{var}"` interpolation) | string |
| Number | `42`, `3.14` | number |
| Currency | `$0.10`, `$0.05`, `$1.50`, `£5`, `€20`, `¥100` | budget literal |
| Duration | `30s`, `5m`, `1h`, `2d`, `100ms` | duration literal |
| Bool | `true`, `false` | bool |
| List | `[a, b, c]` | list |
| Range | `0 to 100`, `1 to 10` | numeric range |

Currency and duration literals are only valid inside `budget:` and time-related fields. They are not general numbers.

---

## 4. Identifiers

- `snake_case` — variables, step names, parameter names, tool names, verbs, field names.
- `PascalCase` — agent names, schema names, type names.

The lexer distinguishes them. Using PascalCase where snake_case is expected (or vice versa) is a parse error.

---

## 5. Types

Primitive: `string`, `int`, `float`, `number`, `bool`.

Generic containers: `list<T>`, `map<K, V>` (not `dict`). No spaces inside the `<...>`. There is no `optional<T>` generic — that's a parse error. The parser does not currently accept `dict<...>`, `set<...>`, `tuple<...>`, or `optional<...>` — use `map<...>` for key/value pairs and `list<...>` for everything else. To mark a field nullable, use the trailing `optional` modifier on a schema field (see below), not a generic wrapper.

Schema-defined types: any `PascalCase` name declared via `schema Name { ... }`.

Confidence-wrapped: `confident<T>` — the result of an intent verb whose confidence is exposed for branching. Only `confident<T>` can be tested with `is confident`.

Refinement clauses (schemas and return types only):
- `number between A and B`
- `one of "a", "b", "c"`

Optional fields: append the `optional` modifier **after** the type in a schema field (`note: string optional`). Codegen emits `Optional[str] = None`. This is the only way to make a field nullable — there is no `optional<T>` type.

Example:
```drift
schema FitScore {
  score: number between 0 and 100
  recommendation: one of "strong fit", "possible fit", "weak fit", "no fit"
  confidence: number between 0 and 1
  tags: list<string>
  note: string optional
}
```

---

## 6. `config` block

```drift
config {
  name: "Project Name"
  version: "1.0.0"
}
```

Both fields are strings. Used in run banner and budget reporting. Not required.

---

## 7. `schema` declaration

```drift
schema Name {
  field_name: type
  field_name: type
  ...
}
```

- Each field is `name: type`.
- Types can include refinements (`number between 0 and 1`, `one of "..."`).
- Schemas become Python dataclasses at codegen.
- Use a schema as a step's return type to force structured output from an intent verb.

---

## 8. `agent` declaration

```drift
agent AgentName {
  model: <model spec>
  budget: <currency> per run
  state { ... }            -- optional
  memory: <memory spec>    -- optional
  step <name>(params) -> Type { body }
  ...
}
```

Every agent has at least one `step`. The first step is the default entry point if `--step` isn't passed to `drift run`.

### 8.1 `model` — three forms

**Single string:**
```drift
model: "claude-haiku"
```

**Prefer + fallback:**
```drift
model: prefer "claude-sonnet" fallback "gpt-4o"
```

**Stream-then (fast preview, slow reasoning):**
```drift
model: stream "claude-haiku" then "claude-sonnet"
```

**Block form (full routing):**
```drift
model {
  default: "claude-haiku"
  fallback: "gpt-4o-mini", "gpt-4o"
  never: "claude-opus"
  upgrade to "claude-sonnet" when {
    confidence < 0.7
    input_tokens > 8000
    step is risky_step
  }
}
```

Upgrade conditions (any one triggers): `confidence < <float>`, `input_tokens > <int>`, `step is <step_name>`.

**Routing rules** (handled by runtime, you don't need to encode them):
- `gpt-*`, `o1`, `o3`, `o4`, `openai/*` → OpenAI provider
- `claude-*`, `anthropic/*` → Anthropic provider
- Anything else with no key set → mock provider (returns deterministic stub data)

### 8.2 `budget`

```drift
budget: $0.10 per run
budget: £5 per run
```

Hard ceiling. Exceeding raises `BudgetExceeded` and stops the agent.

### 8.2.1 `quality` (optional)

```drift
quality: 0.85 minimum confidence
```

Overrides the default `min_confidence` threshold (0.85) used by `is confident` tests. The trailing `minimum confidence` is optional sugar; `quality: 0.85` works too.

### 8.3 `state`

```drift
state {
  user_id: string
  attempts: int = 0
  history: list<string> = []
}
```

Persists across steps in one run; reset per run. Defaults optional.

### 8.4 `memory`

Two forms.

**Shorthand (Dendric):**
```drift
memory: dendric("persona_key")
```
The string is a persona key — distinct keys give isolated memories. Requires `DATABASE_URL` and `OPENAI_API_KEY` (for embeddings); falls back to SQLite when `DATABASE_URL` is unset.

**Block form:**
```drift
memory {
  store: "sqlite"
  recall strategy: "semantic"
  max recall: 10 items
  decay: enabled
}
```

### 8.5 `step`

```drift
step name(p1: T1, p2: T2) -> ReturnType {
  <statements>
}
```

Modifiers (prefix the `step` keyword, mutually exclusive): `cached`, `manual`, `silent`.

- `cached` — memoizes on `(step name, args, kwargs)` for the lifetime of one agent instance (i.e. one run). Not a durable/cross-run cache. A failed call is not cached, so it retries next time.
- `manual` — excluded from auto-selection as the run's entry point (the step `drift run` picks when no `--step` is given). Still runs via `drift run file.drift --step name` or an ordinary internal call from another step.
- `silent` — suppresses `respond` output (both the printed line and the `self._outputs` entry) for the duration of that step only; nested non-silent steps called from within it are unaffected once they return.

`parallel step` **does not exist** — it's a parse error (`CodegenError`) raised at `drift check`/`transpile`/`run` time. A bare step-level modifier can't say what it runs in parallel *with*; express concurrency at the call site instead: `for each x in xs parallel { ... }` inside a step, or a pipeline `=>` fan-out edge.

A step with no return type returns whatever the last expression evaluates to.

---

## 9. Statements (inside step bodies)

### `let`
```drift
let name = <expression>
```
Single assignment. Type is inferred.

### `return`
```drift
return <expression>
```
Returns the value. Must match the step's declared return type.

### `respond`
```drift
respond "Message with {var} interpolation"
respond "Member access works too: {result.summary}"
respond "Including arithmetic: {score * 100}"
```
Prints to the user-visible output. The braces interpolate any expression in scope — bare variables, member access (`result.summary`), method calls (`name.upper()`), arithmetic. Multiple `respond`s accumulate; they're shown in order.

### `if` / `otherwise` / `otherwise if`
```drift
if <condition> {
  <statements>
} otherwise if <condition> {
  <statements>
} otherwise {
  <statements>
}
```
`condition` can include `is confident`, `>`, `<`, `==`, `!=`, `>=`, `<=`. The branch chains use `otherwise` and `otherwise if`, not `else`.

### `for each ... in ... [parallel]`
```drift
for each item in items {
  <statements>
}
```
Add `parallel` to fan out via `asyncio.gather`:
```drift
for each email in emails parallel {
  let result = classify email as EmailAnalysis
  results.add(result)
}
```

### `attempt` / `recover from`
```drift
attempt {
  <statements>
} recover from {
  BudgetExceeded -> { fail "out of money" }
  RateLimited    -> retry
  any error      -> { respond "something went wrong" }
}
```
Each arm is `<ErrorType> -> <body>` (or `any error -> <body>` for the catch-all). The body is either a single statement or a `{ ... }` block.

Error types (PascalCase, matched against runtime exception classes):
- `BudgetExceeded`
- `RateLimited`
- `AuthError`
- `StepFailed`
- `SchemaViolation`
- `ModelUnavailable`
- `DriftError` (base class — catches everything below it)
- `any error` — catch-all default arm

Inside a recover arm:
- `retry` — restart the attempt block
- `fail "<message>"` — abort with `StepFailed`
- Or any normal statements (fallback logic)

### `match` (pattern statement)
```drift
match value {
  "case_1" -> { <statements> }
  "case_2" -> { <statements> }
  _        -> { <statements> }    -- catch-all
}
```
Distinct from the `match` intent verb (which uses `against`).

### Memory statements

```drift
remember <expression> tagged "<tag>", "<tag>"
let memories = recall "<query>" for "<context>"
deja_vu match on <expression> {
  "<pattern>" -> { <statements> }
  any other  -> { <statements> }
}
forget memories tagged "<tag>"
forget memories older than 30d
forget memories where temp < 0.2
```

`remember` writes; `recall` returns a list of strings (the matched memory contents); `deja_vu` consults Dendric's archive for activations and routes on pattern match; `forget` removes by predicate.

### Intent expressions

An intent expression starts with a verb and is composed with clause keywords. It's a statement when used bare, an expression when used after `let`.

Verbs: `classify`, `extract`, `summarize`, `rate`, `generate`, `rewrite`, `answer`, `compare`, `decide`, `translate`, `match`, plus any verbs you declared with `define verb`.

**Picking the right verb when the output is a schema.** Every verb can produce structured output via `as <Schema>` — the verb mostly biases the LLM's framing:
- `classify` — assigns categories to input; pick when the result has discrete labels (priority, sentiment, type).
- `extract` — pulls specific fields out of text; pick when the input is unstructured and the schema's fields name things to find.
- `rate` — assigns numeric scores along axes; pick when the schema is numeric-heavy. (There is no `score` verb — use `rate`.)
- `summarize` — condenses text; pick when the result is mostly prose.
- `generate` — produces new content; pick when nothing in the input maps directly to the output.
- `answer` — Q&A over a context (`from <docs>`); pick when there's a question + a source.

When in doubt for "produce a full structured analysis from a single text input", `classify` and `extract` both work — they only differ in how the prompt is framed.

Clause keywords: `as`, `from`, `in`, `against`, `to`, `using`, `considering`, `with`.

Examples:
```drift
let label  = classify email as EmailAnalysis
let summary = summarize document as string
let answer  = answer "what is the policy?" from doc as string
let result  = rate company against criteria as FitScore
let fit     = match candidate against criteria as MatchResult
let drafted = generate a friendly reply considering tone, length as string
```

Every intent call needs an `as` clause naming the result type. The clauses after `as` (`from`, `in`, `against`, `to`, `using`, `considering`, `with`) attach context to the LLM call. Their values can be variables, expressions, or comma-separated lists.

`confident<T>` is enabled by suffixing the type:
```drift
let scored = rate company against criteria as confident<FitScore>
if scored is confident {
  return scored.value
} otherwise {
  fail "low confidence — needs human review"
}
```
`is confident` tests `result.confidence >= agent.min_confidence` (default 0.85, override with `quality:`). `scored.value` is the unwrapped result.

There is no `escalate` keyword. To exit a step early on low confidence, use `fail "<message>"` (raises `StepFailed`) or `return` a sentinel value declared in the schema.

---

## 10. Cross-agent calls

```drift
let result = OtherAgent.step_name(arg1, arg2)
```
Transpiles to `await OtherAgent().step_name(arg1, arg2)`. The called agent runs with its own budget and model config.

---

## 11. `tool` declaration

Three forms.

**MCP (official SDK; stdio or streamable HTTP):**
```drift
tool weather from mcp "https://example.com/mcp"
tool fs       from mcp "stdio:/usr/local/bin/mcp-fs"
```
Inside a step:
```drift
let forecast = weather.get_forecast(city: "Boston")
```

**Python module function:**
```drift
tool slack from python "myproj.integrations:slack_post"
```
The path is `module.path:function_name`.

**REST (inline block):**
```drift
tool github {
  endpoint: "https://api.github.com"
  auth: env("GITHUB_TOKEN")
  action list_issues(repo: string) -> list<dict> {
    GET "/repos/{repo}/issues"
  }
  action create_issue(repo: string, title: string) -> dict {
    POST "/repos/{repo}/issues"
  }
}
```
HTTP methods: `GET`, `POST`, `PUT`, `PATCH`, `DELETE`, `HEAD`. Path uses `{param}` for interpolation. `auth: env("VAR")` reads the token from environment.

---

## 12. `pipeline` declaration

Compose steps and agent calls into a directed graph. **Pipeline names must be PascalCase** (`Triage`, not `triage`).

```drift
pipeline Triage {
  input_email -> Classifier.tag -> Router.route => Action.execute
}
```

Operators (what codegen actually emits today):
- `->` sequential (default) — thread the previous node's output into the next.
- `=>` **parallel fan-out** — the previous node's output must be iterable; the next node runs concurrently over each item via `asyncio.gather`. Not "strict sequential".
- `~>` conditional and `|>` stream — **not implemented.** Both parse (the grammar accepts them) but codegen raises `CodegenError` the moment it sees one as a pipeline edge, at `drift check`/`transpile`/`run` time — there is no silent fallback. Don't emit these; use `->` or `=>` instead. (Note: `|>` *inside an expression*, e.g. `x |> f |> g`, is a separate, working function-composition pipe — unrelated to `|>` as a pipeline edge.)

Each node is a step name (current file's first agent) or `Agent.step` (any agent). Pipelines run via `drift run --pipeline <name>`.

---

## 13. `define verb` declaration

```drift
define verb evaluate {
  pattern: "evaluate {target} against {criteria}"
  prompt: "Evaluate the given input thoroughly."
  output: FitScore
  temperature: 0.2
}
```
Fields (all optional except where logic requires them):
- `pattern` — the call-site shape, used by the parser to validate verb invocations.
- `prompt` — the system prompt sent to the LLM.
- `output` — the default output type when the call site omits `as`.
- `temperature` — sampling temperature for this verb's LLM call.

Once declared, `evaluate` works like any built-in intent verb. Custom verbs use the same clause keywords (`as`, `from`, `against`, etc.).

---

## 14. `import` declaration

```drift
import { GrantSchema, FitScore } from "./schemas.drift"
import GrantChecker from "./agents/checker.drift"
```
Pulls named declarations into the current file. Path is relative to the importing file.

---

## 15. Standard library imports

Drift ships modules you can import at the top of any file. Stdlib modules live under the `drift/` namespace — you **must** write the `drift/` prefix. Codegen maps `drift/io` → `from drift.io import ...`; a bare `"io"` would emit `from io import ...` and collide with Python's own stdlib (ImportError). The actual function names:

```drift
import { read, write, fetch_url, load_pdf, load_csv } from "drift/io"
import { email, slack, webhook, push } from "drift/notify"
import { redact_pii, check_content, sanitize, rate_limit } from "drift/safety"
import { filter_, sort, group_by, deduplicate, paginate } from "drift/data"
import { now, wait, deadline, schedule } from "drift/time"
import { chunk, tokenize, similarity, embed } from "drift/text"
import { log, trace, metric, cost_report } from "drift/observe"
```

Use any imported function as a normal call inside step bodies. Examples:
- `notify.email(to: "x@y.com", subject: "...", body: "...")`
- `time.wait(2.0)` — async, awaitable
- `safety.redact_pii(user_message)`
- `io.fetch_url("https://...")` — async
- `text.chunk(long_doc, max_chars: 2000)`

Stubs vs real: some functions (notify.email, observe.metric) are v0.2 stubs that log to stdout instead of doing the real thing. Treat them as wired but no-op until production needs them.

---

## 16. Runtime behavior (essential context)

When `drift run file.drift` runs, the transpiler:
1. Lexes → parses → generates Python (`file.py` written next to the source).
2. Loads `.env` walking up from the `.drift` file's dir.
3. Picks a provider by model name + available keys (see 8.1).
4. Instantiates the first `agent` class found.
5. Calls the entry step (`--step` or the first step).
6. Tracks token cost; aborts on `BudgetExceeded`.
7. Prints a cost report at the end.

**Errors raised by the runtime:**
- `DriftError` (base)
- `BudgetExceeded`
- `StepFailed`
- `SchemaViolation` (intent output didn't match schema after N retries)
- `ModelUnavailable` (404, network, retryable)
- `RateLimited` (429; carries `retry_after`)
- `AuthError` (401, 403; not retryable)

Handle them with `attempt`/`recover`.

**Mock provider:** when no API key matches the requested model family, the runtime falls back to a deterministic mock that returns plausible structured data. The CLI prints `provider: mock` in the banner so this is never silent.

---

## 17. Quick patterns

### Confidence-gated branching

Two equally valid forms.

**A. `confident<T>` wrapper (preferred when you want a typed `.value`):**
```drift
schema Decision { approved: bool, reasoning: string }

step assess(input: string) -> Decision {
  let scored = rate input against rubric as confident<Decision>
  if scored is confident {
    return scored.value
  }
  fail "low confidence — escalate"
}
```

**B. Plain schema with explicit confidence field:**
```drift
schema Decision {
  approved: bool
  reasoning: string
  confidence: number between 0 and 1
}

step assess(input: string) -> Decision {
  let result = rate input against rubric as Decision
  if result.confidence < 0.7 {
    fail "low confidence — escalate"
  }
  return result
}
```

Both work. Pick `confident<T>` when you want the agent's `min_confidence` (or `quality:`) threshold to control the gate; pick form B when you want to gate at an arbitrary per-step threshold.

### Parallel triage
```drift
step sort(emails: list<string>) -> list<EmailAnalysis> {
  let results = []
  for each email in emails parallel {
    let analysis = classify email as EmailAnalysis
    results.add(analysis)
  }
  return results
}
```

### Retry with budget escape
```drift
step fetch_and_score() -> FitScore {
  attempt {
    let data = api.get_company()
    return rate data against criteria as FitScore
  } recover from {
    RateLimited    -> retry
    BudgetExceeded -> fail "ran out of budget — try a cheaper model"
  }
}
```

### Memory-aware agent
```drift
agent Advisor {
  model: "claude-haiku"
  memory: dendric("user_123")
  budget: $0.20 per run

  step advise(question: string) -> string {
    let context = recall question for "advice"
    let answer = answer question using context as string
    remember answer tagged "advice", "user_123"
    return answer
  }
}
```

### MCP tool use
```drift
tool fs from mcp "stdio:/usr/local/bin/mcp-fs"

agent FileReader {
  model: "claude-haiku"
  step read_doc(path: string) -> string {
    let content = fs.read(path: path)
    return summarize content as string
  }
}
```

---

## 18. Anti-patterns — DON'T do these

- **Don't invent verbs.** If you need a verb not in section 9, declare it with `define verb` first.
- **Don't use `escalate`.** It doesn't exist. Use `fail "<message>"` or return a schema with a confidence field and gate on it.
- **Don't use `else`.** Drift uses `otherwise` and `otherwise if`.
- **Don't put spaces inside generic types.** `list<string>` not `list< string >`. The formatter normalizes this but parsing accepts both.
- **Don't put space between function name and `(`.** `step greet(name: string)` not `step greet (name: string)`.
- **Don't omit the `as <Type>` clause** on intent verbs. Without it, codegen can't pick an output schema and the runtime returns raw text.
- **Don't use Python-style boolean operators.** Use `and`, `or`, `not` (lowercase).
- **Don't try to import from arbitrary URLs.** `import` only accepts relative paths.
- **Don't mix `memory:` shorthand and `memory {}` block.** Pick one form per agent.
- **Don't expect `state` to persist between runs.** It's per-run scratch. Use `memory:` for cross-run persistence.
- **Don't write `recover on X`.** It's `recover from { X -> body }`.
- **Don't use snake_case for pipeline names.** Pipeline names must be PascalCase.

---

## 19. CLI

```
drift new <name>             # scaffold a starter project
drift run <file> [--step S] [--input '<json>'] [--watch] [--trace]
drift check <file>           # syntax-only validation
drift fmt <file> [--check] [--stdout]
drift transpile <file> [-o out.py]
drift lex <file> | drift parse <file>   # debug
```

`drift run` auto-loads `.env` from the file's directory tree, transpiles to a `.py` next to the source, runs the first agent's first step (or `--step`/`--agent` if specified). `--input` takes a JSON object mapped to the step's parameters by name.

---

## 20. When uncertain, do this

1. Match the user's task to one of the section-17 patterns. Start from a known shape.
2. Use existing intent verbs (section 9) before defining new ones.
3. If a structured output is needed, declare a `schema` for it.
4. If you need confidence-gated branching, return `confident<T>` from the intent.
5. If a step might fail on a provider error, wrap it in `attempt`/`recover`.
6. Never emit Python — Drift is the target language. The runtime handles the rest.
7. After writing, mentally trace one execution: which step runs, which model, what gets returned. If you can't trace it, the user can't either.
