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

Currency and duration literals are *intended* for `budget:` and time-related fields — that's the only place they're documented and the only place that makes semantic sense. In practice the parser doesn't restrict where they can appear: `let price = $5.00` or `let x = 5m` both parse and transpile cleanly, silently discarding the unit and becoming a plain number (`5.0`, `300.0` seconds respectively) usable anywhere a `number` is. This isn't enforced — don't rely on using a currency/duration literal outside a budget/time field being rejected.

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

### 7.1 Constructing a schema value directly

Most schema instances come from an intent verb's `as <Schema>` result — but to build one yourself (e.g. to `return` a value assembled from pieces rather than straight from an LLM call), use `SchemaName { field: value, ... }`:
```drift
return Outcome {
  ticket_id: ticket_id,
  category: scored.value.category,
  action: "responded"
}
```
Fields can be comma-separated on one line or newline-separated across multiple lines — both work. This is the only reliable constructor form. `SchemaName(field: value, ...)` (parens instead of braces) also parses and works *on a single line*, but breaks with a `ParseError` the moment a newline appears inside the parens — since multi-field constructors are almost always written multi-line for readability, prefer `{ }` and don't use the paren form at all.

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

**Stream-then — parses, but not usable from .drift yet:**
```drift
model: stream "claude-haiku" then "claude-sonnet"
```
This is a **compile error** (`CodegenError`), not a working feature. The syntax parses and the runtime has a real `StreamThenRouter.stream_then_call()` that fires both models concurrently and calls a bridge callback — but no `.drift` step-body syntax exists to supply that callback, and generated intent calls never invoke `stream_then_call()` at all. Emitting it silently would behave exactly like `model: default "claude-sonnet"` (no bridge, no speedup, no error), so codegen refuses instead. Usable today only from hand-written Python via `drift.runtime.StreamThenRouter`. Don't suggest this syntax to a user who wants working fast/slow streaming — write `model: default "claude-sonnet"` and note the limitation.

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
budget: $1 max        -- `max` is shorthand for `per run`
```

Hard ceiling. Exceeding raises `BudgetExceeded` and stops the agent. **`per run` (or `max`) is the only period implemented.** `per day`/`per company`/anything else is a `ParseError` — there's no cross-run/cross-entity budget ledger in Drift, so declaring one would either need to be silently downgraded to per-run (misleading) or rejected; it's rejected. Don't emit `per day` or similar expecting it to work.

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
  store: "sqlite://:memory:"
  recall strategy: "semantic"
  max recall: 10 items
  decay: enabled
}
```
`store` must be a `sqlite://` URL (`sqlite://:memory:` for in-process/lost-on-restart, `sqlite://path/to/file.db` for file-backed) — a bare `"sqlite"` is a `ValueError` at agent construction, not a recognized shorthand. `recall strategy: "semantic"` currently behaves identically to `"relevant"` (substring match) — no embedding model is wired in yet; falls back visibly (prints a notice), not silently. `decay: enabled` parses and is stored but has no effect — there's no decay/forgetting-over-time logic implemented for the SQLite backend.

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

A literal brace character (not interpolation) is written doubled — `{{` for a literal `{`, `}}` for a literal `}` — e.g. `respond "set notation: {{1, 2, 3}}"` prints `set notation: {1, 2, 3}`, not an interpolation attempt. This applies anywhere `{...}` interpolation is recognized (`respond`, string literals generally, REST tool path templates).

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
- `DriftError` (base class — catches only the Drift runtime exceptions listed above, not exceptions from tool calls or external I/O)
- `any error` — genuinely catches anything (`except Exception`), including non-`DriftError` exceptions like network errors from a REST/MCP tool call, exceptions raised by a `python`-kind tool's own code, or file I/O errors from `drift/io`. This is the arm to use around a `tool` call if you want the attempt block to survive an API outage or similar external failure — `DriftError` alone will not catch those.

Inside a recover arm:
- `retry` — restart the attempt block **from the top, literally** — any statement before the point of failure re-executes on every attempt, including side effects. `results.add("before") ... let r = classify x as string ... results.add("after")` with `RateLimited -> retry` will append `"before"` once per attempt (not just once total) if the failure happens after the first append but before the classify call resolves — retries aren't automatically idempotent. If you need a side effect to happen exactly once regardless of retries, put it either strictly after the risky call, or outside the `attempt` block entirely (before it, if it must happen first; after it, once the attempt has already succeeded).
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

> **Mock backend caveat.** Without a configured Dendric store (`DATABASE_URL` unset — the default, and what every test uses), memory runs against an in-process SQLite mock. Only `forget memories tagged "..."` does anything there — it deletes by tag-substring. `forget memories older than Nd` and `forget memories where temp < N` are silent no-ops against the mock (it tracks no ages or temperature), and `deja_vu` never fires (no archive/sleep-cycle to surface activations). All of this works against a real Dendric backend. If a `.drift` file uses age/temp-based `forget` or `deja_vu` and you need to verify it actually did something, either configure Dendric or check the behavior in `drift/runtime/core.py`'s mock store — don't assume the mock run proved the logic works.

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
- `compare` / `decide` / `match` — weigh one thing against another; pick when the input is a single item and the schema's judgment depends on a second thing (criteria, a candidate, an alternative). Put the *other* thing in the `against` clause: `compare a against b as Comparison`, `decide option against criteria as Verdict`. Don't write `compare a, b as ...` — a comma-separated input list is only meaningful for `extract`'s field-name list (see below); for every other verb, a bare comma-separated *input* is collected as literal joined text (`"a , b"`), not a reference to each variable's value — pass the second item via `against` instead.
- `rewrite` — transforms existing text; put style/tone guidance in `using <guidance>`.
- `translate` — put the target language in `to <language>` (a string, quoted or a variable).

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

A bare (unquoted) multi-word input like `generate a warm one-sentence greeting as string` is free-form English collected up to the next clause keyword — the actual LLM prompt text preserves hyphenated words correctly (`one-sentence` stays `one-sentence`, not `one - sentence`). `drift fmt`, however, only tokenizes (it doesn't parse), so it can't tell a hyphen inside such a description apart from a subtraction operator and will visually rewrite `one-sentence` to `one - sentence` in the `.drift` file itself — cosmetic only (idempotent, and the prompt text stays correct when re-parsed), but if you want to avoid the visual churn, prefer a quoted string (`generate a warm "one-sentence" greeting`) or reword without the hyphen.

**Single bare word = variable, not literal text.** This is the one case the "multi-word is free text" rule above doesn't cover: a bare (unquoted) input that is exactly *one* word is always parsed as a variable reference, never as a one-word literal description — `classify email as EmailAnalysis` reads the value of a variable named `email`, and `summarize doc as string` reads the value of a variable named `doc`. This is intentional and is what makes `let context = recall question for "advice"` (§ "Memory-aware agent" example) work — `question` there is a real step parameter, not the four-letter word "question". The practical hazard: if you write a one-word description that happens to match an in-scope variable name (`generate summary as string` when a variable called `summary` also exists), Drift silently uses that variable's *runtime value* as the LLM input instead of the word you meant as a description — `drift check` reports clean syntax either way, with no way to tell the two cases apart from the output alone. If you mean the literal word, quote it: `generate "summary" as string`. The same rule applies to `recall <description> for <key>` — `recall tips for advice` treats both `tips` and `advice` as variable references, not literal text; quote either side you mean literally.

**Clause-keyword trap (more serious — silently drops text, not just visual churn).** If a bare multi-word description contains one of the 8 clause keywords (`as`, `from`, `in`, `against`, `to`, `using`, `considering`, `with`) as an ordinary English word — e.g. `generate a reply to this ticket as string`, where "to" is just part of the sentence — the parser has no way to tell that apart from an intentional new clause. It truncates your description at that word (`input_data` becomes just `"a reply"`) and starts parsing `this ticket ... as string` as a new clause/statement, which can silently produce nonsense or a confusing parse/codegen error far from the real cause. **Always wrap a multi-word description in a quoted string** to make it unambiguous:
```drift
let reply = generate "a reply to this ticket" as string        -- correct
let reply = generate a reply to this ticket as string           -- WRONG: truncates at "to"
```
This is not optional style advice — an unquoted description containing any of the 8 keywords is a real correctness bug waiting to happen, not just a formatting quirk.

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
MCP method names aren't validated at `drift check`/transpile time — there's no static schema, since the server's tool list is only known by actually connecting to it. A typo'd method name (`weather.get_forcast`) parses and transpiles cleanly and, in tests using the mock MCP session (`use_mock()`), even *runs* cleanly — the mock echoes back any unrecognized method as a fake success rather than erroring, by design (so tests aren't blocked on full server wiring). It will only fail against a real server, as an MCP protocol error at call time. Double-check MCP method names against the actual server (or its docs) rather than trusting a clean `drift check`/mock-backed `drift run`.

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

Each node is a step name (current file's first agent) or `Agent.step` (any agent) — **every node, including the first, must be a real declared step.** There's no syntax for "start the pipeline with this raw data" — if you need the pipeline's `--input` to become the starting value, declare an entry step that takes it as a parameter and returns it (or transforms it), e.g. `step tickets(batch: list<Ticket>) -> list<Ticket> { return batch }`, then start the pipeline with `tickets => ...`. A bare identifier that isn't a step name (like a variable) still parses — it's read as `<first-agent>.<that-name>` — but fails at runtime with an `AttributeError` since no such step exists.

**`--input` for `--pipeline` runs works differently than for a plain agent run.** For a plain `drift run file.drift --input '{"name": "Riley"}'`, the JSON is mapped to the step's parameters *by name* (`step(name="Riley")`) — this is NOT true for pipelines. For `drift run file.drift --pipeline P --input '...'`, the parsed JSON is passed as a single positional value to the entry node's one parameter, whole and unmapped — `--input '{"batch": [...]}'` does NOT become `tickets(batch=[...])`; it becomes `tickets({"batch": [...]})`, the *whole* dict as the value of `batch`. If your entry step takes one parameter, pass `--input` as exactly that parameter's shape directly (e.g. `--input '[{"ticket_id": "T-1", ...}, ...]'` for `step tickets(batch: list<Ticket>)`), not wrapped in an object keyed by the parameter name.

Pipeline run via `drift run --pipeline <name>`. JSON object/dict values passed via `--input` (for both plain and `--pipeline` runs) ARE coerced into declared schema types — a `Schema`-typed parameter receives a real dataclass instance (attribute access like `param.field` works), not a bare `dict`.

### 12.1 Pipeline modifiers

```drift
pipeline Triage {
  timeout: 30s
  use A use B
  A.classify -> B.route on failure in route: skip and continue
  A.classify -> B.route on budget exceeded: finish current item then stop
}
```

- `timeout: <duration>` — real: wraps the whole run in `asyncio.wait_for`; a timed-out run raises `asyncio.TimeoutError`.
- `on failure in <step>: skip ...` — real: wraps that node in try/except, logs, and continues the pipeline with the pre-failure value. **Only a `skip...`-prefixed phrase is implemented.** Any other phrase (`retry twice then fail`, `notify oncall`, etc.) parses but is a `CodegenError` at compile time — it is not silently ignored, but it's also not interpreted; write your own retry logic in the step instead if you need it.
- `on budget exceeded: <phrase>` — real but limited: catches `BudgetExceeded`, logs the declared phrase, and **always re-raises**. The phrase is not parsed as an instruction (there's no "finish current item then stop" behavior distinct from "stop") — it's just echoed into the log line so you know which handler fired.
- `schedule: "<phrase>"` — **not implemented.** Parses, but is a `CodegenError` at compile time: there's no daemon/cron loop in Drift, so nothing would ever re-invoke `run()` on a schedule. Drive scheduling externally (cron, a task queue) and call `drift run` from there.

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
- `pattern` — **documentation only.** Written to make the call-site shape readable at the definition site; not validated against actual call sites anywhere. A call that doesn't match the declared pattern still works (or fails) exactly as if `pattern` weren't there.
- `prompt` — the system prompt sent to the LLM.
- `output` — the default output type when the call site omits `as`.
- `temperature` — sampling temperature for this verb's LLM call. Passed through to the provider request; omitted (provider default applies) when left at 0/unset.

Once declared, `evaluate` works like any built-in intent verb. Custom verbs use the same clause keywords (`as`, `from`, `against`, etc.).

---

## 14. `import` declaration

```drift
import { GrantSchema, FitScore } from "./schemas.drift"
import GrantChecker from "./agents/checker.drift"
```
Pulls named declarations into the current file. Path is relative to the importing file. `drift run` automatically transpiles any `.drift` file reachable via `import` before running the importer, and resolves the generated Python correctly regardless of the importing file's own subdirectory or the caller's working directory — you don't need to manually `drift transpile` a dependency first, or `cd` into its directory. (`drift check`/`drift transpile` on the importer alone do NOT recurse into or validate the imported file — only `drift run` does the dependency walk, since only `drift run` actually executes the generated Python and needs the import to resolve.)

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

Use any imported function as a **bare call** inside step bodies — the import brings the function name itself into scope, not a namespaced module object. There is no `io.`/`notify.`/`time.`/etc. prefix at the call site, even though the `from` clause says `"drift/io"`; `import { fetch_url } from "drift/io"` makes `fetch_url` callable directly. (`io.fetch_url(...)` looks natural by analogy to the tool-call syntax in §11, but tools bind a namespaced object while stdlib imports don't — writing the prefix is a `NameError` at runtime, not caught by `drift check`.) Examples:
- `email(to: "x@y.com", subject: "...", body: "...")`
- `wait(2.0)` — async, awaitable
- `redact_pii(user_message)`
- `fetch_url("https://...")` — async
- `chunk(long_doc, max_chars: 2000)`

**`drift/data`'s `filter_`/`group_by` (and `sort`/`deduplicate`'s optional `key` parameter) need a Python callable — Drift has no lambda/function-reference syntax, so there's no way to construct one from `.drift` source.** `filter_(items, predicate)` and `group_by(items, key)` always require a predicate/key argument, making them **effectively unusable** — passing a bare identifier there (e.g. `filter_(items, some_name)`) transpiles to a plain Python name reference that isn't a real function, and crashes with `NameError` at runtime (`drift check` doesn't catch it — it's syntactically a valid expression). `sort(items)` and `deduplicate(items)` work fine called **without** their optional `key` argument; `paginate(items, page, page_size)` takes no callable and always works. If you need custom filter/sort/group logic, do it with `if`/`for each` in a step body instead of reaching for `filter_`/`group_by`.

Only 3 stdlib functions are actually async and need to be called with `let x = <fn>(...)` inside a step (implicitly awaited by codegen) rather than assumed synchronous: `fetch_url`, `webhook`, `wait`. Every other stdlib function (including everything in `drift/safety`, `drift/data`, `drift/text`, `drift/observe`, and the rest of `drift/io`/`drift/notify`) is plain sync — calling it is a normal expression, nothing special.

Stubs vs real, per function (named bare, matching the actual call syntax — module names below are just which import line each comes from, not part of the call):
- **Real:** everything in `drift/io` (`read`, `write`, `fetch_url`, `load_pdf`, `load_csv`), all of `drift/safety`, all of `drift/data`, `webhook` (raises on 4xx/5xx), `now`/`wait`/`deadline` (from `drift/time`), `chunk`/`tokenize`/`similarity` (from `drift/text`).
- **Stub (logs/prints, no real backend):** `email`, `slack`, `push` (from `drift/notify`), all of `drift/observe` (`log`, `trace`, `metric`; `cost_report` wraps a real `CostTracker.summary()` when one exists, otherwise stubs too).
- **Stub by design (raises, doesn't silently no-op):** `embed` (from `drift/text`) raises `NotImplementedError`. `schedule` (from `drift/time`) stores to an in-process registry only — there's no daemon/cron loop, so nothing ever fires it (see the pipeline `schedule:` note in §12.1 — same underlying gap).

If a `.drift` program depends on a stub actually delivering (an email arriving, a metric landing in a real dashboard), say so — don't assume "no error" means "it happened."

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
**One failed item loses the whole batch, as written above.** `parallel` compiles to a plain `asyncio.gather` over all items — if any single item's intent call ultimately fails (e.g. `SchemaViolation` after exhausting its own retries), the exception propagates out of the whole `for each`, and every already-collected result in `results` is discarded along with it (never returned) — not just the failed item. If you're batch-processing and want the good results even when some items fail, wrap the per-item work in `attempt`/`recover` **inside** the parallel body, not around the whole `for each`:
```drift
for each email in emails parallel {
  attempt {
    let analysis = classify email as EmailAnalysis
    results.add(analysis)
  } recover from {
    any error -> respond "skipping {email}: classification failed"
  }
}
```
This makes each item's failure independent — a caught item is simply missing from `results` instead of taking every other item down with it.

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
drift new <name> [--model M]                    # scaffold a starter project
drift run <file> [--step S] [--agent A] [--pipeline P] [--input '<json>'] [--watch] [--trace] [--json]
drift check <file>                               # syntax-only validation
drift fmt <file> [--check] [--stdout]
drift transpile <file> [-o out.py]
drift schema <file> [--name S]                   # render schema block(s) as JSON Schema
drift lex <file> | drift parse <file>            # debug
drift mcp                                        # run as an MCP stdio server
```

`drift run` auto-loads `.env` from the file's directory tree, transpiles to a `.py` next to the source, runs the first agent's first step (or `--step`/`--agent`/`--pipeline` if specified). `--input` takes a JSON object mapped to the step's parameters by name — **except for `--pipeline` runs**, where `--input` is passed as a single positional value to the entry node instead; see §12 for the exact difference. `--input -` reads the JSON blob from stdin instead of the argv string, for large inputs or callers that don't want to shell-escape JSON into an argument.

**`drift run --json`** emits exactly one JSON object to stdout instead of the human banner/box-drawing output, for a caller with no human reading the terminal (e.g. a sandboxed execution wrapper). The shape matches `drift_run`'s MCP response below — `{ok, result, cost, outputs}` on success, `{ok: false, stage, kind, error, agent, step, cost, outputs}` on failure — plus a `stage` of `lex`/`parse`/`codegen`/`import`/`input`/`discover`/`run` on every failure (MCP's `drift_run` only sets `stage: "run"` failures' `kind`; a compile-time `--json` failure has `stage` but no `kind`). Exit code is `0` on `ok: true`, `1` otherwise. Cannot be combined with `--watch`.

**`drift schema <file>`** renders the file's `schema` block(s) as JSON Schema without running anything (no LLM calls, no budget spent) — printed as `{name: json_schema, ...}` for every schema declared, or a single `json_schema` object with `--name <schema>`. Useful for deriving an external tool's input/output schema from a Drift program mechanically instead of hand-writing a parallel schema.

**`drift mcp`** exposes Drift itself over MCP, for another coding agent (you, if invoked that way) to call instead of shelling out to the CLI: `drift_check(source)`, `drift_transpile(source)`, `drift_schema(source, name?)`, `drift_run(source, input?, pipeline?, agent?, step?)`. Prefer `drift_check` first when iterating — it's free (no LLM calls) and catches lex/parse/codegen errors before `drift_run` would spend budget. `drift_run`'s response is `{ok, result, cost, outputs}` on success, or `{ok: false, stage, kind, error, agent, step, cost, outputs}` on failure — `kind` is one of `budget`/`auth`/`business-logic`/`bug`, `agent`/`step` name where the failure occurred (may be `null` for a failure that isn't step-scoped, e.g. an import error), `cost` is `{total_cost, budget, currency, calls}` and reflects spend even when the run failed partway through, and `outputs` is the list of the agent's `respond`-statement lines (also present on failure, showing whatever printed before the error). All tools use `error` (not `message`) for failure text, consistently. If you're calling Drift through an MCP connection rather than the CLI directly, prefer these tools over shelling out to `drift run`.

`drift_run`'s optional `pipeline` argument names a `pipeline` declaration to run instead of an agent, mirroring `drift run --pipeline <name>` — `input` becomes the pipeline's initial input (any JSON value, not a per-parameter object) when `pipeline` is given, same as `--pipeline --input` on the CLI (see §12). Without `pipeline`, a source containing agents always runs an agent by default (even if it also declares a pipeline); a source with ONLY pipelines and no agents returns a `{ok: false, stage: "discover", ...}` error naming the available pipeline(s) rather than running anything — pass `pipeline` explicitly to run one. Cost/outputs tracking is agent-run-only; a pipeline run's response has no `cost`/`outputs` fields.

`drift_run`'s optional `agent`/`step` arguments select a specific agent/step by name, mirroring `drift run --agent`/`--step` — without `agent`, the first-declared agent runs (see `first_declared`'s rationale in §19's CLI behavior above); ignored when `pipeline` is given.

`drift_schema`'s optional `name` argument selects a single schema to render; without it, every schema declared in `source` is returned as `{ok: true, schemas: {name: json_schema, ...}}` (in source order). With `name`: `{ok: true, schema: json_schema}`. Failure stages are `lex`/`parse`/`codegen`/`discover` (no schema by that name, or none declared at all)/`import`.

**Cross-file `import` does not work through `drift_run`/`drift_check`/`drift_transpile`/`drift_schema`.** These MCP tools take raw source *text*, not a file path — there is no directory for a relative `import { X } from "./other.drift"` to resolve against, so any program using cross-file import fails with `stage: "import"` and a `ModuleNotFoundError`. (`drift run <file>` on disk, via the CLI, DOES resolve cross-file imports correctly — it auto-transpiles the dependency and adds the right directories to the Python import path.) If you're iterating on a multi-file program through the MCP tools, either inline everything into one `source` string (no `import`), or write the files to disk and shell out to `drift run <file>` instead of using `drift_run(source)`.

---

## 20. When uncertain, do this

1. Match the user's task to one of the section-17 patterns. Start from a known shape.
2. Use existing intent verbs (section 9) before defining new ones.
3. If a structured output is needed, declare a `schema` for it.
4. If you need confidence-gated branching, return `confident<T>` from the intent.
5. If a step might fail on a provider error, wrap it in `attempt`/`recover`.
6. Never emit Python — Drift is the target language. The runtime handles the rest.
7. After writing, mentally trace one execution: which step runs, which model, what gets returned. If you can't trace it, the user can't either.
