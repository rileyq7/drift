# Gotchas

Common mistakes when writing Drift. If something looks like it should work but doesn't, you're probably here.

## Don't use `else`

```drift
-- WRONG
if x { return a } else { return b }

-- RIGHT
if x { return a } otherwise { return b }
```

Also: `otherwise if` (not `else if`). No `elif`.

## Don't write `recover on X`

```drift
-- WRONG (doesn't exist)
attempt { ... } recover on rate limited { retry }

-- RIGHT
attempt { ... } recover from {
  RateLimited    -> retry
  BudgetExceeded -> fail "out of budget"
  any error      -> respond "fallback"
}
```

Each arm is `<ErrorType> -> <body>`. Error types are PascalCase exception class names.

## There is no `escalate`

```drift
-- WRONG (parses as four bare identifiers, becomes broken Python)
escalate to human review

-- RIGHT
fail "low confidence — escalate"
-- or
return Decision { needs_human: true, ... }
```

## `confident<T>` and plain schemas with confidence both work

Either form is fine. Pick by where you want the threshold:

```drift
-- confident<T>: threshold comes from agent's quality: or min_confidence
let scored = rate input as confident<Decision>
if scored is confident { return scored.value }

-- plain schema: threshold inline
schema Decision { ..., confidence: number between 0 and 1 }
let result = rate input as Decision
if result.confidence < 0.7 { fail "uncertain" }
```

## Always include `as <Type>` on intent verbs

```drift
-- WRONG — codegen has no schema to validate against
let summary = summarize document

-- RIGHT
let summary = summarize document as string
```

Without `as`, the LLM call returns raw text and the runtime can't enforce shape.

## `confident<T>` only wraps intent results

```drift
-- WRONG — confidence comes from the LLM, not literals
let x: confident<int> = 5

-- RIGHT
let scored = rate input against rubric as confident<Score>
```

## State doesn't persist between runs

```drift
agent Counter {
  state { count: int = 0 }
  step bump() -> int { count = count + 1; return count }
}
```

This **always returns 1**. Use `memory:` for cross-run persistence. `state` is for within-run scratch.

## Memory shorthand vs block — pick one

```drift
-- WRONG — can't have both
memory: dendric("x")
memory { store: "sqlite" }

-- RIGHT
memory: dendric("x")
-- or
memory { store: "sqlite" }
```

## PascalCase vs snake_case is enforced

```drift
-- WRONG — agent names must be PascalCase
agent my_agent { ... }

-- WRONG — step names must be snake_case
step CheckEligibility(...) { ... }

-- RIGHT
agent MyAgent { step check_eligibility(...) { ... } }
```

## Don't put spaces inside type parameters

```drift
-- Works but ugly; `drift fmt` will fix it
emails: list < string >

-- Canonical
emails: list<string>
```

## Don't put space between function/step name and `(`

```drift
-- WRONG (parses, but `drift fmt` would rewrite)
step greet (name: string) -> string { ... }
let r = MyTool.call (x)

-- RIGHT
step greet(name: string) -> string { ... }
let r = MyTool.call(x)
```

## Don't invent intent verbs

If you write `analyze foo as Bar` without declaring `analyze`, you get a parse error. Either use an existing verb (`classify`, `extract`, `summarize`, `rate`, `generate`, `rewrite`, `answer`, `compare`, `decide`, `translate`, `match`) or declare your own with `define verb`.

## `match` is two things

```drift
-- Intent verb: match X against Y as T
let result = match candidate against criteria as MatchResult

-- Statement: match X { case -> body }
match priority {
  "urgent" -> { respond "now" }
  _        -> { respond "later" }
}
```

The parser disambiguates by lookahead: if `against` appears before `{`, it's an intent. Otherwise, a statement.

## Currency literals are scoped to budgets

```drift
-- WRONG — $0.10 isn't a general number
let cost = $0.10

-- RIGHT
budget: $0.10 per run
```

## Imports are file-relative, not URL-based

```drift
-- WRONG
import { Foo } from "https://example.com/schemas.drift"

-- RIGHT
import { Foo } from "./schemas.drift"
```

## Model names route by prefix — no quotes for provider

```drift
-- WRONG — there is no "openai:" namespace
model: "openai:gpt-4o"

-- RIGHT — model name alone; routing picks provider
model: "gpt-4o"
model: "openai/gpt-4o"   -- if you really want explicit
```

`gpt-*`, `o1`, `o3`, `o4`, `openai/*` → OpenAI. `claude-*`, `anthropic/*` → Anthropic. Anything else → mock provider (with a banner).

## Mock provider is silent only without keys

```drift
-- This will hit mock if no key is set, with a banner. That's fine for dev.
-- But if you have OPENAI_API_KEY set and you pick "claude-sonnet" with no
-- ANTHROPIC_API_KEY, the runtime falls back to OpenAI (which won't recognize
-- the model) → 404. The CLI banner shows "anthropic + openai (auto-routed
-- by model)" only when BOTH keys are set.
```

## `forget` predicates are limited

```drift
-- Supported
forget memories tagged "user_123"
forget memories older than 30d
forget memories where temp < 0.2

-- Not supported (would need a custom Python tool)
forget memories matching some_lambda
```

## Block comments must balance

```drift
{- this is {- nested -} -}    -- ok
{- this is broken             -- LexError: unterminated block comment
```

## Generic types have no comma in `list<>`

```drift
-- WRONG
list<string,>

-- RIGHT
list<string>
dict<string, int>     -- comma only between multiple params
```
