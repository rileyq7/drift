"""
Drift AST Nodes — The tree structure that represents a parsed Drift program.

The pipeline: source → tokens → [PARSER] → AST → codegen → Python

Each node type here maps to a construct in the Drift language.
The code generator walks this tree and emits Python source code.
"""

from dataclasses import dataclass, field
from typing import Optional


# ─── Top-Level Declarations ───────────────────────────────────────────

@dataclass
class Program:
    """Root node: a .drift file is a list of declarations."""
    declarations: list = field(default_factory=list)


@dataclass
class ImportDecl:
    """import X[, Y] from "./other.drift"  — names imported from another file.

    For v0.2 the codegen emits a plain `from <basename>_drift import X, Y`.
    Dependencies must be transpiled before the importer; the CLI handles this
    automatically when running `drift run`/`drift transpile` on a tree.
    """
    names: list = field(default_factory=list)  # list of str
    source_path: str = ""                       # path string from the `from` clause


@dataclass
class ConfigDecl:
    """config { name: "...", version: "..." }"""
    entries: dict = field(default_factory=dict)


@dataclass
class SchemaDecl:
    """schema FitScore { field: type [constraints] }"""
    name: str = ""
    fields: list = field(default_factory=list)  # list of SchemaField


@dataclass
class SchemaField:
    name: str = ""
    type_expr: 'TypeExpr | None' = None
    optional: bool = False
    constraints: list = field(default_factory=list)  # list of Constraint


@dataclass
class AgentDecl:
    """agent GrantChecker { model: ..., budget: ..., steps... }"""
    name: str = ""
    model_config: 'ModelConfig | None' = None
    budget_config: 'BudgetConfig | None' = None
    quality_config: 'QualityConfig | None' = None
    memory_config: 'MemoryConfig | None' = None
    state_block: list = field(default_factory=list)  # list of StateField
    steps: list = field(default_factory=list)        # list of StepDecl


@dataclass
class MemoryConfig:
    """Memory configuration. Two forms:

      memory: dendric("persona_name")
          → backend="dendric", persona="persona_name". The codegen wires
            this to DendricStore (with mock fallback when DATABASE_URL
            is unset).

      memory { store: "sqlite://...", recall strategy: "semantic", ... }
          → legacy block form, backend="sqlite" (default).
    """
    backend: str = "sqlite"             # "sqlite" | "dendric"
    persona: str = ""                   # only used when backend=="dendric"
    store: str = "sqlite://:memory:"    # only used when backend=="sqlite"
    recall_strategy: str = "recent"     # "semantic" | "recent" | "relevant" | "all"
    max_recall: int = 20
    decay_enabled: bool = False


@dataclass
class StateField:
    """A single line inside an agent's state block: name: type = default."""
    name: str = ""
    type_expr: 'TypeExpr | None' = None
    default: 'Expression | None' = None


@dataclass
class PipelineDecl:
    """A pipeline: a DAG of step calls across one or more agents.

    Top-level declaration. See §2.5 and §12.1 of the spec.

    `use_agents` lists agent type names that participate (`use GrantChecker`).
    `edges` lists each `from -> to` flow with its operator.
    `failure_handlers` map step names to recovery actions.
    """
    name: str = ""
    budget_config: 'BudgetConfig | None' = None
    timeout_seconds: float = 0.0
    schedule: str = ""
    use_agents: list = field(default_factory=list)         # list of str
    edges: list = field(default_factory=list)              # list of PipelineEdge
    failure_handlers: dict = field(default_factory=dict)   # step_name -> action
    budget_handler: str = ""                               # action on budget exceeded
    inline_steps: list = field(default_factory=list)       # list of StepDecl (defined inline)


@dataclass
class PipelineEdge:
    """One arrow in a pipeline:  from_node OP to_node

    OP is "->" (seq), "=>" (parallel fan-out), "~>" (conditional), "|>" (stream).
    Nodes are either "step" or "AgentType.step".
    """
    from_node: str = ""
    to_node: str = ""
    op: str = "->"


@dataclass
class ToolDecl:
    """A tool declaration. Three forms — `kind` distinguishes them.

    kind == "mcp":     `tool name from mcp "url"`
    kind == "python":  `tool name from python "module.path:fn"`
    kind == "rest":    `tool name { endpoint: ..., auth: ..., action ... }`
    """
    name: str = ""
    kind: str = ""           # "mcp" | "python" | "rest"
    source: str = ""         # URL (mcp) or "module:fn" (python)
    endpoint: str = ""       # REST base URL
    auth_env: str = ""       # `auth: env("VAR_NAME")` — env var name for auth header
    auth_literal: str = ""   # `auth: "token"` — static dev token, used as-is
    actions: list = field(default_factory=list)  # list of ToolAction


@dataclass
class ToolAction:
    """One action inside a REST tool.

      action lookup(company_number: string) -> CompanyProfile {
        GET "/company/{company_number}"
      }
    """
    name: str = ""
    params: list = field(default_factory=list)        # list of Param
    return_type: 'TypeExpr | None' = None
    method: str = "GET"                                # GET, POST, ...
    path: str = ""                                     # URL template with {var}


@dataclass
class VerbDecl:
    """define verb name { pattern: ..., prompt: ..., output: ..., temperature: ... }

    Top-level declaration. Registers a custom intent verb that can be used
    anywhere a built-in verb appears (classify, extract, summarize, ...).
    """
    name: str = ""
    pattern: str = ""           # cosmetic for v0.2 — docs only
    prompt: str = ""            # system prompt for the LLM
    output: 'TypeExpr | None' = None
    temperature: float = 0.0    # 0 = unspecified; default model temp


@dataclass
class StepDecl:
    """step check(doc: Document) -> FitScore { ... }"""
    name: str = ""
    params: list = field(default_factory=list)    # list of Param
    return_type: 'TypeExpr | None' = None
    body: list = field(default_factory=list)       # list of Statement
    modifier: str = ""  # "cached", "parallel", "manual", "silent", ""


@dataclass
class Param:
    name: str = ""
    type_expr: 'TypeExpr | None' = None


# ─── Configuration Blocks ─────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Model routing configuration.

    Routing modes:
      "default"      — single model, optionally with fallback list
      "prefer"       — preferred model + fallback chain
      "stream_then"  — temporal model routing: fire `stream_model` for an
                       instant bridge response while `then_model` does the
                       real reasoning, supersede the bridge when ready.
                       Used by the voice stack but the language feature
                       is provider-agnostic (works for any two models).
    """
    mode: str = "default"     # "default" | "prefer" | "stream_then"
    default: str = ""
    prefer: str = ""
    fallback: str = ""        # comma-allowed string OR list, normalized in parser
    fallback_list: list = field(default_factory=list)
    never: str = ""
    never_list: list = field(default_factory=list)
    upgrades: list = field(default_factory=list)  # list of ModelUpgrade
    stream_model: str = ""    # fast bridge model
    then_model: str = ""      # slow reasoning model


@dataclass
class ModelUpgrade:
    """upgrade to "X" when { conditions... }"""
    target_model: str = ""
    conditions: list = field(default_factory=list)  # list of UpgradeCondition


@dataclass
class UpgradeCondition:
    """One condition inside an upgrade rule.

    Forms supported in v0.2:
      - `confidence < 0.8`           (kind="confidence_lt", value=0.8)  [parsed, unenforced]
      - `input_tokens > N`           (kind="tokens_gt", value=N)
      - `step is <name>`             (kind="step_is", value=name)
    """
    kind: str = ""
    value: 'float | int | str' = 0


@dataclass
class BudgetConfig:
    amount: str = ""       # e.g. "£5"
    currency_sym: str = ""  # £, $, €
    value: float = 0.0
    per: str = "run"       # "run", "day", "company"


@dataclass
class QualityConfig:
    min_confidence: float = 0.85


# ─── Type Expressions ─────────────────────────────────────────────────

@dataclass
class TypeExpr:
    """Base type expression."""
    name: str = ""  # "string", "number", "bool", "document", or a schema name


@dataclass
class ListType(TypeExpr):
    element_type: 'TypeExpr | None' = None


@dataclass
class MapType(TypeExpr):
    key_type: 'TypeExpr | None' = None
    value_type: 'TypeExpr | None' = None


@dataclass
class ConfidentType(TypeExpr):
    inner_type: 'TypeExpr | None' = None


@dataclass
class EnumType(TypeExpr):
    """one of "a", "b", "c" """
    values: list = field(default_factory=list)


# ─── Constraints ───────────────────────────────────────────────────────

@dataclass
class BetweenConstraint:
    low: float = 0.0
    high: float = 0.0


# ─── Statements ────────────────────────────────────────────────────────

@dataclass
class LetStmt:
    name: str = ""
    value: 'Expression' = None


@dataclass
class ReturnStmt:
    value: 'Expression' = None


@dataclass
class RespondStmt:
    value: 'Expression' = None


@dataclass
class IfStmt:
    condition: 'Expression' = None
    body: list = field(default_factory=list)
    otherwise_body: list = field(default_factory=list)
    otherwise_if: 'IfStmt | None' = None


@dataclass
class ForEachStmt:
    var_name: str = ""
    iterable: 'Expression' = None
    body: list = field(default_factory=list)
    parallel: bool = False


@dataclass
class MatchStmt:
    target: 'Expression' = None
    arms: list = field(default_factory=list)  # list of MatchArm


@dataclass
class MatchArm:
    pattern: 'Expression' = None  # StringLit or Ident for "any other"
    body: list = field(default_factory=list)
    is_default: bool = False


@dataclass
class DejaVuStmt:
    """deja_vu match on <context_expr> { "pattern_name" -> { body } ... }

    Semantically: pass context_expr to memory.deja_vu_check(); when the
    archive trigger fires, dispatch to the matching arm. The match's
    pattern_type is compared substring-wise against each arm's pattern
    string (v1 classification — see DendricStore.DejaVuMatch.matches)."""
    context_expr: 'Expression' = None
    arms: list = field(default_factory=list)  # list of DejaVuArm


@dataclass
class DejaVuArm:
    pattern: str = ""
    body: list = field(default_factory=list)
    is_default: bool = False


@dataclass
class ForgetStmt:
    """forget memories tagged X / older than Nd / where temp < X

    `mode` is one of: "by_tag" (tag expr), "by_age" (older_than_days int),
    "by_temp" (below_temp float). All three route to the adapter's
    forget() method — see DendricStore.forget."""
    mode: str = "by_temp"
    tag: 'Expression' = None
    older_than_days: int = 0
    below_temp: float = 0.0


@dataclass
class ExprStmt:
    """An expression used as a statement (e.g., a function call)."""
    expr: 'Expression' = None


@dataclass
class AttemptStmt:
    """attempt { body } recover from { ErrorType -> handler ... }

    Each arm matches an exception class by name. The arm body is a list of
    statements; a single-statement arm can be inline via `-> statement`.
    Special arm patterns:
      - `any error` matches any DriftError (and is the default fallthrough).
      - `retry` (as a single-statement body) re-runs the attempt block.
      - `fail with "..."` raises StepFailed with the given message.
    """
    body: list = field(default_factory=list)
    arms: list = field(default_factory=list)  # list of RecoverArm
    max_retries: int = 3


@dataclass
class RecoverArm:
    error_type: str = ""        # exception class name, or "any" for default
    body: list = field(default_factory=list)
    is_default: bool = False


@dataclass
class RetryStmt:
    """`retry` inside a recover arm — re-run the attempt block."""
    pass


@dataclass
class RecallStmt:
    """recall [similar] <description> [for <key>] — wraps memory.recall().

    Always an expression (returns a list). Parsed at expression position.
    `description` is a real Expression (StringLit with interpolation,
    Ident for a bare variable reference, or a StringLit built from
    collected free-form words) — NOT a raw string. A bare identifier
    (`recall question for "advice"`, LLM.md's own documented example)
    must evaluate `question`'s runtime value, not the literal text
    "question".
    """
    description: 'Expression' = None
    key: 'Expression | None' = None


@dataclass
class RememberStmt:
    """remember <expr> [tagged <tag>[, <tag>, ...]] — wraps memory.remember().
    `tagged` accepts one or more comma-separated tags (LLM.md's own
    documented example uses two: `tagged "advice", "user_123"`)."""
    value: 'Expression' = None
    tags: list = field(default_factory=list)


@dataclass
class FailStmt:
    """`fail with "<message>"` — raise StepFailed."""
    message: 'Expression' = None


# ─── Expressions ───────────────────────────────────────────────────────

@dataclass
class IntentExpr:
    """classify doc as Category — the core Drift primitive."""
    verb: str = ""         # "classify", "extract", "summarize", etc.
    input_expr: 'Expression' = None
    clauses: dict = field(default_factory=dict)
    # Possible clause keys: "as" (type), "from" (source), "in" (count),
    # "against" (criteria), "to" (target), "using" (context),
    # "considering" (factors)


@dataclass
class FnCall:
    name: str = ""
    args: list = field(default_factory=list)
    # For method calls like results.add(x), target is set
    target: 'Expression | None' = None


@dataclass
class FieldAccess:
    target: 'Expression' = None
    field_name: str = ""


@dataclass
class BinOp:
    op: str = ""
    left: 'Expression' = None
    right: 'Expression' = None


@dataclass
class UnaryOp:
    op: str = ""
    operand: 'Expression' = None


@dataclass
class Ident:
    name: str = ""


@dataclass
class StringLit:
    value: str = ""
    has_interpolation: bool = False


@dataclass
class NumberLit:
    value: float = 0.0


@dataclass
class CurrencyLit:
    symbol: str = ""
    value: float = 0.0


@dataclass
class DurationLit:
    value: float = 0.0
    unit: str = ""  # s, m, h, d


@dataclass
class BoolLit:
    value: bool = False


@dataclass
class ListLit:
    elements: list = field(default_factory=list)


@dataclass
class SchemaConstructor:
    """FitScore { field: value, ... }"""
    type_name: str = ""
    fields: dict = field(default_factory=dict)


@dataclass
class PipeExpr:
    """expr |> fn"""
    input_expr: 'Expression' = None
    operations: list = field(default_factory=list)


# Type alias for documentation
Expression = (IntentExpr | FnCall | FieldAccess | BinOp | Ident |
              StringLit | NumberLit | CurrencyLit | DurationLit |
              BoolLit | ListLit | SchemaConstructor | PipeExpr | UnaryOp)

Statement = (LetStmt | ReturnStmt | RespondStmt | IfStmt |
             ForEachStmt | MatchStmt | DejaVuStmt | ForgetStmt | ExprStmt)
