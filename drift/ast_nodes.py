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
    state_block: list = field(default_factory=list)
    steps: list = field(default_factory=list)  # list of StepDecl


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
    """Model routing configuration."""
    default: str = ""
    prefer: str = ""
    fallback: str = ""
    never: str = ""
    upgrades: list = field(default_factory=list)


@dataclass
class ModelUpgrade:
    target_model: str = ""
    conditions: list = field(default_factory=list)


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
class ExprStmt:
    """An expression used as a statement (e.g., a function call)."""
    expr: 'Expression' = None


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
             ForEachStmt | MatchStmt | ExprStmt)
