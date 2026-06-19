"""
Drift Runtime — The smart layer that generated code calls into.

This is where the actual intelligence lives:
  - ModelRouter: multi-provider dispatch with failover
  - Budget/CostTracker: per-run cost enforcement
  - Intent: translates intent verbs into LLM calls
  - Checkpoint: durable state between steps
  - Agent: base class that wires everything together

The transpiler generates code that calls this library.
The library handles model calls, cost tracking, retries, and validation.
"""

from .core import (
    Agent,
    step_decorator,
    Budget,
    CostTracker,
    ModelRouter,
    Intent,
    Checkpoint,
    run_agent,
    DriftError,
    BudgetExceeded,
    StepFailed,
    SchemaViolation,
    ModelUnavailable,
    RateLimited,
    AuthError,
)

__all__ = [
    'Agent',
    'step_decorator',
    'Budget',
    'CostTracker',
    'ModelRouter',
    'Intent',
    'Checkpoint',
    'run_agent',
    'DriftError',
    'BudgetExceeded',
    'StepFailed',
    'SchemaViolation',
    'ModelUnavailable',
    'RateLimited',
    'AuthError',
]
