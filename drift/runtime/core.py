"""
Drift Runtime Core — All runtime components in one module.

Components:
  Agent        — Base class for all Drift agents
  ModelRouter  — Multi-provider model dispatch with failover
  Budget       — Cost cap definition
  CostTracker  — Real-time cost tracking and enforcement
  Intent       — Translates intent verbs (classify, extract, etc.) into LLM calls
  Checkpoint   — Durable state serialization between steps
"""

import os
import json
import time
import asyncio
import dataclasses
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Type
from functools import wraps


# ─── Errors ────────────────────────────────────────────────────────────

class DriftError(Exception):
    """Base error for all Drift runtime errors."""
    pass

class BudgetExceeded(DriftError):
    """Raised when an agent exceeds its cost budget."""
    pass

class StepFailed(DriftError):
    """Raised when a step fails after all retries."""
    pass

class SchemaViolation(DriftError):
    """Raised when LLM output doesn't match expected schema."""
    pass

class ModelUnavailable(DriftError):
    """
    Raised when a model can't be reached right now (5xx, network, timeout)
    or when the router has exhausted all candidates. Recoverable by
    falling back to another model.
    """
    def __init__(self, message: str, model: str = None):
        super().__init__(message)
        self.model = model

class RateLimited(DriftError):
    """
    Raised on HTTP 429. Recoverable by waiting and retrying — usually
    with backoff. The current model is still viable.
    """
    def __init__(self, message: str, model: str = None, retry_after: float = None):
        super().__init__(message)
        self.model = model
        self.retry_after = retry_after

class AuthError(DriftError):
    """
    Raised on HTTP 401/403. NOT recoverable — the key is bad or missing
    permissions. Fail fast rather than retrying on every model.
    """
    pass


class Intent:
    """Namespace for intent verb constants."""
    CLASSIFY = "classify"
    EXTRACT = "extract"
    SUMMARIZE = "summarize"
    RATE = "rate"
    GENERATE = "generate"
    REWRITE = "rewrite"
    ANSWER = "answer"
    COMPARE = "compare"
    DECIDE = "decide"


# ─── Confident<T> ──────────────────────────────────────────────────────

class Confident:
    """Wraps an LLM output with a confidence score.

    Supports Drift's `is confident` / `is uncertain` branching: the runtime
    compares the score against the agent's `min_confidence` threshold to
    decide which branch to take.

        let result = classify doc as confident<Category>
        if result is confident { ... }       # confidence >= threshold
        otherwise if result is uncertain { ... }  # below threshold

    Subscripting at runtime (Confident[T]) is a no-op for codegen — the
    type parameter is documentation only; the runtime stores Any.
    """
    __slots__ = ("value", "confidence")

    def __init__(self, value, confidence: float):
        self.value = value
        try:
            c = float(confidence)
        except (TypeError, ValueError):
            c = 0.0
        # Clamp to [0, 1]. LLMs occasionally return 0.95% as 95.
        if c > 1.0:
            c = c / 100.0 if c <= 100.0 else 1.0
        self.confidence = max(0.0, min(1.0, c))

    def is_confident(self, threshold: float) -> bool:
        return self.confidence >= threshold

    def __repr__(self):
        return f"Confident(value={self.value!r}, confidence={self.confidence:.3f})"

    @classmethod
    def __class_getitem__(cls, item):
        # Make Confident[T] work as a type annotation. The element type is
        # only documentation at runtime — validation happens via the wrapped
        # value's schema (if any).
        return cls


# ─── Budget ────────────────────────────────────────────────────────────

@dataclass
class Budget:
    """Defines cost constraints for an agent run."""
    max_per_run: float = 10.0
    currency: str = "USD"

    @property
    def symbol(self):
        return {'GBP': '£', 'USD': '$', 'EUR': '€'}.get(self.currency, '$')


class CostTracker:
    """Tracks cumulative cost during a run and enforces budget."""

    def __init__(self, budget: Budget):
        self.budget = budget
        self.total_cost = 0.0
        self.call_log: list[dict] = []

    def pre_check(self, estimated_cost: float = 0.01):
        """Check if we have budget remaining before a call."""
        if self.total_cost + estimated_cost > self.budget.max_per_run:
            raise BudgetExceeded(
                f"Budget exceeded: {self.budget.symbol}{self.total_cost:.4f} spent "
                f"of {self.budget.symbol}{self.budget.max_per_run:.2f} limit"
            )

    def record(self, cost: float, model: str, tokens_in: int, tokens_out: int):
        """Record the cost of a completed call."""
        self.total_cost += cost
        self.call_log.append({
            'model': model,
            'cost': cost,
            'tokens_in': tokens_in,
            'tokens_out': tokens_out,
            'timestamp': time.time(),
            'cumulative_cost': self.total_cost,
        })

    @property
    def remaining(self) -> float:
        return max(0, self.budget.max_per_run - self.total_cost)

    def summary(self) -> str:
        """Human-readable cost summary."""
        s = self.budget.symbol
        lines = [
            f"╔══ Cost Report ═══════════════════════════════╗",
            f"║  Total: {s}{self.total_cost:.4f} / {s}{self.budget.max_per_run:.2f} budget     ║",
            f"║  Calls: {len(self.call_log):<38}║",
        ]
        if self.call_log:
            lines.append(f"║{'─' * 46}║")
            for i, call in enumerate(self.call_log, 1):
                model = call['model'][:20]
                cost_str = f"{s}{call['cost']:.4f}"
                toks = f"{call['tokens_in']}→{call['tokens_out']} tok"
                lines.append(f"║  {i}. {model:<20} {cost_str:<10} {toks:<10}║")
        lines.append(f"╚══════════════════════════════════════════════╝")
        return "\n".join(lines)


# ─── Model Router ──────────────────────────────────────────────────────

# Rough cost per 1K tokens (input/output) for routing decisions.
# Source: Anthropic pricing as of mid-2026 (USD).
MODEL_COSTS = {
    'claude-haiku':   {'input': 0.001,   'output': 0.005},
    'claude-sonnet':  {'input': 0.003,   'output': 0.015},
    'claude-opus':    {'input': 0.015,   'output': 0.075},
    'claude-fable':   {'input': 0.005,   'output': 0.025},
    'gpt-4o':         {'input': 0.005,   'output': 0.015},
    'gpt-4o-mini':    {'input': 0.00015, 'output': 0.0006},
}

# Map logical names to current API model IDs.
# This registry is the single point of update when providers release new
# versions — agent code keeps using the logical name.
MODEL_REGISTRY = {
    'claude-haiku':  'claude-haiku-4-5-20251001',
    'claude-sonnet': 'claude-sonnet-4-6',
    'claude-opus':   'claude-opus-4-7',
    'claude-fable':  'claude-fable-5',
    'gpt-4o':        'gpt-4o',
    'gpt-4o-mini':   'gpt-4o-mini',
}


@dataclass
class ModelRouter:
    """Routes model calls to the best available provider.

    Upgrade rules (§4 model block form):
      Each rule is {"target": "<model>", "conditions": [{"kind": ..., "value": ...}]}
      Conditions, when ALL true, escalate the selected model to `target`.
      Supported kinds:
        - "tokens_gt"     : input_tokens > value
        - "step_is"       : current step name == value
        - "confidence_lt" : LAST call's confidence < value  (best-effort;
                            requires the runtime to track confidence across
                            calls — v0.2 stores it on the agent's last
                            Confident result and reads it here)
    """
    default: str = "claude-sonnet"
    prefer: str = ""
    fallback: list[str] = field(default_factory=list)
    never: list[str] = field(default_factory=list)
    upgrades: list[dict] = field(default_factory=list)
    _unavailable: set = field(default_factory=set)
    _last_confidence: float = 1.0  # updated after each Confident result

    def __post_init__(self):
        if isinstance(self.fallback, str):
            self.fallback = [self.fallback] if self.fallback else []

    def candidates(self) -> list[str]:
        """All viable models in preference order."""
        ordered = [self.prefer or self.default] + list(self.fallback)
        seen = set()
        out = []
        for m in ordered:
            if not m or m in seen:
                continue
            seen.add(m)
            if m in self._unavailable or m in self.never:
                continue
            out.append(m)
        return out

    def _apply_upgrades(self, base: str, context: dict) -> str:
        """Evaluate upgrade rules. First matching rule wins."""
        for rule in self.upgrades:
            target = rule.get("target")
            if not target or target in self._unavailable or target in self.never:
                continue
            if all(self._cond_holds(c, context) for c in rule.get("conditions", [])):
                return target
        return base

    def _cond_holds(self, cond: dict, context: dict) -> bool:
        kind, value = cond.get("kind"), cond.get("value")
        if kind == "tokens_gt":
            return context.get("input_tokens", 0) > value
        if kind == "step_is":
            return context.get("step") == value
        if kind == "confidence_lt":
            return self._last_confidence < value
        return False

    def select(self, budget_remaining: float = float('inf'),
               context: dict = None) -> str:
        """Select the best available model given current constraints."""
        cands = self.candidates()
        if not cands:
            raise ModelUnavailable("No models available — all candidates are marked unavailable or banned")
        # If budget is tight, prefer cheaper models
        if budget_remaining < 0.10:
            cands.sort(key=lambda m: MODEL_COSTS.get(m, {}).get('input', 999))
        base = cands[0]
        if self.upgrades and context is not None:
            return self._apply_upgrades(base, context)
        return base

    def record_confidence(self, value: float):
        """Called after a Confident-returning intent so upgrade rules can
        condition on `confidence < N` for the *next* call."""
        try:
            self._last_confidence = float(value)
        except (TypeError, ValueError):
            pass

    def mark_unavailable(self, model: str):
        if model:
            self._unavailable.add(model)

    def reset_availability(self):
        """Clear transient unavailability marks — call at step start."""
        self._unavailable.clear()

    def api_model_id(self, logical_name: str) -> str:
        return MODEL_REGISTRY.get(logical_name, logical_name)

    def estimate_cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        costs = MODEL_COSTS.get(model, {'input': 0.003, 'output': 0.015})
        return (tokens_in / 1000 * costs['input']) + (tokens_out / 1000 * costs['output'])


# ─── Checkpoint ────────────────────────────────────────────────────────

class Checkpoint:
    """Saves step outputs for durability. In-memory for MVP, swappable to SQLite/Redis."""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.data: dict[str, Any] = {}

    def save(self, step_name: str, result: Any):
        key = f"{self.agent_name}.{step_name}"
        self.data[key] = {
            'result': self._serialize(result),
            'timestamp': time.time(),
        }

    def load(self, step_name: str) -> Any:
        key = f"{self.agent_name}.{step_name}"
        if key in self.data:
            return self.data[key]['result']
        return None

    def _serialize(self, obj):
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        return obj


# ─── Intent System ─────────────────────────────────────────────────────

# System prompts for each intent verb
INTENT_PROMPTS = {
    'classify': (
        "You are a precise classifier. Classify the given input according to "
        "the specified categories or schema. Return ONLY valid JSON matching "
        "the schema. No preamble, no markdown."
    ),
    'extract': (
        "You are a precise data extractor. Extract the requested fields from "
        "the given input. Return ONLY valid JSON matching the schema. "
        "No preamble, no markdown."
    ),
    'summarize': (
        "You are a concise summarizer. Summarize the given input. "
        "Be precise and informative. Return ONLY the summary text."
    ),
    'rate': (
        "You are an evaluator. Rate/score the given input against the provided "
        "criteria. Return ONLY valid JSON matching the schema. "
        "No preamble, no markdown."
    ),
    'generate': (
        "You are a content generator. Generate content matching the description "
        "and schema provided. Return ONLY valid JSON matching the schema. "
        "No preamble, no markdown."
    ),
    'rewrite': (
        "You are a rewriter. Rewrite the given text in the specified style or "
        "format. Return ONLY the rewritten text."
    ),
    'answer': (
        "Answer the question using only the provided context. Be precise. "
        "If the context doesn't contain enough information, say so."
    ),
    'compare': (
        "Compare the given items. Return ONLY valid JSON matching the schema. "
        "No preamble, no markdown."
    ),
    'decide': (
        "Make a decision based on the given context. Return ONLY valid JSON. "
        "No preamble, no markdown."
    ),
    'match': (
        "Match the input against the given criteria. Return ONLY valid JSON "
        "matching the schema. No preamble, no markdown."
    ),
    'translate': (
        "Translate the input to the target language. Return ONLY the "
        "translated text — no preamble, no quotes, no markdown."
    ),
}


def build_intent_prompt(verb: str, input_data: Any, **kwargs) -> str:
    """Build a complete prompt for an intent expression."""
    system = INTENT_PROMPTS.get(verb, INTENT_PROMPTS['classify'])
    parts = [f"INPUT:\n{_format_input(input_data)}"]

    if 'source' in kwargs and kwargs['source'] is not None:
        parts.append(f"\nSOURCE DOCUMENT:\n{_format_input(kwargs['source'])}")

    if 'criteria' in kwargs and kwargs['criteria'] is not None:
        parts.append(f"\nCRITERIA:\n{_format_input(kwargs['criteria'])}")

    if 'context' in kwargs and kwargs['context'] is not None:
        parts.append(f"\nCONTEXT:\n{_format_input(kwargs['context'])}")

    if 'count' in kwargs:
        unit = kwargs.get('unit', 'items')
        parts.append(f"\nReturn exactly {kwargs['count']} {unit}.")

    if 'target' in kwargs and kwargs['target'] is not None:
        parts.append(f"\nTARGET: {kwargs['target']}")

    if 'factors' in kwargs and kwargs['factors'] is not None:
        parts.append(f"\nCONSIDER THESE FACTORS: {', '.join(str(f) for f in kwargs['factors'])}")

    if 'output_schema' in kwargs and kwargs['output_schema'] is not None:
        schema = kwargs['output_schema']
        if isinstance(schema, type) and schema is Confident:
            parts.append(
                "\nOUTPUT SCHEMA (return valid JSON):\n"
                '{"value": <your answer>, "confidence": <float 0-1>}'
            )
        elif isinstance(schema, type) and dataclasses.is_dataclass(schema):
            schema_info = _describe_schema(schema)
            parts.append(f"\nOUTPUT SCHEMA (return valid JSON matching this):\n{schema_info}")
        elif isinstance(schema, str):
            parts.append(f"\nRETURN TYPE: {schema}")

    return system, "\n".join(parts)


def _format_input(data: Any) -> str:
    if isinstance(data, str):
        return data
    if isinstance(data, list):
        return "\n".join(str(item) for item in data)
    if dataclasses.is_dataclass(data):
        return json.dumps(dataclasses.asdict(data), indent=2)
    return str(data)


def _describe_schema(cls) -> str:
    """Generate a JSON schema description from a dataclass."""
    if not dataclasses.is_dataclass(cls):
        return str(cls)

    fields_desc = {}
    for f in dataclasses.fields(cls):
        type_str = str(f.type).replace('typing.', '')
        fields_desc[f.name] = type_str

    return json.dumps(fields_desc, indent=2)


def parse_llm_response(text: str, output_schema=None) -> Any:
    """Parse LLM response text into the expected schema."""
    if output_schema is None or output_schema == str:
        return text.strip()

    # Try to parse as JSON
    cleaned = text.strip()
    if cleaned.startswith('```'):
        lines = cleaned.split('\n')
        cleaned = '\n'.join(lines[1:-1] if lines[-1].strip() == '```' else lines[1:])
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        raise SchemaViolation(f"LLM output is not valid JSON: {text[:200]}")

    # Confident<T>: expect {value, confidence}. The inner value isn't validated
    # against a specific type at this layer — Drift's intent codegen passes the
    # bare Confident class, not Confident[SomeSchema], so we accept whatever
    # the LLM gave us for `value` and just enforce the wrapper shape.
    if isinstance(output_schema, type) and output_schema is Confident:
        if not isinstance(data, dict):
            raise SchemaViolation(
                f"Expected JSON object for Confident, got {type(data).__name__}"
            )
        if 'value' not in data or 'confidence' not in data:
            raise SchemaViolation(
                f"Confident output must have 'value' and 'confidence' keys, got {list(data)}"
            )
        return Confident(data['value'], data['confidence'])

    # Instantiate the dataclass if applicable
    if isinstance(output_schema, type) and dataclasses.is_dataclass(output_schema):
        if not isinstance(data, dict):
            raise SchemaViolation(
                f"Expected JSON object for {output_schema.__name__}, got {type(data).__name__}: {data!r:.200}"
            )
        valid_fields = {f.name for f in dataclasses.fields(output_schema)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        try:
            instance = output_schema(**filtered)
        except TypeError as e:
            raise SchemaViolation(f"Cannot instantiate {output_schema.__name__}: {e}")
        # Run generated constraint validation (between, etc.) — raises
        # SchemaViolation so step_decorator can retry with a stricter prompt.
        if hasattr(instance, 'validate'):
            try:
                instance.validate()
            except AssertionError as e:
                raise SchemaViolation(
                    f"{output_schema.__name__} failed constraint validation: {e}"
                )
        return instance

    return data


# ─── LLM Provider ─────────────────────────────────────────────────────

class MockProvider:
    """
    Mock LLM provider for testing. Generates plausible structured output
    based on the schema without making real API calls.
    """

    async def call(self, model: str, system: str, prompt: str,
                   output_schema=None) -> tuple[str, int, int]:
        """Returns (response_text, tokens_in, tokens_out)"""
        tokens_in = len(prompt.split()) * 2  # rough estimate
        await asyncio.sleep(0.1)  # simulate latency

        if output_schema is Confident:
            # Confident<T>: return a deterministic, high-confidence wrapper so
            # the `is confident` branch fires by default in tests. Tests that
            # need to exercise the uncertain branch can monkeypatch the agent's
            # min_confidence above 0.88.
            response = json.dumps({"value": "mock_value", "confidence": 0.88})
        elif output_schema is not None and dataclasses.is_dataclass(output_schema):
            # Generate mock data matching the schema
            mock_data = {}
            for f in dataclasses.fields(output_schema):
                mock_data[f.name] = self._mock_field(f)
            response = json.dumps(mock_data, indent=2)
        else:
            response = f"[Mock {model} response to: {prompt[:80]}...]"

        tokens_out = len(response.split()) * 2
        return response, tokens_in, tokens_out

    def _mock_field(self, f):
        type_str = str(f.type)
        name = f.name.lower()

        # Literals must be checked first — picking a value outside the allowed
        # set would fail schema validation, so we always return the first one.
        if 'Literal' in type_str:
            import re
            match = re.search(r"['\"]([^'\"]+)['\"]", type_str)
            return match.group(1) if match else "value"

        # Optional must be checked before list/str, since "Optional[list[str]]"
        # contains both substrings. For tests we just return None.
        if 'Optional' in type_str or 'None' in type_str:
            return None

        if 'list' in type_str:
            # Recursively pick mock values matching the element type.
            inner_match = __import__('re').search(r'list\[(.+)\]', type_str)
            inner = inner_match.group(1) if inner_match else 'str'
            if 'gap' in name:
                return ["No significant gaps identified"]
            if 'str' in inner or "'" in inner or '"' in inner:
                return ["criterion_1", "criterion_2", "criterion_3"]
            if 'float' in inner or 'int' in inner:
                return [1.0, 2.0, 3.0]
            return []

        if 'bool' in type_str:
            return True

        if 'float' in type_str or 'int' in type_str:
            if 'confidence' in name or 'probability' in name:
                return 0.88
            if 'score' in name or 'rating' in name:
                return 82.0
            return 75.0

        if 'str' in type_str:
            if 'name' in name:
                return "TechCo Ltd"
            if 'title' in name:
                return "Smart Grants R&D"
            if 'summary' in name or 'reasoning' in name:
                return f"Analysis complete for {name}"
            return f"sample_{f.name}"

        return f"mock_{f.name}"


class AnthropicProvider:
    """Real LLM provider using the Anthropic API."""

    def __init__(self):
        self.api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise DriftError(
                "ANTHROPIC_API_KEY not set. Use MockProvider or set the env var."
            )

    async def call(self, model: str, system: str, prompt: str,
                   output_schema=None) -> tuple[str, int, int]:
        import httpx

        api_model = MODEL_REGISTRY.get(model, model)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": api_model,
                        "max_tokens": 2048,
                        "system": system,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=60.0,
                )
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            raise ModelUnavailable(
                f"Network error reaching Anthropic for {model}: {e}", model=model
            )

        status = response.status_code
        if status == 200:
            data = response.json()
            text = data['content'][0]['text']
            usage = data.get('usage', {})
            return text, usage.get('input_tokens', 0), usage.get('output_tokens', 0)

        body = response.text[:200]
        if status in (401, 403):
            # Auth errors aren't recoverable by retrying or falling back.
            raise AuthError(f"Anthropic auth failed ({status}): {body}")
        if status == 429:
            retry_after = response.headers.get('retry-after')
            try:
                retry_after = float(retry_after) if retry_after else None
            except ValueError:
                retry_after = None
            raise RateLimited(
                f"Rate limited by Anthropic ({status}): {body}",
                model=model,
                retry_after=retry_after,
            )
        if status == 404:
            # Model ID is wrong — treat as that specific model being unavailable
            # so the router can try a fallback.
            raise ModelUnavailable(
                f"Anthropic returned 404 for model {api_model!r}: {body}", model=model
            )
        # 5xx, anything else — provider-side problem, try a fallback.
        raise ModelUnavailable(
            f"Anthropic API error ({status}) for {model}: {body}", model=model
        )


def get_provider():
    """Get the best available provider.

    Honors DRIFT_USE_MOCK=1 to force the mock provider even when an API key
    is present — useful for offline development and tests.
    """
    if os.environ.get('DRIFT_USE_MOCK') == '1':
        print("  ℹ  Using mock provider (DRIFT_USE_MOCK=1)")
        return MockProvider()
    if os.environ.get('ANTHROPIC_API_KEY'):
        try:
            return AnthropicProvider()
        except Exception:
            pass
    print("  ℹ  Using mock provider (set ANTHROPIC_API_KEY for real LLM calls)")
    return MockProvider()


# ─── Step Decorator ────────────────────────────────────────────────────

def step_decorator(output=None, modifier=""):
    """Decorator that wraps agent steps with cost tracking, checkpointing, and retries.

    Recovery policy:
      - SchemaViolation: retry up to max_retries (LLM might just give better JSON).
      - ModelUnavailable: mark the failed model unavailable and retry; the router
        picks the next candidate. If candidates exhaust, give up.
      - RateLimited: wait (respecting retry_after) and retry on the same model.
      - AuthError: fail fast — no retry helps.
      - BudgetExceeded: fail fast — retrying would only burn more budget.
    """
    def decorator(func):
        func._drift_step = True
        func._drift_output = output
        func._drift_modifier = modifier

        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            step_name = func.__name__
            max_retries = 3

            # Clear transient unavailability at the start of each step. A 503
            # on a previous step shouldn't degrade this one.
            if hasattr(self, 'model') and self.model is not None:
                self.model.reset_availability()
            # Remember the current step name so router upgrade rules
            # (`step is final_recommendation`) can fire.
            self._current_step = step_name

            last_error = None
            for attempt in range(max_retries):
                try:
                    result = await func(self, *args, **kwargs)
                    if output and dataclasses.is_dataclass(output) and isinstance(result, output):
                        if hasattr(result, 'validate'):
                            result.validate()
                    return result

                except SchemaViolation as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        print(f"  ⟳  Schema violation in {step_name}, retrying ({attempt + 1}/{max_retries})")
                        continue
                    raise StepFailed(
                        f"Step '{step_name}' failed after {max_retries} attempts: {e}"
                    )

                except ModelUnavailable as e:
                    last_error = e
                    # Mark the specific model that failed, not whatever
                    # select() would return next.
                    self.model.mark_unavailable(getattr(e, 'model', None))
                    if attempt < max_retries - 1 and self.model.candidates():
                        next_model = self.model.candidates()[0]
                        print(f"  ⟳  {getattr(e, 'model', '?')} unavailable, falling back to {next_model}")
                        continue
                    raise StepFailed(
                        f"Step '{step_name}' failed: no models available ({e})"
                    )

                except RateLimited as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        wait = e.retry_after if e.retry_after else min(2 ** attempt, 30)
                        print(f"  ⟳  Rate limited on {e.model}, waiting {wait:.1f}s")
                        await asyncio.sleep(wait)
                        continue
                    raise StepFailed(
                        f"Step '{step_name}' failed after {max_retries} rate-limited attempts"
                    )

                except (AuthError, BudgetExceeded):
                    # Both are fail-fast: retrying won't make a bad key good
                    # or refund spent budget.
                    raise

            raise StepFailed(
                f"Step '{step_name}' failed after {max_retries} attempts: {last_error}"
            )

        return wrapper
    return decorator


# ─── Agent Base Class ──────────────────────────────────────────────────

class Agent:
    """
    Base class for all Drift agents.

    Provides:
      - Model routing (self.model)
      - Cost tracking (self.cost_tracker)
      - Checkpointing (self.checkpoint)
      - Intent execution (self.intent())
      - Output handling (self.output())
    """

    def __init__(self, name: str, model: ModelRouter = None,
                 budget: Budget = None, min_confidence: float = 0.85):
        self.name = name
        self.model = model or ModelRouter()
        self.budget = budget or Budget()
        self.min_confidence = min_confidence
        self.cost_tracker = CostTracker(self.budget)
        self.checkpoint = Checkpoint(name)
        self._provider = get_provider()
        self._outputs: list[str] = []

    async def intent(self, verb: str, input_data: Any = None,
                     output_schema=None, **kwargs) -> Any:
        """
        Execute an intent expression.

        This is the core runtime method. Every intent verb in Drift
        (classify, extract, summarize, etc.) becomes a call to this method.
        """
        # Build prompt first so we have a token estimate for the router.
        system_prompt, user_prompt = build_intent_prompt(
            verb, input_data, output_schema=output_schema, **kwargs
        )
        # Approximate token count: ~4 chars per token is a reasonable rule
        # of thumb across recent models.
        estimated_in = (len(system_prompt) + len(user_prompt)) // 4

        # Select model. Pass routing context so upgrade rules can fire.
        context = {
            "input_tokens": estimated_in,
            "step": getattr(self, "_current_step", None),
        }
        model_name = self.model.select(
            self.cost_tracker.remaining, context=context
        )

        # Pre-check budget
        estimated_cost = self.model.estimate_cost(model_name, estimated_in, 500)
        self.cost_tracker.pre_check(estimated_cost)

        # Call LLM
        print(f"  ▸  {verb}() via {model_name}")
        response_text, tokens_in, tokens_out = await self._provider.call(
            model_name, system_prompt, user_prompt, output_schema
        )

        # Track cost
        actual_cost = self.model.estimate_cost(model_name, tokens_in, tokens_out)
        self.cost_tracker.record(actual_cost, model_name, tokens_in, tokens_out)

        # Parse response
        result = parse_llm_response(response_text, output_schema)
        # Feed confidence back to the router so `confidence < N` upgrade
        # rules can fire on the NEXT call (we can't retroactively upgrade
        # the call that just finished).
        if isinstance(result, Confident):
            self.model.record_confidence(result.confidence)
        return result

    def output(self, text: str):
        """Handle respond statements."""
        self._outputs.append(str(text))
        print(f"  ◆  {text}")


# ─── Runner ────────────────────────────────────────────────────────────

async def run_agent(agent_class: type, step_name: str = None,
                    inputs: dict = None) -> Any:
    """
    Run a Drift agent from the command line.

    Creates an instance, finds the target step, calls it with inputs,
    and prints the cost report.
    """
    agent = agent_class()
    inputs = inputs or {}

    print(f"\n{'═' * 50}")
    print(f"  Drift — Running {agent.name}")
    print(f"  Budget: {agent.budget.symbol}{agent.budget.max_per_run:.2f}")
    print(f"  Model: {agent.model.default}")
    print(f"{'═' * 50}\n")

    # Find the step to run
    if step_name:
        method = getattr(agent, step_name, None)
        if method is None:
            raise DriftError(f"Step '{step_name}' not found on agent '{agent.name}'")
    else:
        # Find first step
        for attr_name in dir(agent):
            attr = getattr(agent, attr_name)
            if callable(attr) and hasattr(attr, '__wrapped__'):
                method = attr
                step_name = attr_name
                break
        else:
            # Just run the first method that isn't __init__ or inherited
            steps = [name for name in dir(agent)
                     if not name.startswith('_')
                     and callable(getattr(agent, name))
                     and name not in ('intent', 'output')]
            if steps:
                step_name = steps[0]
                method = getattr(agent, step_name)
            else:
                raise DriftError(f"No steps found on agent '{agent.name}'")

    print(f"  Step: {step_name}({', '.join(f'{k}={v!r}' for k, v in inputs.items())})")
    print()

    try:
        start = time.time()
        result = await method(**inputs)
        elapsed = time.time() - start

        print()
        print(agent.cost_tracker.summary())
        print(f"\n  Time: {elapsed:.2f}s")

        if result is not None:
            print(f"\n  ── Result ──")
            if dataclasses.is_dataclass(result):
                print(f"  {json.dumps(asdict(result), indent=2, default=str)}")
            else:
                print(f"  {result}")

        return result

    except BudgetExceeded as e:
        print(f"\n  ✗ Budget exceeded: {e}")
        print(agent.cost_tracker.summary())
        raise

    except StepFailed as e:
        print(f"\n  ✗ Step failed: {e}")
        raise
