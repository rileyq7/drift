"""drift/observe — Logging, tracing, metrics.

Stubs that print structured records to stdout. Production use should
swap in a real backend (OpenTelemetry, Honeycomb, etc.).
"""
import json
import sys
import time


def log(message: str, **fields) -> None:
    """Emit a structured log line as JSON."""
    record = {"ts": time.time(), "message": message}
    record.update(fields)
    print(json.dumps(record), file=sys.stdout)


def trace(name: str, **fields):
    """Context manager that emits start/end log records and timing."""
    return _Trace(name, fields)


def metric(name: str, value: float, **labels) -> None:
    record = {"ts": time.time(), "metric": name, "value": value, "labels": labels}
    print(json.dumps(record), file=sys.stdout)


def cost_report(agent) -> str:
    """Human-readable cost report for an agent — wraps CostTracker.summary()."""
    if not hasattr(agent, "cost_tracker"):
        return "(no cost tracker on this object)"
    return agent.cost_tracker.summary()


class _Trace:
    def __init__(self, name, fields):
        self.name = name
        self.fields = fields
        self.start = 0.0

    def __enter__(self):
        self.start = time.time()
        log(f"trace.start:{self.name}", **self.fields)
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed_ms = (time.time() - self.start) * 1000
        log(f"trace.end:{self.name}", elapsed_ms=elapsed_ms,
            error=str(exc) if exc else None, **self.fields)


__all__ = ["log", "trace", "metric", "cost_report"]
