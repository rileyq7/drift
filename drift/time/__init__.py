"""drift/time — Time helpers.

`schedule` is intentionally a stub — scheduling is a deployment-tier concern,
not a runtime one. Pipelines declare `schedule: "every Monday at 9am"` as
metadata; an external scheduler (cron, GitHub Actions, etc.) is expected to
read that and invoke the pipeline.
"""
import asyncio as _asyncio
import time as _time
from datetime import datetime as _datetime


def now() -> _datetime:
    return _datetime.now()


async def wait(seconds: float) -> None:
    await _asyncio.sleep(seconds)


def deadline(seconds: float) -> float:
    """Return a wall-clock time `seconds` from now."""
    return _time.time() + seconds


def schedule(cron_or_phrase: str) -> None:
    """Declare a schedule. v0.2 stores the string in a global registry.
    Deployment tooling reads this to set up the actual cron."""
    SCHEDULES.append(cron_or_phrase)


SCHEDULES: list[str] = []


__all__ = ["now", "wait", "deadline", "schedule", "SCHEDULES"]
