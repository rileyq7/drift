"""drift/notify — Sending alerts.

v0.2 stubs. `email`, `slack`, `webhook`, and `push` log to stdout by default
so example programs run without configuration. Real send paths are
opt-in via env vars (`DRIFT_NOTIFY_REAL=1`) or by overriding the functions.
"""
import os
import sys


def _is_real() -> bool:
    return os.environ.get("DRIFT_NOTIFY_REAL") == "1"


def email(to: str, subject: str, body: str) -> None:
    """Send an email. v0.2 stub — prints to stdout unless DRIFT_NOTIFY_REAL=1."""
    if _is_real():
        raise NotImplementedError(
            "Real email send requires configuring a backend. "
            "Override drift.notify.email or wire a tool decl."
        )
    print(f"  📧  to={to!r} subject={subject!r}\n      {body}", file=sys.stdout)


def slack(channel: str, message: str) -> None:
    if _is_real():
        raise NotImplementedError("Configure DRIFT_SLACK_WEBHOOK and override.")
    print(f"  💬  slack#{channel}: {message}")


async def webhook(url: str, payload: dict) -> None:
    import httpx
    async with httpx.AsyncClient() as client:
        await client.post(url, json=payload, timeout=10.0)


def push(title: str, body: str) -> None:
    """Push notification stub — prints to stdout."""
    print(f"  🔔  {title}: {body}")


__all__ = ["email", "slack", "webhook", "push"]
