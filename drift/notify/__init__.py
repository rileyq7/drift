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
    """Send an email. v0.2 stub — prints to stdout.

    With DRIFT_NOTIFY_REAL=1 but no configured backend there's nothing to send
    through, so we warn once and still print, rather than crashing the run.
    """
    if _is_real():
        print(
            "  ⚠  DRIFT_NOTIFY_REAL=1 but no email backend is configured "
            "(override drift.notify.email or wire a tool decl); logging instead.",
            file=sys.stderr,
        )
    print(f"  📧  to={to!r} subject={subject!r}\n      {body}", file=sys.stdout)


def slack(channel: str, message: str) -> None:
    if _is_real():
        print(
            "  ⚠  DRIFT_NOTIFY_REAL=1 but no Slack backend is configured "
            "(use drift.notify.webhook or override); logging instead.",
            file=sys.stderr,
        )
    print(f"  💬  slack#{channel}: {message}")


async def webhook(url: str, payload: dict) -> None:
    import httpx
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=payload, timeout=10.0)
        # A 4xx/5xx must not read as a silent success.
        resp.raise_for_status()


def push(title: str, body: str) -> None:
    """Push notification stub — prints to stdout."""
    print(f"  🔔  {title}: {body}")


__all__ = ["email", "slack", "webhook", "push"]
