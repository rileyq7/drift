"""drift/safety — Content sanitization and rate limits.

v0.2: simple regex-based PII filters. Production users should swap in
a real DLP library — these are first-pass scrubbers, not bulletproof.
"""
import re
import time
from collections import defaultdict


# Quick patterns — cover the obvious cases.
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)\d{3}[\s-]?\d{4}\b")
# UK national insurance number; US SSN; etc.
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CC_RE = re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")


def redact_pii(text: str) -> str:
    """Replace email, phone, SSN, and credit-card-shaped strings with placeholders."""
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _PHONE_RE.sub("[PHONE]", text)
    text = _SSN_RE.sub("[SSN]", text)
    text = _CC_RE.sub("[CC]", text)
    return text


def check_content(text: str, banned_patterns: list[str] = None) -> bool:
    """Return True if text contains none of the banned patterns."""
    banned_patterns = banned_patterns or []
    return not any(re.search(p, text) for p in banned_patterns)


def sanitize(text: str, max_length: int = 4000) -> str:
    """Trim text to a max length and strip control characters."""
    cleaned = "".join(ch for ch in text if ch >= " " or ch in "\n\t")
    return cleaned[:max_length]


class _RateLimiter:
    """In-memory sliding window rate limiter, keyed by an arbitrary string."""
    def __init__(self):
        self._buckets: dict[str, list[float]] = defaultdict(list)

    def allow(self, key: str, max_per_window: int, window_seconds: float) -> bool:
        now = time.time()
        bucket = self._buckets[key]
        # Drop expired entries
        cutoff = now - window_seconds
        bucket[:] = [t for t in bucket if t >= cutoff]
        if len(bucket) >= max_per_window:
            return False
        bucket.append(now)
        return True


_default_limiter = _RateLimiter()


def rate_limit(key: str, max_per_window: int, window_seconds: float) -> bool:
    """Return True if the key is allowed; False if the budget is exhausted."""
    return _default_limiter.allow(key, max_per_window, window_seconds)


__all__ = ["redact_pii", "check_content", "sanitize", "rate_limit"]
