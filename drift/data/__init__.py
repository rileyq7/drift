"""drift/data — Functional collection helpers used in pipeline bodies."""
from typing import Callable, TypeVar

T = TypeVar("T")
K = TypeVar("K")


def filter_(items: list[T], predicate: Callable[[T], bool]) -> list[T]:
    """Return items satisfying predicate."""
    return [i for i in items if predicate(i)]


def sort(items: list[T], key: Callable[[T], K] = None, descending: bool = False) -> list[T]:
    return sorted(items, key=key, reverse=descending)


def group_by(items: list[T], key: Callable[[T], K]) -> dict[K, list[T]]:
    out: dict[K, list[T]] = {}
    for item in items:
        out.setdefault(key(item), []).append(item)
    return out


def deduplicate(items: list[T], key: Callable[[T], K] = None) -> list[T]:
    """Preserves order; first occurrence wins."""
    seen: set = set()
    out: list[T] = []
    for item in items:
        k = key(item) if key else item
        if k not in seen:
            seen.add(k)
            out.append(item)
    return out


def paginate(items: list[T], page: int, page_size: int) -> list[T]:
    start = (page - 1) * page_size
    return items[start:start + page_size]


# Exposing as `filter` would shadow the builtin in the imported namespace.
# Drift codegen uses the bare name; we export both for ergonomics.
filter = filter_  # noqa: A001  intentional


__all__ = ["filter_", "filter", "sort", "group_by", "deduplicate", "paginate"]
