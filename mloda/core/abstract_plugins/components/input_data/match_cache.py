"""Per-run memoization of match-time work, scoped by a context manager."""

from collections.abc import Callable, Hashable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, TypeVar

T = TypeVar("T")

_RUN_MATCH_CACHE: ContextVar[dict[Hashable, Any] | None] = ContextVar("_RUN_MATCH_CACHE", default=None)


@contextmanager
def run_match_cache() -> Iterator[None]:
    """Open a cache scope; a nested use reuses the outer scope."""
    if _RUN_MATCH_CACHE.get() is not None:
        yield
        return
    token = _RUN_MATCH_CACHE.set({})
    try:
        yield
    finally:
        _RUN_MATCH_CACHE.reset(token)


def run_cached(key: Hashable, compute: Callable[[], T]) -> T:
    """Compute once per key inside a scope; compute every time outside one."""
    cache = _RUN_MATCH_CACHE.get()
    if cache is None:
        return compute()
    if key not in cache:
        cache[key] = compute()
    result: T = cache[key]
    return result
