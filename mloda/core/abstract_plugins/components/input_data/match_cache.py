"""Per-run memoization of match-time work, scoped by a context manager."""

from collections.abc import Callable, Hashable
from contextlib import AbstractContextManager, nullcontext
from contextvars import ContextVar
from typing import Any, TypeVar

from mloda.core.abstract_plugins.components.var_scope import VarScope

T = TypeVar("T")

_RUN_MATCH_CACHE: ContextVar[dict[Hashable, Any] | None] = ContextVar("_RUN_MATCH_CACHE", default=None)


def run_match_cache() -> AbstractContextManager[Any]:
    """Open a cache scope; a nested use reuses the outer scope."""
    if _RUN_MATCH_CACHE.get() is not None:
        return nullcontext()
    return VarScope(_RUN_MATCH_CACHE, {})


def run_cached(key: Hashable, compute: Callable[[], T]) -> T:
    """Compute once per key inside a scope; compute every time outside one."""
    cache = _RUN_MATCH_CACHE.get()
    if cache is None:
        return compute()
    if key not in cache:
        cache[key] = compute()
    result: T = cache[key]
    return result
