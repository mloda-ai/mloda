"""HookContext: the delivery seam handed to Extender implementations.

Pins the ambient current()/activate() scope, row_count's __len__ gating,
and instrument's timing/status bookkeeping around a wrapped call.
"""

import contextlib
import functools
import time
from collections.abc import Callable, Generator
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from mloda.core.abstract_plugins.components.utils import safe_field
from mloda.core.abstract_plugins.function_extender import ExtenderHook

_current_hook_context: ContextVar["HookContext | None"] = ContextVar("_current_hook_context", default=None)


@dataclass(kw_only=True)
class HookContext:
    """Ambient, per-call context describing an Extender hook invocation."""

    hook: ExtenderHook
    feature_group_class: str
    feature_group_version: str
    plugin_version: str | None
    feature_names: tuple[str, ...]
    input_features: frozenset[str] | None
    compute_framework_name: str
    rows_in: int | None = None
    rows_out: int | None = None
    duration_seconds: float | None = None
    status: str | None = None
    run_id: str | None = None
    data_access_identity: str | None = None
    tenant_id: str | None = None
    principal: str | None = None

    @staticmethod
    def row_count(data: Any) -> int | None:
        """Return len(data) when data supports __len__, else None. Never calls len() otherwise."""
        if hasattr(data, "__len__"):
            return len(data)
        return None

    @classmethod
    def current(cls) -> "HookContext | None":
        """Return the HookContext active in the current activate() scope, else None."""
        return _current_hook_context.get()

    @contextlib.contextmanager
    def activate(self) -> Generator["HookContext", None, None]:
        """Make this instance the current() context for the scope, restoring the previous one on exit."""
        token = _current_hook_context.set(self)
        try:
            yield self
        finally:
            _current_hook_context.reset(token)


def instrument(
    context: HookContext,
    func: Callable[..., Any],
    row_count: Callable[[Any], int | None] = HookContext.row_count,
) -> Callable[..., Any]:
    """Wrap func, updating context.status/duration_seconds/rows_out around the call."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        context.status = "error"
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        finally:
            context.duration_seconds = time.perf_counter() - start
        context.status = "success"
        context.rows_out = safe_field(lambda: row_count(result), None, field="rows_out")
        return result

    if hasattr(func, "__self__"):
        wrapper.__self__ = func.__self__  # type: ignore[attr-defined]

    return wrapper
