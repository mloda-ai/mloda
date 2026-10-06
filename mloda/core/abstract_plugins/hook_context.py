"""HookContext: the delivery seam handed to Extender implementations.

Pins the ambient current()/activate() scope, row_count's __len__ gating,
and instrument's timing/status bookkeeping around a wrapped call.
"""

import contextlib
import functools
import threading
import time
from collections.abc import Callable, Generator, Mapping
from contextvars import ContextVar
from dataclasses import FrozenInstanceError, dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID

from mloda.core.abstract_plugins.components.declared_attributes import scalar_attributes
from mloda.core.abstract_plugins.components.read_only_dict import _frozen_dict
from mloda.core.abstract_plugins.components.utils import safe_field
from mloda.core.abstract_plugins.function_extender import ExtenderHook

if TYPE_CHECKING:
    from mloda.core.abstract_plugins.components.input_data.base_input_data import BaseInputData
    from mloda.core.abstract_plugins.components.link import AsOfJoinConfig

_current_hook_context: ContextVar["HookContext | None"] = ContextVar("_current_hook_context", default=None)

OutputSchema = tuple[tuple[str, str | None], ...]

_WRITABLE_FIELDS = frozenset({"rows_out", "output_schema", "duration_seconds", "status"})

_gate_scopes = 0
_gate_scopes_lock = threading.Lock()
_result_attributes_lock = threading.Lock()


@contextlib.contextmanager
def input_data_load_gate_scope() -> Generator[None, None, None]:
    """Count one active INPUT_DATA_LOAD gate calculation process-wide, exception-safe."""
    global _gate_scopes
    with _gate_scopes_lock:
        _gate_scopes += 1
    try:
        yield
    finally:
        with _gate_scopes_lock:
            _gate_scopes -= 1


def input_data_load_gate_scopes_active() -> int:
    with _gate_scopes_lock:
        return _gate_scopes


@dataclass(kw_only=True)
class HookContext:
    """Ambient, per-call context describing an Extender hook invocation."""

    hook: ExtenderHook
    feature_group_class: str | None
    feature_group_version: str | None
    plugin_version: str | None = None
    feature_names: tuple[str, ...] = ()
    specialized_from: tuple[str, ...] = ()
    input_features: frozenset[str] | None = None
    input_feature_edges: dict[str, tuple[str, ...]] | None = None
    compute_framework_name: str | None
    rows_in: int | None = None
    rows_out: int | None = None
    output_schema: OutputSchema | None = None
    duration_seconds: float | None = None
    status: str | None = None
    run_id: str | None = None
    plan_id: str | None = None
    structure_hash: str | None = None
    step_uuid: UUID | None = None
    data_access_identity: str | None = None
    data_access_identity_is_fallback: bool | None = None
    tenant_id: str | None = None
    project_id: str | None = None
    principal: str | None = None
    carrier: dict[str, str] | None = None
    worker_index: int | None = None
    data_access_format: str | None = None
    data_access_dataset_version: str | None = None
    join_type: str | None = None
    join_keys: tuple[str, ...] | None = None
    asof_config: "AsOfJoinConfig | None" = None
    join_left_feature_group: str | None = None
    join_right_feature_group: str | None = None
    plan_feature_count: int | None = None
    plan_node_count: int | None = None
    plan_depth: int | None = None
    declared_attributes: dict[str, str | int | float | bool] | None = None
    reader_class: "type[BaseInputData] | None" = None
    result_attributes: dict[str, str | int | float | bool] | None = None

    def __post_init__(self) -> None:
        if self.result_attributes is not None:
            self.result_attributes = _frozen_dict(self.result_attributes)
        if self.declared_attributes is not None:
            self.declared_attributes = _frozen_dict(self.declared_attributes)
        # Copy on ingest so a hook mutating the carrier or input_feature_edges never reaches the caller's dict.
        if self.carrier is not None:
            self.carrier = _frozen_dict(self.carrier)
        if self.input_feature_edges is not None:
            self.input_feature_edges = _frozen_dict(self.input_feature_edges)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if self.__dict__.get("_sealed") and name not in _WRITABLE_FIELDS:
            raise FrozenInstanceError(f"cannot assign to field {name!r}")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if self.__dict__.get("_sealed"):
            raise FrozenInstanceError(f"cannot delete field {name!r}")
        object.__delattr__(self, name)

    @staticmethod
    def row_count(data: Any) -> int | None:
        """Return len(data) when the TYPE declares __len__; a dict counts its first column's rows
        (a scalar str/bytes or a nested dict first value is not a column).
        """
        if isinstance(data, dict):
            if not data:
                return 0
            first = next(iter(data.values()))
            if isinstance(first, (str, bytes, dict)):
                return None
            return HookContext.row_count(first)
        if callable(getattr(type(data), "__len__", None)):
            return len(data)
        return None

    @classmethod
    def current(cls) -> "HookContext | None":
        """Return the HookContext active in the current activate() scope, else None.

        Thread-scoped: a contextvars variable, invisible to another thread unless it shares the copied context.
        """
        return _current_hook_context.get()

    @classmethod
    def publish_result_attributes(cls, attributes: Mapping[Any, Any]) -> None:
        """Merge validated scalar attributes into the current context's result_attributes; no-op without one."""
        validated = scalar_attributes(attributes, "result_attributes", "must be a Mapping")
        current = cls.current()
        if current is None:
            return
        with _result_attributes_lock:
            merged = _frozen_dict({**(current.result_attributes or {}), **validated})
            object.__setattr__(current, "result_attributes", merged)

    @contextlib.contextmanager
    def activate(self) -> Generator["HookContext", None, None]:
        """Make this instance the current() context for the scope, restoring the previous one on exit."""
        token = _current_hook_context.set(self)
        try:
            yield self
        finally:
            _current_hook_context.reset(token)


def _no_schema(data: Any) -> OutputSchema | None:
    """output_schema stand-in for hooks whose return value carries no schema semantics."""
    return None


def _no_rows(data: Any) -> int | None:
    """row_count stand-in for hooks whose return value carries no row semantics."""
    return None


def instrument(
    context: HookContext,
    func: Callable[..., Any],
    row_count: Callable[[Any], int | None] = HookContext.row_count,
    output_schema: Callable[[Any], OutputSchema | None] = _no_schema,
) -> Callable[..., Any]:
    """Wrap func, updating context.status/duration_seconds/rows_out/output_schema around the call."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        context.rows_out = None
        context.output_schema = None
        object.__setattr__(context, "result_attributes", None)
        succeeded = False
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            succeeded = True
        finally:
            context.duration_seconds = time.perf_counter() - start
            context.status = "success" if succeeded else "error"
        context.rows_out = safe_field(lambda: row_count(result), None)
        context.output_schema = safe_field(lambda: output_schema(result), None)
        return result

    if hasattr(func, "__self__"):
        wrapper.__self__ = func.__self__  # type: ignore[attr-defined]

    return wrapper
