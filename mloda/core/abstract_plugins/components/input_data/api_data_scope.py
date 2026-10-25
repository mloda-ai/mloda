"""Run-scoped access to one api_data set for the step that loads it; imports no FeatureGroup."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

_API_DATA_LOOKUP: ContextVar[tuple[str, Callable[[], Any]] | None] = ContextVar("mloda_api_data_lookup", default=None)


@dataclass(frozen=True)
class ApiDataRef:
    """Reference to one api_data set: its key and the column names frozen at prepare time."""

    key: str
    columns: tuple[str, ...]
    value_type: type | None = None


@contextmanager
def api_data_scope(lookup: Callable[[], Any], key: str) -> Iterator[None]:
    """Make lookup, bound to the set key, the reader of api_data inside the block."""
    token = _API_DATA_LOOKUP.set((key, lookup))
    try:
        yield
    finally:
        _API_DATA_LOOKUP.reset(token)


def read_api_data(key: str) -> Any:
    """The values of the set bound by the active scope."""
    lookup = _API_DATA_LOOKUP.get()
    if lookup is None:
        raise ValueError(
            f"api_data set '{key}' can only be read while its step runs; no api_data scope is active here."
        )
    bound_key, reader = lookup
    if key != bound_key:
        raise ValueError(f"api_data set '{key}' cannot be read here; this scope is bound to set {bound_key!r}.")
    return reader()
