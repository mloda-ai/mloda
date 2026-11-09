"""Start a mloda run with every step-scoped context var clean; the outer values return on exit."""

from contextvars import ContextVar, Token
from typing import Any

from mloda.core.abstract_plugins.components.credential_access import _READABLE
from mloda.core.abstract_plugins.components.credential_grant import _GRANT_HOLDER, _READ_RECORDERS
from mloda.core.abstract_plugins.components.failed_rows import _COLLECTOR
from mloda.core.abstract_plugins.components.input_data.api_data_scope import _API_DATA_LOOKUP
from mloda.core.abstract_plugins.compute_framework import _current_compute_framework
from mloda.core.abstract_plugins.hook_context import _current_hook_context
from mloda.core.abstract_plugins.input_data_load_marker import current_input_data_load_marker

# _in_gate_scope is left out on purpose: an outer gate stays closed over a nested run.
_CLEAN: tuple[tuple[ContextVar[Any], Any], ...] = (
    (_READABLE, None),
    (_READ_RECORDERS, ()),
    (_GRANT_HOLDER, None),
    (_current_hook_context, None),
    (_current_compute_framework, None),
    (current_input_data_load_marker, None),
    (_COLLECTOR, None),
    (_API_DATA_LOOKUP, None),
)


class _RunBoundary:
    """Reset the step-scoped vars for a block (a class, not @contextmanager, like _ReadableScope)."""

    def __init__(self) -> None:
        self._tokens: list[list[Token[Any]]] = []

    def __enter__(self) -> None:
        self._tokens.append([var.set(clean) for var, clean in _CLEAN])

    def __exit__(self, *exc_info: object) -> None:
        tokens = self._tokens.pop()
        for (var, _), token in zip(reversed(_CLEAN), reversed(tokens), strict=True):
            var.reset(token)


def run_boundary() -> _RunBoundary:
    """Start a mloda run with the step-scoped context vars clean."""
    return _RunBoundary()
