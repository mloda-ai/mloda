"""CloseContext: the ambient context handed to Extender.close() on a MULTIPROCESSING worker.

Pins the current()/activate() scope and the shared close-time deadline extenders
closing in the same worker read via remaining().
"""

import contextlib
import time
from collections.abc import Generator
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Literal

CloseReason = Literal["stop", "error", "parent_gone"]

_current_close_context: ContextVar["CloseContext | None"] = ContextVar("_current_close_context", default=None)


@dataclass(frozen=True, kw_only=True)
class CloseContext:
    """Ambient context describing a worker's graceful close, shared across all extenders closing in it."""

    deadline: float
    reason: CloseReason
    run_id: str | None = None
    worker_index: int | None = None
    carrier: dict[str, str] | None = field(default=None, hash=False)
    tenant_id: str | None = None
    project_id: str | None = None
    principal: str | None = None

    def __post_init__(self) -> None:
        # Copy on ingest so an extender mutating the carrier never reaches the RunContext's dict.
        if self.carrier is not None:
            object.__setattr__(self, "carrier", dict(self.carrier))

    def remaining(self) -> float:
        """Seconds left of the shared close deadline, clamped to 0.0 once it has passed."""
        return max(0.0, self.deadline - time.monotonic())

    @classmethod
    def current(cls) -> "CloseContext | None":
        """Return the CloseContext active in the current activate() scope, else None.

        Thread-scoped: a contextvars variable, invisible to another thread unless it shares the copied context.
        """
        return _current_close_context.get()

    @contextlib.contextmanager
    def activate(self) -> Generator["CloseContext", None, None]:
        """Make this instance the current() context for the scope, restoring the previous one on exit."""
        token = _current_close_context.set(self)
        try:
            yield self
        finally:
            _current_close_context.reset(token)
