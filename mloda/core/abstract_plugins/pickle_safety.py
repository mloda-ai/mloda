"""Trial-pickle-and-warn-once helpers for an Extender managing an injected sink that must
survive MULTIPROCESSING worker dispatch. See docs/docs/chapter1/extender.md.

mloda/core/runtime/validate_multiprocessing_link.py has its own picklability check (used to
raise at plan-time), using the same three-exception filter as here, kept as a separately
defined constant on purpose to keep the two modules decoupled.
"""

from __future__ import annotations

import pickle  # nosec B403
import threading
from collections.abc import Callable
from typing import Any


_UNPICKLABLE_ERRORS = (pickle.PicklingError, AttributeError, TypeError)


def pickle_failure_reason(value: Any) -> str | None:
    """None if value pickles cleanly, else the caught exception's type name."""
    try:
        pickle.dumps(value)
    except _UNPICKLABLE_ERRORS as exc:
        return type(exc).__name__
    return None


def is_picklable(value: Any) -> bool:
    return pickle_failure_reason(value) is None


class WarnOncePerInstance:
    """Thread-safe double-checked-locking guard: attempts `emit` at most once per instance.

    Marks itself fired before calling `emit`, so a raising `emit` is never retried. A copy
    (pickle, copy.copy, copy.deepcopy) always starts fresh and unwarned via __reduce__, since
    a raw lock can't pickle and each copy must decide independently whether to warn. Not intended
    to be subclassed; a subclass instance safely becomes a plain WarnOncePerInstance on any copy.
    """

    def __init__(self) -> None:
        self._warned = False
        self._lock = threading.RLock()

    def warn_once(self, emit: Callable[[], None]) -> None:
        if self._warned:
            return
        with self._lock:
            if not self._warned:
                self._warned = True
                emit()

    def __reduce__(self) -> tuple[type[WarnOncePerInstance], tuple[()]]:
        return (WarnOncePerInstance, ())
