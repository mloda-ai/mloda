"""Reclaim of the test-local subclasses a suite builds, so none outlives its test in the class registry."""

from __future__ import annotations

import gc
import weakref
from collections.abc import Iterator
from typing import Any, TypeVar

T = TypeVar("T", bound=type)

_LOG: list[weakref.ref[Any]] = []


def track(cls: T) -> T:
    """Log a built class weakly and return it."""
    _LOG.append(weakref.ref(cls))
    return cls


def mark() -> int:
    """The current log position."""
    return len(_LOG)


def _alive_since(position: int) -> list[str]:
    alive = (ref() for ref in _LOG[position:])
    return [cls.__name__ for cls in alive if cls is not None]


def reclaim_since(position: int) -> list[str]:
    """Collect young-first until nothing tracked since the mark lives; return survivor names."""
    survivors = _alive_since(position)
    for generation in (0, 1, 2):
        if not survivors:
            break
        gc.collect(generation)
        survivors = _alive_since(position)
    del _LOG[position:]
    return survivors


def reclaimed_scope() -> Iterator[None]:
    """Fixture body: fail if a class tracked during the test survives it."""
    position = mark()
    yield
    survivors = reclaim_since(position)
    assert not survivors, f"suite-built classes left registered: {survivors}"
