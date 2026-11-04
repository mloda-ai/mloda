"""Reclaim of the test-local subclasses a suite builds, so none outlives its test in the class registry."""

from __future__ import annotations

import gc
import weakref
from collections.abc import Iterator
from typing import Any, TypeVar

T = TypeVar("T", bound=type)

_LOG: list[weakref.ref[Any]] = []
_SURVIVORS: weakref.WeakSet[type] = weakref.WeakSet()


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


def _remember_survivors(position: int) -> None:
    _SURVIVORS.update(cls for cls in (ref() for ref in _LOG[position:]) if cls is not None)


def reclaim_since(position: int) -> list[str]:
    """Collect young-first; return the names of survivors logged since the mark, keeping them logged.

    Survivors of earlier scopes are collected too once released, but only those since the mark are reported.
    """
    survivors = _alive_since(position)
    for generation in (0, 1, 2):
        if not survivors and not _SURVIVORS:
            break
        gc.collect(generation)
        survivors = _alive_since(position)
    if survivors:
        _remember_survivors(position)
    _LOG[position:] = [ref for ref in _LOG[position:] if ref() is not None]
    return survivors


def reclaimed_scope() -> Iterator[None]:
    """Fixture body: fail if a class tracked during the test survives it."""
    position = mark()
    yield
    survivors = reclaim_since(position)
    assert not survivors, (
        f"suite-built classes left registered: {survivors}; one may be pinned by a failing test's traceback "
        "or held by the host, which is unsupported"
    )
