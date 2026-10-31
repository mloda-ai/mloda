"""Test helpers for subclass-tree walks and reclaiming transient subclasses."""

import gc
import weakref
from collections.abc import Callable
from typing import Any

import pytest


def count_root_walks(monkeypatch: pytest.MonkeyPatch, root: type[Any]) -> list[int]:
    """Install a `__subclasses__` spy on `root`; the returned list gains one entry per call made on `root` itself."""
    walks: list[int] = []

    def spy(cls: type) -> list[type]:
        if cls is root:
            walks.append(1)
        return type.__subclasses__(cls)

    monkeypatch.setattr(root, "__subclasses__", classmethod(spy))
    return walks


def collect_until(done: Callable[[], bool]) -> bool:
    """Escalate gc generations until `done()` holds; return whether it did.

    `done` must not hold what it waits on between calls, or no collection can reclaim it.
    """
    for generation in (0, 1, 2):
        gc.collect(generation)
        if done():
            return True
    return False


def collect_until_dead(reference: weakref.ref[type[Any]]) -> bool:
    """Return whether gc reclaims the class behind `reference`."""
    return collect_until(lambda: reference() is None)
