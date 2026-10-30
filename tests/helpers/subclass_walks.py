"""Subclass-tree test helpers: a spy that counts walks from a root, and a young-first reclaim of transient classes."""

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
    """Collect generation 0, then 1, then 2, stopping as soon as `done()` holds; return whether it held.

    A class defined inside a test sits in a reference cycle and stays in `__subclasses__()` until cyclic GC runs.
    It is almost always still in generation 0, so the young collection is usually the only one paid for.
    """
    for generation in (0, 1, 2):
        gc.collect(generation)
        if done():
            return True
    return False


def collect_until_dead(reference: weakref.ref[type[Any]]) -> bool:
    """Collect young-first until the class behind `reference` is gone; return whether it died."""
    return collect_until(lambda: reference() is None)
