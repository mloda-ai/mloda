"""Test helpers for subclass-tree walks and reclaiming transient subclasses."""

from abc import ABC, ABCMeta
import gc
import weakref
from collections.abc import Callable
from typing import Any

import pytest

from mloda.core.abstract_plugins.plugin_registry.plugin_registry import _PLUGIN_BASE_TYPES


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


def spy_on_plugin_subclasschecks(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Wrap ABCMeta.__subclasscheck__; the returned list collects calls whose cls is or derives from a plugin base."""
    original = ABCMeta.__subclasscheck__
    counted: list[str] = []

    def spy(cls: ABCMeta, subclass: type[Any]) -> bool:
        if any(base in cls.__mro__ for base in _PLUGIN_BASE_TYPES):
            counted.append(cls.__qualname__)
        return original(cls, subclass)

    monkeypatch.setattr(ABCMeta, "__subclasscheck__", spy)
    return counted


def invalidate_abc_caches() -> None:
    """Bump ABCMeta's invalidation counter so every negative cache misses on its next check."""

    class _Throwaway(ABC):
        pass

    class _Registered:
        pass

    _Throwaway.register(_Registered)
