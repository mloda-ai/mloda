"""Shared registry-isolation helper (#845) over FeatureGroup, BaseTransformer, Implementation, ComputeFramework, Extender.

A subclass defined inside a test sits in a reference cycle, so it stays in ``root.__subclasses__()`` until
cyclic GC runs and a later test on the same worker trips over it. A creation log replaces per-test tree walks.
"""

from __future__ import annotations

import gc
import sys
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Generic, TypeVar

from mloda.core.abstract_plugins.components.framework_transformer.base_transformer import BaseTransformer
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.function_extender import Extender
from mloda.core.abstract_plugins.implementation import Implementation, loaded_implementations

from tests.helpers.subclass_walks import collect_until

T = TypeVar("T")


class SubclassLog(Generic[T]):
    """Weak log of every subclass of `root` in definition order, filled from __init_subclass__."""

    def __init__(self, root: type[T]) -> None:
        self._refs: list[weakref.ref[type[T]]] = []
        self._hook(root)
        for cls in get_all_subclasses(root):
            if "__init_subclass__" in vars(cls):
                self._hook(cls)

    def _hook(self, owner: type[Any]) -> None:
        original = vars(owner).get("__init_subclass__")
        # The wrapper replaces the owner's own __init_subclass__ in vars(owner) for the test process.
        # A class whose definition raises before any hooked __init_subclass__ runs (a mixin before the root) is not logged.
        if getattr(getattr(original, "__func__", None), "_subclass_log", None) is self:
            return

        def logged(cls: type[Any], /, **kwargs: Any) -> None:
            # Log before delegating: a class the original rejects is already in __subclasses__ and must be reclaimed.
            self._refs.append(weakref.ref(cls))
            if "__init_subclass__" in vars(cls):
                self._hook(cls)
            if original is not None:
                original.__func__(cls, **kwargs)
            else:
                super(owner, cls).__init_subclass__(**kwargs)

        setattr(logged, "_subclass_log", self)
        setattr(owner, "__init_subclass__", classmethod(logged))

    def mark(self) -> int:
        """A snapshot that costs no walk."""
        return len(self._refs)

    def live_since(self, mark: int) -> list[type[T]]:
        """A fresh list of still-alive classes logged after `mark`, deduplicated in order."""
        alive = (ref() for ref in self._refs[mark:])
        return list(dict.fromkeys(cls for cls in alive if cls is not None))


FEATURE_GROUPS = SubclassLog(FeatureGroup)
TRANSFORMERS = SubclassLog(BaseTransformer)  # type: ignore[type-abstract]
IMPLEMENTATIONS = SubclassLog(Implementation)  # type: ignore[type-abstract]
COMPUTE_FRAMEWORKS = SubclassLog(ComputeFramework)
EXTENDERS = SubclassLog(Extender)  # type: ignore[type-abstract]

RECLAIM_LOGS: dict[type[Any], SubclassLog[Any]] = {
    FeatureGroup: FEATURE_GROUPS,
    BaseTransformer: TRANSFORMERS,
    Implementation: IMPLEMENTATIONS,
    ComputeFramework: COMPUTE_FRAMEWORKS,
    Extender: EXTENDERS,
}
EXTRA_RECLAIM_ROOTS: tuple[type[Any], ...] = (ComputeFramework, Extender)
YOUNG_WINDOW_THRESHOLD = 100_000
SHARED_TEST_MODULE_PREFIX = "tests.mixins."


@contextmanager
def young_window() -> Iterator[None]:
    """Raise the gen-0 threshold so automatic collections do not promote a class the test still holds."""
    saved = gc.get_threshold()
    gc.set_threshold(YOUNG_WINDOW_THRESHOLD, *saved[1:])
    try:
        yield
    finally:
        gc.set_threshold(*saved)


def _is_owned_by_its_module(cls: type[Any]) -> bool:
    """Is this class still bound under its own name in its live module, and so owned by that module?

    A module-owned class lives as long as the module does, so no collection would ever reclaim it. The dict
    lookup here skips a collection the suite otherwise pays hundreds of times (#995).
    """
    if "." in cls.__qualname__:  # a dotted qualname is never the plain module binding this reads
        return False
    module = sys.modules.get(cls.__module__)
    # vars(), not getattr(): a module-level __getattr__ (PEP 562) would run user code inside teardown.
    return vars(module).get(cls.__name__, None) is cls if module is not None else False


def _owned_by(cls: type[Any], module_name: str) -> bool:
    """A type()-built subclass (module 'abc') or one built in a shared mixin module during the test is also its own."""
    return cls.__module__ in (module_name, "abc") or cls.__module__.startswith(SHARED_TEST_MODULE_PREFIX)


def reclaim_leaked_subclasses(log: SubclassLog[T], mark: int, module_name: str) -> list[type[T]]:
    """Collect subclasses logged since `mark`; return the ones from `module_name` that survived."""

    def new_reclaimable() -> list[type[T]]:
        """A fresh list each call, so no survivor is held across a collection and pins what it would reclaim."""
        return [cls for cls in log.live_since(mark) if not _is_owned_by_its_module(cls)]

    def new_from_module() -> list[type[T]]:
        """Everything new this module owns, module-bound or not: a bound one is a leak no collection can fix."""
        return [cls for cls in log.live_since(mark) if _owned_by(cls, module_name)]

    # Cheap path: only collect when something collectable appeared. The gate reads ANY such new subclass, not
    # only this module's; only the RETURN value stays filtered to module_name.
    if not new_reclaimable():
        return sorted(new_from_module(), key=lambda cls: cls.__name__)
    # Youngest generation first: a class the test just defined is almost always still in gen 0, where a
    # collection is bounded by the young generation instead of the whole heap. Escalate to older generations
    # only while any new reclaimable class survives, whichever module owns it. Trade-off: a new class from another
    # module that is genuinely still alive now costs the full ladder in the test that created it.
    collect_until(lambda: not new_reclaimable())
    if not new_from_module():
        return []
    # Something is left, so pay the second collection: it reclaims what the first one only made collectable.
    gc.collect()
    return sorted(new_from_module(), key=lambda cls: cls.__name__)


def snapshot_roots() -> dict[type[Any], int]:
    """Mark the creation log of every reclaimed root."""
    return {root: log.mark() for root, log in RECLAIM_LOGS.items()}


def reclaim_since(snapshot: dict[type[Any], int], module_name: str) -> list[type[Any]]:
    """Reclaim every root since `snapshot`; report survivors of FeatureGroup, BaseTransformer, Implementation only."""
    leaked: list[type[Any]] = []
    for root, mark in snapshot.items():
        survivors = reclaim_leaked_subclasses(RECLAIM_LOGS[root], mark, module_name)
        if root in (FeatureGroup, BaseTransformer, Implementation):
            leaked.extend(survivors)
    return leaked


def reclaim_leaked_feature_groups(mark: int, module_name: str) -> list[type[FeatureGroup]]:
    """Collect FeatureGroup subclasses created since `mark`; return the ones from `module_name` that survived."""
    return reclaim_leaked_subclasses(FEATURE_GROUPS, mark, module_name)


def reclaim_leaked_transformers(mark: int, module_name: str) -> list[type[BaseTransformer]]:
    """Same over BaseTransformer: a leftover subclass joins the worker's next ComputeFrameworkTransformer() discovery
    and makes the ComputeFramework carrying it unpicklable for multiprocessing."""
    return reclaim_leaked_subclasses(TRANSFORMERS, mark, module_name)


def reclaim_released_implementations(mark: int, module_name: str) -> list[type[Any]]:
    """Reclaim Implementation subclasses created since `mark`; return those from `module_name` still alive and loaded."""
    reclaim_leaked_subclasses(IMPLEMENTATIONS, mark, module_name)
    loaded = loaded_implementations()
    return sorted(
        (cls for cls in IMPLEMENTATIONS.live_since(mark) if cls in loaded and cls.__module__ == module_name),
        key=lambda c: c.__name__,
    )
