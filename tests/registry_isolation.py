"""Shared FeatureGroup registry-isolation helper (#845).

A FeatureGroup subclass defined inside a test sits in a reference cycle, so it stays in
``FeatureGroup.__subclasses__()`` until cyclic GC runs and a later test on the same worker trips over it.
"""

from __future__ import annotations

import gc
import sys

from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.feature_group import FeatureGroup


def _is_owned_by_its_module(cls: type[FeatureGroup]) -> bool:
    """Is this class still bound under its own name in its live module, and so owned by that module?

    A test that imports a plugin module registers every FeatureGroup it defines. Those are not the
    transient, cycle-held classes this helper reclaims: they live as long as the module does, and no
    collection would ever take them. Reading the binding costs a dict lookup, while the collection it
    skips costs about a second (#995).
    """
    if "." in cls.__qualname__:  # defined inside a function or class body, so never a module attribute
        return False
    module = sys.modules.get(cls.__module__)
    return getattr(module, cls.__name__, None) is cls


def reclaim_leaked_feature_groups(before: set[type[FeatureGroup]], module_name: str) -> list[type[FeatureGroup]]:
    """Collect FeatureGroup subclasses created since `before`; return the ones from `module_name` that survived."""

    def new_reclaimable() -> list[type[FeatureGroup]]:
        """A fresh list each call, so no survivor is held across a collection and pins what it would reclaim."""
        return [cls for cls in get_all_subclasses(FeatureGroup) - before if not _is_owned_by_its_module(cls)]

    def new_from_module() -> list[type[FeatureGroup]]:
        return [cls for cls in new_reclaimable() if cls.__module__ == module_name]

    # Cheap path: a collection costs up to ~1s, so only pay it when something reclaimable appeared. The gate
    # reads ANY such new subclass, not only this module's; only the RETURN value stays filtered to module_name.
    if not new_reclaimable():
        return []
    # Youngest generation first: a class the test just defined is almost always still in gen 0, where a
    # collection is bounded by the young generation instead of the whole heap. Only a class that automatic
    # collections promoted needs the older generations, and a full collection is the last resort (#995).
    for generation in (0, 1, 2):
        gc.collect(generation)
        if not new_reclaimable():
            return []
    if not new_from_module():
        return []
    # Something is left, so pay the second collection: it reclaims what the first one only made collectable.
    gc.collect()
    return sorted(new_from_module(), key=lambda cls: cls.__name__)
