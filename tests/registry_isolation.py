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

    A module-owned class lives as long as the module does, so no collection would ever reclaim it. The dict
    lookup here skips a collection the suite otherwise pays hundreds of times (#995).
    """
    if "." in cls.__qualname__:  # a dotted qualname is never the plain module binding this reads
        return False
    module = sys.modules.get(cls.__module__)
    # vars(), not getattr(): a module-level __getattr__ (PEP 562) would run user code inside teardown.
    return vars(module).get(cls.__name__, None) is cls if module is not None else False


def reclaim_leaked_feature_groups(before: set[type[FeatureGroup]], module_name: str) -> list[type[FeatureGroup]]:
    """Collect FeatureGroup subclasses created since `before`; return the ones from `module_name` that survived."""

    def new_reclaimable() -> list[type[FeatureGroup]]:
        """A fresh list each call, so no survivor is held across a collection and pins what it would reclaim."""
        return [cls for cls in get_all_subclasses(FeatureGroup) - before if not _is_owned_by_its_module(cls)]

    def new_from_module() -> list[type[FeatureGroup]]:
        """Everything new this module owns, module-bound or not: a bound one is a leak no collection can fix."""
        return [cls for cls in get_all_subclasses(FeatureGroup) - before if cls.__module__ == module_name]

    # Cheap path: only collect when something collectable appeared. The gate reads ANY such new subclass, not
    # only this module's; only the RETURN value stays filtered to module_name.
    if not new_reclaimable():
        return sorted(new_from_module(), key=lambda cls: cls.__name__)
    # Youngest generation first: a class the test just defined is almost always still in gen 0, where a
    # collection is bounded by the young generation instead of the whole heap. Escalating costs the whole-heap
    # scan, so it is spent only while this module still has a survivor, the one thing the caller asserts on.
    # Another module's promoted transient therefore gets the gen-0 pass only, and the module that owns it pays
    # the full ladder on its own tests (#995).
    for generation in (0, 1, 2):
        gc.collect(generation)
        if not new_from_module():
            break
    if not new_from_module():
        return []
    # Something is left, so pay the second collection: it reclaims what the first one only made collectable.
    gc.collect()
    return sorted(new_from_module(), key=lambda cls: cls.__name__)
