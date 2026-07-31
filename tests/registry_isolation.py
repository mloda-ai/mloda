"""Shared FeatureGroup registry-isolation helper (#845).

A FeatureGroup subclass defined inside a test sits in a reference cycle, so it stays in
``FeatureGroup.__subclasses__()`` until cyclic GC runs and a later test on the same worker trips over it.
"""

from __future__ import annotations

import gc

from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.feature_group import FeatureGroup


def reclaim_leaked_feature_groups(before: set[type[FeatureGroup]], module_name: str) -> list[type[FeatureGroup]]:
    """Collect FeatureGroup subclasses created since `before`; return the ones from `module_name` that survived."""

    def new_from_module() -> list[type[FeatureGroup]]:
        """A fresh list each call, so no survivor is held across a collection and pins what it would reclaim."""
        return [cls for cls in get_all_subclasses(FeatureGroup) - before if cls.__module__ == module_name]

    # Cheap path: a full collection costs ~1s, so only pay it when something new appeared. The gate reads ANY
    # new subclass, not only this module's; only the RETURN value stays filtered to module_name.
    if not get_all_subclasses(FeatureGroup) - before:
        return []
    gc.collect()
    if not new_from_module():
        return []
    # Something is left, so pay the second collection: it reclaims what the first one only made collectable.
    gc.collect()
    return sorted(new_from_module(), key=lambda cls: cls.__name__)
