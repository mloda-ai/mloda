from __future__ import annotations

import logging
import sys
from copy import deepcopy
from typing import Optional

from mloda.core.abstract_plugins.components.base_feature_group_version import BaseFeatureGroupVersion
from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.components.utils import get_all_subclasses


logger = logging.getLogger(__name__)


FeatureGroupEnvironmentMapping = dict[type[FeatureGroup], set[type[ComputeFramework]]]


def _running_in_zmq_shell() -> bool:
    """Detect IPython kernel (Jupyter) environment for the kernel-restart hint."""
    try:
        from IPython import get_ipython  # type: ignore[attr-defined]
    except ImportError:
        return False

    ipython_instance = get_ipython()  # type: ignore[no-untyped-call]
    if ipython_instance is None:
        return False
    return bool(ipython_instance.__class__.__name__ == "ZMQInteractiveShell")


def _safe_class_source_hash(cls: type[FeatureGroup]) -> Optional[str]:
    """Return source hash for a FeatureGroup subclass or None if unavailable.

    ``inspect.getsource`` raises ``OSError``/``TypeError`` for classes built
    dynamically via ``type()``, and may raise ``IndentationError``/``ValueError``
    on malformed source backings. All four leave the class without a stable
    source hash, so we return ``None``.
    """
    try:
        return BaseFeatureGroupVersion.class_source_hash(cls)
    except (OSError, TypeError, IndentationError, ValueError):
        return None


def dedup_feature_group_subclasses(
    classes: set[type[FeatureGroup]],
    allow_redefinition: bool = False,
) -> set[type[FeatureGroup]]:
    """Deduplicate FeatureGroup subclasses sharing the same ``(module, qualname)``.

    Long-lived namespaces (Jupyter notebooks, ``importlib.reload``) accumulate stale
    class objects in ``FeatureGroup.__subclasses__()``. This helper collapses
    identical-content duplicates silently and raises a clear ``ValueError`` when
    duplicates differ in source (override via ``allow_redefinition=True``).
    """
    grouped: dict[tuple[str, str], list[type[FeatureGroup]]] = {}
    for cls in classes:
        key = (cls.__module__, cls.__qualname__)
        grouped.setdefault(key, []).append(cls)

    survivors: set[type[FeatureGroup]] = set()
    conflicts: list[tuple[tuple[str, str], list[tuple[type[FeatureGroup], str]]]] = []

    for key, members in grouped.items():
        if len(members) == 1:
            survivors.add(members[0])
            continue

        # Factory pattern: groups whose members are not bound under their name in
        # any module namespace are produced by factories or held only via
        # closures. They are intentionally distinct objects, so preserve all.
        if not _any_live_in_module(members):
            logger.debug(
                "dedup_feature_group_subclasses: preserving group %s (no member bound in its module)",
                key,
            )
            survivors.update(members)
            continue

        # Live-descendants check: preserve the whole group if any member has
        # subclasses already alive in the FG tree.
        has_live_descendants = any(len(get_all_subclasses(member)) > 0 for member in members)
        if has_live_descendants:
            logger.debug("dedup_feature_group_subclasses: preserving group %s due to live descendants", key)
            survivors.update(members)
            continue

        hashes = [_safe_class_source_hash(m) for m in members]
        if any(h is None for h in hashes):
            logger.debug("dedup_feature_group_subclasses: preserving group %s (source unavailable)", key)
            survivors.update(members)
            continue

        unique_hashes = set(hashes)
        if len(unique_hashes) == 1:
            survivor = _pick_survivor(members)
            logger.debug(
                "dedup_feature_group_subclasses: collapsed %d identical members of %s",
                len(members),
                key,
            )
            survivors.add(survivor)
            continue

        if allow_redefinition:
            survivor = _pick_survivor(members)
            logger.debug(
                "dedup_feature_group_subclasses: kept newest of %d differing members of %s (allow_redefinition=True)",
                len(members),
                key,
            )
            survivors.add(survivor)
            continue

        conflicts.append((key, list(zip(members, hashes))))  # type: ignore[arg-type]

    if conflicts:
        raise ValueError(_build_conflict_error(conflicts))

    return survivors


def _pick_survivor(members: list[type[FeatureGroup]]) -> type[FeatureGroup]:
    """Pick the live-in-module class, falling back to the last entry."""
    for cls in members:
        if _is_live_in_module(cls):
            return cls
    return members[-1]


def _is_live_in_module(cls: type[FeatureGroup]) -> bool:
    """True if ``cls`` is currently bound under its name in its own module."""
    module = sys.modules.get(cls.__module__)
    if module is None:
        return False
    return getattr(module, cls.__name__, None) is cls


def _any_live_in_module(members: list[type[FeatureGroup]]) -> bool:
    """True if any class is currently bound under its name in its module."""
    return any(_is_live_in_module(cls) for cls in members)


def _build_conflict_error(
    conflicts: list[tuple[tuple[str, str], list[tuple[type[FeatureGroup], str]]]],
) -> str:
    lines = ["FeatureGroup redefined with different source code:"]
    for (module, qualname), hashed in conflicts:
        for _cls, source_hash in hashed:
            lines.append(f"  - {qualname} ({module}) source hash {source_hash[:8]}")
    lines.append(
        "Set PluginCollector(...).set_allow_redefinition() to keep only the most "
        "recently defined version of each class."
    )
    if _running_in_zmq_shell():
        lines.append("If you are running this in a notebook, restart the kernel to clear stale class definitions.")
    return "\n".join(lines)


class PreFilterPlugins:
    def __init__(
        self,
        compute_frameworks: set[type[ComputeFramework]],
        plugin_collector: Optional[PluginCollector] = None,
    ) -> None:
        feature_groups = self._set_feature_groups(plugin_collector)
        compute_frameworks = self._set_compute_frameworks(compute_frameworks)

        self.accessible_plugins = self.resolve_feature_group_compute_framework_limitations(
            feature_groups, compute_frameworks
        )

    def get_accessible_plugins(self) -> FeatureGroupEnvironmentMapping:
        return self.accessible_plugins

    def _set_feature_groups(self, plugin_collector: Optional[PluginCollector] = None) -> set[type[FeatureGroup]]:
        accessible_feature_groups = self.get_featuregroup_subclasses()

        if plugin_collector:
            for accessible_fg in deepcopy(accessible_feature_groups):
                if not plugin_collector.applicable_feature_group_class(accessible_fg):
                    accessible_feature_groups.remove(accessible_fg)

        allow_redefinition = plugin_collector.allow_redefinition if plugin_collector is not None else False
        accessible_feature_groups = dedup_feature_group_subclasses(
            accessible_feature_groups, allow_redefinition=allow_redefinition
        )

        if len(accessible_feature_groups) == 0:
            raise ValueError("No feature groups are loaded. Did you call PluginLoader.all()?")
        return accessible_feature_groups

    def _set_compute_frameworks(
        self,
        compute_frameworks: set[type[ComputeFramework]],
    ) -> set[type[ComputeFramework]]:
        return compute_frameworks.intersection(self.get_cfw_subclasses())

    def resolve_feature_group_compute_framework_limitations(
        self, feature_groups: set[type[FeatureGroup]], compute_frameworks: set[type[ComputeFramework]]
    ) -> FeatureGroupEnvironmentMapping:
        accessible_plugins: FeatureGroupEnvironmentMapping = {}
        for feature_group in feature_groups:
            new_set_of_compute_frameworks = set()
            for cp_fg in feature_group.compute_framework_definition():
                if cp_fg in compute_frameworks:
                    new_set_of_compute_frameworks.add(cp_fg)

            accessible_plugins[feature_group] = new_set_of_compute_frameworks

        return accessible_plugins

    @staticmethod
    def get_cfw_subclasses() -> set[type[ComputeFramework]]:
        all_subclasses = get_all_subclasses(ComputeFramework)
        available_subclasses = {cls for cls in all_subclasses if cls.is_available()}
        return available_subclasses

    @staticmethod
    def get_featuregroup_subclasses() -> set[type[FeatureGroup]]:
        return get_all_subclasses(FeatureGroup)
