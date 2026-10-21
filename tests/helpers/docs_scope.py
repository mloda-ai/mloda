"""Scoping helpers so plugin-docs tests never enumerate every FeatureGroup subclass of the worker."""

from collections.abc import Callable, Iterable

from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.feature_group import FeatureGroup

_SHIPPED_PREFIXES = ("mloda.", "mloda_plugins.")


def shipped_feature_groups() -> set[type[FeatureGroup]]:
    """FeatureGroup subclasses defined in mloda or mloda_plugins."""
    return {fg for fg in get_all_subclasses(FeatureGroup) if fg.__module__.startswith(_SHIPPED_PREFIXES)}


def shipped_feature_groups_collector(extra: Iterable[type[FeatureGroup]] = ()) -> PluginCollector:
    """Collector enabling the shipped groups plus any extra classes."""
    return PluginCollector.enabled_feature_groups(shipped_feature_groups() | set(extra))


def small_feature_group_walk(
    *classes: type[FeatureGroup],
) -> Callable[[type], set[type]]:
    """Stub for the docs walk: the given groups for FeatureGroup, the real walk for any other base."""
    keep = set(classes)

    def walk(base: type) -> set[type]:
        if base is FeatureGroup:
            return set(keep)
        return get_all_subclasses(base)

    return walk


def shipped_feature_group_walk() -> Callable[[type], set[type]]:
    """Stub for the docs walk: shipped groups looked up at call time for FeatureGroup, the real walk otherwise."""

    def walk(base: type) -> set[type]:
        if base is FeatureGroup:
            return set(shipped_feature_groups())
        return get_all_subclasses(base)

    return walk
