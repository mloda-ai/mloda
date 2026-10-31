"""Scoping helpers so plugin-docs tests never enumerate every FeatureGroup subclass of the worker."""

from collections.abc import Callable, Iterable

from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.abstract_plugins.feature_group import FeatureGroup, feature_group_subclasses

_SHIPPED_PREFIXES = ("mloda.", "mloda_plugins.")


def shipped_feature_groups() -> set[type[FeatureGroup]]:
    """FeatureGroup subclasses defined in mloda or mloda_plugins."""
    return {fg for fg in feature_group_subclasses() if fg.__module__.startswith(_SHIPPED_PREFIXES)}


def shipped_feature_groups_collector(extra: Iterable[type[FeatureGroup]] = ()) -> PluginCollector:
    """Collector enabling the shipped groups plus any extra classes."""
    return PluginCollector.enabled_feature_groups(shipped_feature_groups() | set(extra))


def small_feature_group_registry(
    *classes: type[FeatureGroup],
) -> Callable[[], set[type[FeatureGroup]]]:
    """Stub for `feature_group_subclasses`: returns just the given groups."""
    keep = set(classes)

    def registry() -> set[type[FeatureGroup]]:
        return set(keep)

    return registry


def shipped_feature_group_registry() -> Callable[[], set[type[FeatureGroup]]]:
    """Stub for `feature_group_subclasses`: shipped groups looked up at call time."""

    def registry() -> set[type[FeatureGroup]]:
        return shipped_feature_groups()

    return registry
