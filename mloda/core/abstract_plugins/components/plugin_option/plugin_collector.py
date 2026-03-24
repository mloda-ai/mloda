from typing import Set, Type

from mloda.core.abstract_plugins.feature_group import FeatureGroup


class PluginCollector:
    """
    The PluginCollector class is a helper class with the purpose to disable or enable feature groups.

    This class is useful for rapid prototype development, where you want to disable or enable feature groups,
    when the other, competing feature groups are found.

    Further, this class is useful for testing, where you want to test the behavior of the system with different
    feature groups enabled or disabled.
    """

    def __init__(self) -> None:
        self.disabled_feature_group_classes: set[type[FeatureGroup]] = set()
        self.enabled_feature_group_classes: set[type[FeatureGroup]] = set()

    def add_disabled_feature_group_classes(self, feature_group_cls: set[type[FeatureGroup]]) -> None:
        self.disabled_feature_group_classes.update(feature_group_cls)

    def add_enabled_feature_group_classes(self, feature_group_cls: set[type[FeatureGroup]]) -> None:
        self.enabled_feature_group_classes.update(feature_group_cls)

    def applicable_feature_group_class(self, feature_group_cls: type[FeatureGroup]) -> bool:
        if feature_group_cls in self.disabled_feature_group_classes:
            return False

        # If no feature groups are enabled, all feature groups are enabled.
        if len(self.enabled_feature_group_classes) == 0:
            return True

        if feature_group_cls in self.enabled_feature_group_classes:
            return True
        return False

    @staticmethod
    def disabled_feature_groups(
        feature_group_cls: set[type[FeatureGroup]] | type[FeatureGroup],
    ) -> "PluginCollector":
        if not isinstance(feature_group_cls, set):
            feature_group_cls = {feature_group_cls}

        plugin_collector = PluginCollector()
        plugin_collector.add_disabled_feature_group_classes(feature_group_cls)
        return plugin_collector

    @staticmethod
    def enabled_feature_groups(
        feature_group_cls: set[type[FeatureGroup]] | type[FeatureGroup],
    ) -> "PluginCollector":
        if not isinstance(feature_group_cls, set):
            feature_group_cls = {feature_group_cls}

        plugin_collector = PluginCollector()
        plugin_collector.add_enabled_feature_group_classes(feature_group_cls)
        return plugin_collector
