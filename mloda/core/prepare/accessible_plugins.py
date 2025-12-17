from copy import deepcopy
from typing import Optional, Set, Type
from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.abstract_plugins.compute_framework import ComputeFramework

from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.components.utils import get_all_subclasses


FeatureGroupEnvironmentMapping = dict[Type[FeatureGroup], Set[Type[ComputeFramework]]]


class PreFilterPlugins:
    def __init__(
        self,
        compute_frameworks: Set[Type[ComputeFramework]],
        plugin_collector: Optional[PluginCollector] = None,
    ) -> None:
        feature_groups = self._set_feature_groups(plugin_collector)
        compute_frameworks = self._set_compute_frameworks(compute_frameworks)

        self.accessible_plugins = self.resolve_feature_group_compute_framework_limitations(
            feature_groups, compute_frameworks
        )

    def get_accessible_plugins(self) -> FeatureGroupEnvironmentMapping:
        return self.accessible_plugins

    def _set_feature_groups(self, plugin_collector: Optional[PluginCollector] = None) -> Set[Type[FeatureGroup]]:
        accessible_feature_groups = self.get_featuregroup_subclasses()

        if plugin_collector:
            for accessible_fg in deepcopy(accessible_feature_groups):
                if not plugin_collector.applicable_feature_group_class(accessible_fg):
                    accessible_feature_groups.remove(accessible_fg)

        if len(accessible_feature_groups) == 0:
            raise ValueError("No accessible feature groups found.")
        return accessible_feature_groups

    def _set_compute_frameworks(
        self,
        compute_frameworks: Set[Type[ComputeFramework]],
    ) -> Set[Type[ComputeFramework]]:
        return compute_frameworks.intersection(self.get_cfw_subclasses())

    def resolve_feature_group_compute_framework_limitations(
        self, feature_groups: Set[Type[FeatureGroup]], compute_frameworks: Set[Type[ComputeFramework]]
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
    def get_cfw_subclasses() -> Set[Type[ComputeFramework]]:
        all_subclasses = get_all_subclasses(ComputeFramework)
        available_subclasses = {cls for cls in all_subclasses if cls.is_available()}
        return available_subclasses

    @staticmethod
    def get_featuregroup_subclasses() -> Set[Type[FeatureGroup]]:
        return get_all_subclasses(FeatureGroup)
