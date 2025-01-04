from typing import Optional, Set, Tuple, Type
from mloda_core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda_core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.link import Link

import logging

logger = logging.getLogger(__name__)


class IdentifyFeatureGroupClass:
    def __init__(
        self,
        feature: Feature,
        accessible_plugins: FeatureGroupEnvironmentMapping,
        links: Optional[Set[Link]],
        data_access_collection: Optional[DataAccessCollection] = None,
    ):
        feature_group = self._filter_loop(feature, accessible_plugins, links, data_access_collection)

        self.validate(feature_group, feature)
        self.feature_group_compute_framework_mapping = feature_group

    def _filter_loop(
        self,
        feature: Feature,
        accessible_plugins: FeatureGroupEnvironmentMapping,
        links: Optional[Set[Link]],
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> FeatureGroupEnvironmentMapping:
        _identified_feature_groups: FeatureGroupEnvironmentMapping = {}

        for feature_group, compute_frameworks in accessible_plugins.items():
            if not self._filter_feature_group_by_criteria(feature_group, feature, data_access_collection):
                continue

            if not self._filter_feature_group_by_domain(feature_group, feature):
                continue

            if not self._filter_feature_group_by_framework(compute_frameworks, feature):
                continue

            if not self._filter_feature_group_by_links(feature_group, links):
                continue

            if compute_frameworks:
                _identified_feature_groups[feature_group] = compute_frameworks

        _identified_feature_groups = self.filter_subclasses(_identified_feature_groups)
        return _identified_feature_groups

    def _filter_feature_group_by_links(
        self, feature_group: Type[AbstractFeatureGroup], links: Optional[Set[Link]]
    ) -> bool:
        # Case index columns not given, so no validation possible
        if feature_group.index_columns() is None:
            return True

        # Case no links given, so no validation possible
        if links is None:
            return True

        # Validate that atleast one index is supported by the feature group
        for link in links:
            if feature_group.supports_index(link.left_index):
                return True

            if feature_group.supports_index(link.right_index):
                return True

        return False

    def _filter_feature_group_by_criteria(
        self,
        feature_group: Type[AbstractFeatureGroup],
        feature: Feature,
        data_access_collection: Optional[DataAccessCollection],
    ) -> bool:
        return feature_group.match_feature_group_criteria(feature.name, feature.options, data_access_collection)

    def _filter_feature_group_by_domain(self, feature_group: Type[AbstractFeatureGroup], feature: Feature) -> bool:
        return not feature.domain or feature_group.get_domain() == feature.domain

    def _filter_feature_group_by_framework(
        self,
        compute_frameworks: Set[Type[ComputeFrameWork]],
        feature: Feature,
    ) -> bool:
        if feature.compute_frameworks is None:
            return True

        if len(feature.compute_frameworks) > 1:
            raise ValueError(f"Feature should only have one compute framework when set by user {feature.name}.")

        return feature.get_compute_framework() in compute_frameworks

    def validate(self, feature_group: FeatureGroupEnvironmentMapping, feature: Feature) -> None:
        if not feature_group:
            raise ValueError(f"No feature groups found for feature name: {feature.name}.")
        if len(feature_group) > 1:
            raise ValueError(f"Multiple feature groups {feature_group} found for feature name: {feature.name}.")
        elif len(feature_group) == 0:
            raise ValueError(f"No feature groups found for feature name: {feature.name}.")

        feature_group_class, compute_frameworks = next(iter(feature_group.items()))
        if not compute_frameworks:
            raise ValueError(f"Feature {feature.name} {feature_group_class.get_class_name()} has no compute framework.")

    def get(self) -> Tuple[Type[AbstractFeatureGroup], Set[Type[ComputeFrameWork]]]:
        return next(iter(self.feature_group_compute_framework_mapping.items()))

    def filter_subclasses(
        self, _identified_feature_groups: FeatureGroupEnvironmentMapping
    ) -> FeatureGroupEnvironmentMapping:
        """
        This functionality ensures that only subclass feature groups are kept.
        """
        fgs_to_pop: Set[Type[AbstractFeatureGroup]] = set()

        for i_feature_group, i_compute_frameworks in _identified_feature_groups.items():
            for o_feature_group, o_compute_frameworks in _identified_feature_groups.items():
                if i_compute_frameworks != o_compute_frameworks:
                    continue

                if i_feature_group == o_feature_group:
                    continue

                if issubclass(i_feature_group, o_feature_group):
                    fgs_to_pop.add(o_feature_group)

        for fg in fgs_to_pop:
            _identified_feature_groups.pop(fg)

        return _identified_feature_groups
