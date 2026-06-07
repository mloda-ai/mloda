from difflib import get_close_matches
from typing import Optional

from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup, format_feature_group_class
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.link import Link

import logging

logger = logging.getLogger(__name__)


class IdentifyFeatureGroupClass:
    def __init__(
        self,
        feature: Feature,
        accessible_plugins: FeatureGroupEnvironmentMapping,
        links: Optional[set[Link]],
        data_access_collection: Optional[DataAccessCollection] = None,
    ):
        self._criteria_matched_feature_groups: set[type[FeatureGroup]] = set()

        feature_group = self._filter_loop(feature, accessible_plugins, links, data_access_collection)

        self.validate(feature_group, feature, accessible_plugins)
        self.feature_group_compute_framework_mapping = feature_group

    def _filter_loop(
        self,
        feature: Feature,
        accessible_plugins: FeatureGroupEnvironmentMapping,
        links: Optional[set[Link]],
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> FeatureGroupEnvironmentMapping:
        _identified_feature_groups: FeatureGroupEnvironmentMapping = {}

        for feature_group, compute_frameworks in accessible_plugins.items():
            if not self._filter_feature_group_by_criteria(feature_group, feature, data_access_collection):
                continue

            if not self._filter_feature_group_by_domain(feature_group, feature):
                continue

            self._criteria_matched_feature_groups.add(feature_group)

            supported_frameworks = {
                cfw
                for cfw in compute_frameworks
                if feature_group.supports_compute_framework(feature.name, feature.options, cfw)
            }

            if not self._filter_feature_group_by_framework(supported_frameworks, feature):
                continue

            if not self._filter_feature_group_by_links(feature_group, links):
                continue

            if supported_frameworks:
                _identified_feature_groups[feature_group] = supported_frameworks

        _identified_feature_groups = self.filter_subclasses(_identified_feature_groups)
        return _identified_feature_groups

    def _filter_feature_group_by_links(self, feature_group: type[FeatureGroup], links: Optional[set[Link]]) -> bool:
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
        feature_group: type[FeatureGroup],
        feature: Feature,
        data_access_collection: Optional[DataAccessCollection],
    ) -> bool:
        return feature_group.match_feature_group_criteria(feature.name, feature.options, data_access_collection)

    def _filter_feature_group_by_domain(self, feature_group: type[FeatureGroup], feature: Feature) -> bool:
        return not feature.domain or feature_group.get_domain() == feature.domain

    def _filter_feature_group_by_framework(
        self,
        compute_frameworks: set[type[ComputeFramework]],
        feature: Feature,
    ) -> bool:
        if feature.compute_frameworks is None:
            return True

        if len(feature.compute_frameworks) > 1:
            raise ValueError(f"Feature should only have one compute framework when set by user {feature.name}.")

        return feature.get_compute_framework() in compute_frameworks

    def validate(
        self,
        feature_group: FeatureGroupEnvironmentMapping,
        feature: Feature,
        accessible_plugins: FeatureGroupEnvironmentMapping,
    ) -> None:
        if not feature_group or len(feature_group) == 0:
            raise ValueError(self._build_no_feature_group_error(feature, accessible_plugins))
        if len(feature_group) > 1:
            from mloda.core.abstract_plugins.feature_group import format_feature_group_classes

            raise ValueError(
                f"Multiple feature groups found for feature '{feature.name}':\n"
                f"{format_feature_group_classes(feature_group.keys(), include_domain=True)}\n"
                "For troubleshooting guide, see: https://mloda-ai.github.io/mloda/in_depth/troubleshooting/feature-group-resolution-errors/"
            )

        feature_group_class, compute_frameworks = next(iter(feature_group.items()))
        if not compute_frameworks:
            raise ValueError(
                f"Feature {feature.name} {format_feature_group_class(feature_group_class)} has no compute framework."
            )

    def _capability_rejection_message(
        self, feature: Feature, accessible_plugins: FeatureGroupEnvironmentMapping
    ) -> Optional[str]:
        rejected: set[type[ComputeFramework]] = set()
        supported: set[type[ComputeFramework]] = set()

        for fg in self._criteria_matched_feature_groups:
            for cfw in accessible_plugins.get(fg, set()):
                if fg.supports_compute_framework(feature.name, feature.options, cfw):
                    supported.add(cfw)
                else:
                    rejected.add(cfw)

        if not rejected:
            return None

        rejected_names = sorted(fw.get_class_name() for fw in rejected)
        msg = f"Unsupported compute framework(s) for feature '{str(feature.name)}': {rejected_names}."

        if supported:
            supported_names = sorted(fw.get_class_name() for fw in supported)
            msg += f" Supported on: {supported_names}."

        msg += " Pin the feature to a supported compute framework or override supports_compute_framework."
        return msg

    def _build_no_feature_group_error(
        self, feature: Feature, accessible_plugins: FeatureGroupEnvironmentMapping
    ) -> str:
        capability_message = self._capability_rejection_message(feature, accessible_plugins)
        if capability_message is not None:
            return capability_message

        feature_name = str(feature.name)
        msg = f"No feature groups found for feature name: '{feature_name}'."

        if not accessible_plugins:
            msg += "\nNo plugins are loaded. Did you call PluginLoader.all()?"
            return msg

        known_names: list[str] = []
        for fg_class in accessible_plugins:
            known_names.append(fg_class.get_class_name())
            known_names.extend(fg_class.feature_names_supported())
            if fg_class.prefix():
                known_names.append(fg_class.prefix())

        similar = get_close_matches(feature_name, known_names, n=5, cutoff=0.5)
        if similar:
            msg += f"\nDid you mean one of: {similar}?"

        msg += (
            "\nUse resolve_feature(name) to debug feature resolution."
            "\nFor troubleshooting guide, see: "
            "https://mloda-ai.github.io/mloda/in_depth/troubleshooting/feature-group-resolution-errors/"
        )
        return msg

    def get(self) -> tuple[type[FeatureGroup], set[type[ComputeFramework]]]:
        return next(iter(self.feature_group_compute_framework_mapping.items()))

    def filter_subclasses(
        self, _identified_feature_groups: FeatureGroupEnvironmentMapping
    ) -> FeatureGroupEnvironmentMapping:
        """
        This functionality ensures that only subclass feature groups are kept.
        """
        fgs_to_pop: set[type[FeatureGroup]] = set()

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
