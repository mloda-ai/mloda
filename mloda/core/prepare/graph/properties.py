from __future__ import annotations

from typing import Type

from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.components.feature import Feature


class NodeProperties:
    def __init__(self, feature: Feature, feature_group_class: Type[FeatureGroup]) -> None:
        self.feature = feature
        self.feature_group_class = feature_group_class
        self.name = feature.name

    def return_self(self) -> NodeProperties:
        return self


class EdgeProperties:
    def __init__(
        self,
        parent_feature_group_class: Type[FeatureGroup],
        child_feature_group_class: Type[FeatureGroup],
    ) -> None:
        self.parent_feature_group_class = parent_feature_group_class
        self.child_feature_group_class = child_feature_group_class

    def return_self(self) -> EdgeProperties:
        return self
