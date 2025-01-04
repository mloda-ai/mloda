from __future__ import annotations

from typing import Type

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature


class NodeProperties:
    def __init__(self, feature: Feature, feature_group_class: Type[AbstractFeatureGroup]) -> None:
        self.feature = feature
        self.feature_group_class = feature_group_class
        self.name = feature.name

    def return_self(self) -> NodeProperties:
        return self


class EdgeProperties:
    def __init__(
        self,
        parent_feature_group_class: Type[AbstractFeatureGroup],
        child_feature_group_class: Type[AbstractFeatureGroup],
    ) -> None:
        self.parent_feature_group_class = parent_feature_group_class
        self.child_feature_group_class = child_feature_group_class

    def return_self(self) -> EdgeProperties:
        return self
