from typing import Dict, Set, Type
from uuid import UUID
from mloda_core.prepare.graph.graph import Graph
from mloda_core.prepare.graph.properties import EdgeProperties, NodeProperties
from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature


class BuildGraph:
    def __init__(
        self,
        feature_link_parents: Dict[UUID, Set[UUID]],
        feature_group_collection: Dict[Type[AbstractFeatureGroup], Set[Feature]],
    ) -> None:
        self._graph = Graph()
        self.feature_link_parents = feature_link_parents

        self.property_mapping = self.flatten_feature_group_to_uuid_mapping(feature_group_collection)

    @property
    def graph(self) -> Graph:
        return self._graph

    def build_graph_from_feature_links(self) -> None:
        for child, parents in self.feature_link_parents.items():
            self.graph.add_node(child, self.property_mapping[child])

            for parent in parents:
                self.graph.add_node(parent, self.property_mapping[parent])
                self.graph.add_edge(parent, child, self.create_property_edge(parent, child))

    def flatten_feature_group_to_uuid_mapping(
        self, feature_group_collection: Dict[Type[AbstractFeatureGroup], Set[Feature]]
    ) -> Dict[UUID, NodeProperties]:
        flattened_mapping = {}
        for feature_group_class, feature_set in feature_group_collection.items():
            for feature in feature_set:
                flattened_mapping[feature.uuid] = NodeProperties(feature, feature_group_class).return_self()
        return flattened_mapping

    def create_property_edge(self, parent: UUID, child: UUID) -> EdgeProperties:
        parent_feature_group_class = self.property_mapping[parent].feature_group_class
        child_feature_group_class = self.property_mapping[child].feature_group_class

        return EdgeProperties(parent_feature_group_class, child_feature_group_class).return_self()
