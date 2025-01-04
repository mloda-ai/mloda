from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Type, Union
from uuid import UUID
from mloda_core.prepare.graph.graph import Graph
from mloda_core.prepare.resolve_compute_frameworks import ResolveComputeFrameworks
from mloda_core.prepare.resolve_links import LinkFrameworkTrekker, ResolveLinks
from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.link import Link


LinkFeatureQueue = List[Union[LinkFrameworkTrekker, Tuple[Feature, Type[AbstractFeatureGroup]]]]

PlannedQueue = List[Union[LinkFrameworkTrekker, Tuple[Type[AbstractFeatureGroup], Set[Feature]]]]


class ResolveGraph:
    def __init__(self, graph: Graph, links: Optional[Set[Link]] = None):
        self.graph = graph
        self.nodes_per_feature_group: Dict[Type[AbstractFeatureGroup], Set[Feature]] = {}
        self.resolver_compute_framework = ResolveComputeFrameworks(self.graph)
        self.resolver_links = ResolveLinks(self.graph, links)

    def create_initial_queue(self) -> None:
        self.graph.iterate_nodes_and_edges()

    def set_nodes_per_feature_group(self) -> None:
        self.nodes_per_feature_group = self.get_nodes_with_same_feature_group_class()

    def resolve_links(self) -> PlannedQueue:
        # we create feature link relation
        self.resolver_links.resolve_links()
        # we put link into queue depending on feature link relation
        self.links_with_queue = self.resolver_links.add_links_to_queue()

        # we convert uuids of graph to features
        queue_feature_links = self.convert_links_with_queue_to_features(self.links_with_queue)
        # we group features according to feature groups together
        return self.combine_features_of_feature_group(queue_feature_links)

    def combine_features_of_feature_group(self, queue: LinkFeatureQueue) -> PlannedQueue:
        visited_features: Set[UUID] = set()
        planned_queue: PlannedQueue = []

        for item in queue:
            if isinstance(item[0], Link):
                planned_queue.append(item)
                continue

            if len(item) == 2:
                feature, feature_group = item
                if feature.uuid in visited_features:
                    continue

                features = self.nodes_per_feature_group[feature_group]

                planned_queue.append((feature_group, self.nodes_per_feature_group[feature_group]))
                visited_features.update([e.uuid for e in features])

        return planned_queue

    def convert_links_with_queue_to_features(
        self, links_with_queue: List[Union[UUID, LinkFrameworkTrekker]]
    ) -> LinkFeatureQueue:
        feature_link_queue: LinkFeatureQueue = []

        for link_or_uuid in links_with_queue:
            if isinstance(link_or_uuid, UUID):
                node = self.graph.get_nodes()[link_or_uuid]
                feature_link_queue.append((node.feature, node.feature_group_class))
            else:
                feature_link_queue.append(link_or_uuid)

        return feature_link_queue

    def get_nodes_with_same_feature_group_class(self) -> Dict[Type[AbstractFeatureGroup], Set[Feature]]:
        collection: Dict[Type[AbstractFeatureGroup], Set[Feature]] = defaultdict(set)

        for node in self.graph.queue:
            node_properties = self.graph.get_nodes()[node]
            collection[node_properties.feature_group_class].add(node_properties.feature)

        return collection
