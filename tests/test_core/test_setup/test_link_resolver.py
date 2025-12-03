import uuid
from mloda_core.prepare.graph.graph import Graph
from mloda_core.prepare.graph.properties import EdgeProperties, NodeProperties
from mloda_core.prepare.resolve_graph import ResolveGraph
from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.index.index import Index
from mloda_core.abstract_plugins.components.link import Link, JoinSpec
from tests.test_core.test_setup.test_graph_builder import BaseTestGraphFeatureGroup3
from tests.test_core.test_abstract_plugins.test_abstract_feature_group import BaseTestFeatureGroup1
from tests.test_core.test_abstract_plugins.test_abstract_compute_framework import (
    BaseTestComputeFrameWork1,
    BaseTestComputeFrameWork2,
    BaseTestComputeFrameWork3,
)


class BaseLinkTestFeatureGroup1(BaseTestFeatureGroup1):
    pass


uuid_1, uuid_2, uuid_3, uuid_4, uuid_5, uuid_6, uuid_7 = (
    uuid.UUID(int=1),
    uuid.UUID(int=2),
    uuid.UUID(int=3),
    uuid.UUID(int=4),
    uuid.UUID(int=5),
    uuid.UUID(int=6),
    uuid.UUID(int=7),
)


class TestResolveGraph:
    def create_graph(self) -> Graph:
        f1, f2, f3, f4, f5, f6, f7 = (
            Feature("GraphFeature1")._set_uuid(uuid_1)._set_compute_frameworks({BaseTestComputeFrameWork1}),
            Feature("GraphFeature2")._set_uuid(uuid_2)._set_compute_frameworks({BaseTestComputeFrameWork1}),
            Feature("GraphFeature3")._set_uuid(uuid_3)._set_compute_frameworks({BaseTestComputeFrameWork1}),
            Feature("GraphFeature4")._set_uuid(uuid_4)._set_compute_frameworks({BaseTestComputeFrameWork2}),
            Feature("GraphFeature5")._set_uuid(uuid_5)._set_compute_frameworks({BaseTestComputeFrameWork3}),
            Feature("GraphFeature6")._set_uuid(uuid_6)._set_compute_frameworks({BaseTestComputeFrameWork1}),
            Feature("GraphFeature7")._set_uuid(uuid_7)._set_compute_frameworks({BaseTestComputeFrameWork1}),
        )

        graph = Graph()

        graph.add_node(
            uuid_1,
            NodeProperties(f1, BaseLinkTestFeatureGroup1),
        )
        graph.add_node(
            uuid_2,
            NodeProperties(f2, BaseTestGraphFeatureGroup3),
        )
        graph.add_node(
            uuid_3,
            NodeProperties(f3, BaseTestGraphFeatureGroup3),
        )
        graph.add_node(
            uuid_4,
            NodeProperties(f4, BaseTestGraphFeatureGroup3),
        )
        graph.add_node(
            uuid_5,
            NodeProperties(f5, BaseTestGraphFeatureGroup3),
        )
        graph.add_node(
            uuid_6,
            NodeProperties(f6, BaseTestGraphFeatureGroup3),
        )
        graph.add_node(
            uuid_7,
            NodeProperties(f7, BaseLinkTestFeatureGroup1),
        )

        graph.add_edge(
            uuid_1,
            uuid_2,
            EdgeProperties(BaseLinkTestFeatureGroup1, BaseTestGraphFeatureGroup3),
        )
        graph.add_edge(
            uuid_2,
            uuid_3,
            EdgeProperties(BaseTestGraphFeatureGroup3, BaseTestGraphFeatureGroup3),
        )
        graph.add_edge(
            uuid_2,
            uuid_4,
            EdgeProperties(BaseTestGraphFeatureGroup3, BaseTestGraphFeatureGroup3),
        )
        graph.add_edge(
            uuid_5,
            uuid_4,
            EdgeProperties(BaseTestGraphFeatureGroup3, BaseTestGraphFeatureGroup3),
        )
        graph.add_edge(
            uuid_1,
            uuid_5,
            EdgeProperties(BaseTestGraphFeatureGroup3, BaseTestGraphFeatureGroup3),
        )
        graph.add_edge(
            uuid_2,
            uuid_7,
            EdgeProperties(BaseTestGraphFeatureGroup3, BaseTestGraphFeatureGroup3),
        )
        graph.add_edge(
            uuid_6,
            uuid_7,
            EdgeProperties(BaseTestGraphFeatureGroup3, BaseTestGraphFeatureGroup3),
        )
        return graph

    def test_base_link(self) -> None:
        links = {
            Link.inner(
                JoinSpec(BaseLinkTestFeatureGroup1, Index(tuple(["Index1"]))),
                JoinSpec(BaseTestGraphFeatureGroup3, Index(tuple(["Index1"]))),
            )
        }

        graph = self.create_graph()
        resolver = ResolveGraph(graph, links)
        resolver.create_initial_queue()
        assert set(resolver.graph.roots) == set([uuid_1, uuid_6])
        assert resolver.graph.queue == [uuid_1, uuid_6, uuid_2, uuid_3, uuid_4, uuid_7, uuid_5]

        resolver.set_nodes_per_feature_group()
        assert len(resolver.nodes_per_feature_group[BaseTestGraphFeatureGroup3]) == 5
        assert len(resolver.nodes_per_feature_group[BaseLinkTestFeatureGroup1]) == 2

        queue_with_links_and_features = resolver.resolve_links()

        child_roots = resolver.graph.child_with_root
        assert len(child_roots) == 5
        assert child_roots[uuid_7] == {uuid_1, uuid_6}

        link_trekker = resolver.resolver_links.get_link_trekker().data
        assert len(link_trekker) == 2

        result_link = links.pop()
        expected_link_tuple = (result_link, BaseTestComputeFrameWork1, BaseTestComputeFrameWork1)
        assert link_trekker[expected_link_tuple] == {uuid_7, uuid_3, uuid_4}

        collector = []
        for e in queue_with_links_and_features:
            if e == expected_link_tuple:
                collector.append(e)
                continue
            if isinstance(e, tuple):
                if not isinstance(e[0], Link):
                    assert issubclass(e[0], AbstractFeatureGroup)
                    assert e[0] in [BaseLinkTestFeatureGroup1, BaseTestGraphFeatureGroup3]
                    for feature in e[1]:
                        collector.append(feature.uuid)  # type: ignore
                continue
            raise ValueError("Not a valid type")
