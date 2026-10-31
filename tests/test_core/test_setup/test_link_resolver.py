import random
from collections.abc import Sequence
import uuid

import pytest
from mloda.core.prepare.resolve_links import ResolveLinks
from mloda.core.prepare.graph.graph import Graph
from mloda.core.prepare.graph.properties import EdgeProperties, NodeProperties
from mloda.core.prepare.choose_compute_frameworks import ChooseComputeFrameworks
from mloda.core.prepare.resolve_graph import ResolveGraph
from mloda.provider import FeatureGroup
from mloda.user import Feature
from mloda.user import Index
from mloda.user import Link, JoinSpec
from tests.test_core.test_setup.test_graph_builder import BaseTestGraphFeatureGroup3
from tests.test_core.test_abstract_plugins.test_abstract_feature_group import BaseTestFeatureGroup1
from tests.test_core.test_abstract_plugins.test_abstract_compute_framework import (
    BaseTestComputeFramework1,
    BaseTestComputeFramework2,
    BaseTestComputeFramework3,
)


class BaseLinkTestFeatureGroup1(BaseTestFeatureGroup1):
    pass


class LinkChildFeatureGroup(FeatureGroup):
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
        f1 = Feature("GraphFeature1")
        f1.uuid = uuid_1
        f1.compute_frameworks = {BaseTestComputeFramework1}
        f2 = Feature("GraphFeature2")
        f2.uuid = uuid_2
        f2.compute_frameworks = {BaseTestComputeFramework1}
        f3 = Feature("GraphFeature3")
        f3.uuid = uuid_3
        f3.compute_frameworks = {BaseTestComputeFramework1}
        f4 = Feature("GraphFeature4")
        f4.uuid = uuid_4
        f4.compute_frameworks = {BaseTestComputeFramework1, BaseTestComputeFramework2}
        f5 = Feature("GraphFeature5")
        f5.uuid = uuid_5
        f5.compute_frameworks = {BaseTestComputeFramework3}
        f6 = Feature("GraphFeature6")
        f6.uuid = uuid_6
        f6.compute_frameworks = {BaseTestComputeFramework1}
        f7 = Feature("GraphFeature7")
        f7.uuid = uuid_7
        f7.compute_frameworks = {BaseTestComputeFramework1}

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

    def test_base_link(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The synthetic frameworks have no transformer; this test is about link keys, not conversion paths.
        monkeypatch.setattr(ChooseComputeFrameworks, "_convertible", lambda self, source, target: True)
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
        result_link = links.pop()
        expected_link_tuple = (result_link, BaseTestComputeFramework1, BaseTestComputeFramework1)
        assert len(link_trekker) == 1
        assert link_trekker[expected_link_tuple] == {uuid_7}

        collector = []
        for e in queue_with_links_and_features:
            if e == expected_link_tuple:
                collector.append(e)
                continue
            if isinstance(e, tuple):
                if not isinstance(e[0], Link):
                    assert issubclass(e[0], FeatureGroup)
                    assert e[0] in [BaseLinkTestFeatureGroup1, BaseTestGraphFeatureGroup3]
                    for feature in e[1]:
                        collector.append(feature.uuid)  # type: ignore
                continue
            raise ValueError("Not a valid type")

    @staticmethod
    def node(graph: Graph, uid: uuid.UUID, name: str, fg: type[FeatureGroup]) -> None:
        feature = Feature(name)
        feature.uuid = uid
        feature.compute_frameworks = {BaseTestComputeFramework1}
        graph.add_node(uid, NodeProperties(feature, fg))

    @staticmethod
    def recorded_link_consumers(graph: Graph) -> set[uuid.UUID]:
        """Resolve the Index1 link between the two test root groups; return all uuids the trekker recorded."""
        link = Link.inner(
            JoinSpec(BaseLinkTestFeatureGroup1, Index(tuple(["Index1"]))),
            JoinSpec(BaseTestGraphFeatureGroup3, Index(tuple(["Index1"]))),
        )
        resolver = ResolveGraph(graph, {link})
        resolver.create_initial_queue()
        resolver.set_nodes_per_feature_group()
        resolver.resolve_links()
        return set().union(*resolver.resolver_links.get_link_trekker().data.values())

    def test_link_child_recorded_but_not_its_grandchild(self) -> None:
        graph = Graph()
        self.node(graph, uuid_1, "LeftRoot", BaseLinkTestFeatureGroup1)
        self.node(graph, uuid_2, "RightRoot", BaseTestGraphFeatureGroup3)
        self.node(graph, uuid_3, "LinkChild", LinkChildFeatureGroup)
        self.node(graph, uuid_4, "Grandchild", LinkChildFeatureGroup)
        for parent, child in [(uuid_1, uuid_3), (uuid_2, uuid_3), (uuid_3, uuid_4)]:
            graph.add_edge(parent, child, EdgeProperties(LinkChildFeatureGroup, LinkChildFeatureGroup))

        assert self.recorded_link_consumers(graph) == {uuid_3}

    def test_consumer_reading_both_sides_directly_stays_recorded_when_one_is_the_others_ancestor(self) -> None:
        graph = Graph()
        self.node(graph, uuid_1, "A", BaseLinkTestFeatureGroup1)
        self.node(graph, uuid_2, "D", BaseTestGraphFeatureGroup3)
        self.node(graph, uuid_3, "C", LinkChildFeatureGroup)
        edge = EdgeProperties(LinkChildFeatureGroup, LinkChildFeatureGroup)
        for parent, child in [(uuid_1, uuid_2), (uuid_1, uuid_3), (uuid_2, uuid_3)]:
            graph.add_edge(parent, child, edge)

        assert self.recorded_link_consumers(graph) == {uuid_3}


def _pair_link() -> Link:
    return Link.inner(
        JoinSpec(BaseLinkTestFeatureGroup1, Index(tuple(["Index1"]))),
        JoinSpec(BaseTestGraphFeatureGroup3, Index(tuple(["Index1"]))),
    )


def _reference_pairs(
    resolver: ResolveLinks, child: uuid.UUID, parents: set[uuid.UUID]
) -> list[tuple[uuid.UUID, uuid.UUID]]:
    """Pair selection of the original double loop."""
    graph = resolver.graph
    direct_parents = graph.parents_by_direct_.get(child, set())
    closures = [{d} | graph.parent_to_children_mapping.get(d, set()) for d in direct_parents]
    ordered_parents = sorted(parents, key=resolver._parent_sort_key)
    result: list[tuple[uuid.UUID, uuid.UUID]] = []
    for parent_in in ordered_parents:
        for parent_out in ordered_parents:
            if parent_in == parent_out:
                continue
            reads_both = parent_in in direct_parents and parent_out in direct_parents
            if not reads_both and any(parent_in in c and parent_out in c for c in closures):
                continue
            result.append((parent_in, parent_out))
    return result


def _build_graph(nodes: Sequence[tuple[str, type[FeatureGroup]]], edges: list[tuple[int, int]]) -> Graph:
    graph = Graph()
    for i, (name, fg) in enumerate(nodes, start=1):
        TestResolveGraph.node(graph, uuid.UUID(int=i), name, fg)
    for parent, child in edges:
        graph.add_edge(
            uuid.UUID(int=parent + 1),
            uuid.UUID(int=child + 1),
            EdgeProperties(LinkChildFeatureGroup, LinkChildFeatureGroup),
        )
    graph.set_direct_parents_for_each_child()
    graph.set_all_parents_for_each_child()
    return graph


_FGS: list[type[FeatureGroup]] = [BaseLinkTestFeatureGroup1, BaseTestGraphFeatureGroup3, LinkChildFeatureGroup]


def _random_dag(seed: int) -> Graph:
    rng = random.Random(seed)  # nosec B311
    n = rng.randint(15, 25)
    nodes = [(f"N{rng.randint(0, 4)}", rng.choice(_FGS)) for _ in range(n)]
    edges = [(i, j) for j in range(n) for i in range(j) if rng.random() < 0.2]
    return _build_graph(nodes, edges)


_A, _B, _C = BaseLinkTestFeatureGroup1, BaseTestGraphFeatureGroup3, LinkChildFeatureGroup

_HAND_BUILT: dict[str, tuple[list[tuple[str, type[FeatureGroup]]], list[tuple[int, int]]]] = {
    "diamond": ([("a", _A), ("b", _B), ("c", _B), ("d", _C)], [(0, 1), (0, 2), (1, 3), (2, 3)]),
    "three_direct_shared_and_private": (
        [("r", _A), ("p1", _B), ("p2", _A), ("p3", _B), ("x", _A), ("y", _B), ("c", _C)],
        [(0, 1), (0, 2), (0, 3), (4, 1), (5, 3), (1, 6), (2, 6), (3, 6)],
    ),
    "direct_parent_is_ancestor_of_other": (
        [("a", _A), ("b", _B), ("c", _B), ("d", _C)],
        [(0, 1), (1, 2), (0, 3), (1, 3), (2, 3)],
    ),
    "two_disjoint_deep_branches": (
        [("a0", _A), ("a1", _B), ("a2", _A), ("b0", _B), ("b1", _A), ("b2", _B), ("c", _C)],
        [(0, 1), (1, 2), (3, 4), (4, 5), (2, 6), (5, 6)],
    ),
    "tied_sort_keys": (
        [("same", _A), ("same", _A), ("same", _B), ("same", _B), ("same", _C), ("same", _C)],
        [(0, 4), (1, 4), (2, 4), (3, 4), (0, 5), (4, 5), (3, 5)],
    ),
}


class TestLinkCandidatePairs:
    @staticmethod
    def _assert_equivalent(graph: Graph) -> None:
        resolver = ResolveLinks(graph, {_pair_link()})
        assert graph.parent_to_children_mapping
        for child, parents in graph.parent_to_children_mapping.items():
            expected = _reference_pairs(resolver, child, parents)
            assert resolver._link_candidate_pairs(child, parents) == expected

    @pytest.mark.parametrize("case", sorted(_HAND_BUILT))
    def test_hand_built_graphs_match_reference(self, case: str) -> None:
        nodes, edges = _HAND_BUILT[case]
        self._assert_equivalent(_build_graph(nodes, edges))

    @pytest.mark.parametrize("seed", range(8))
    def test_random_dags_match_reference(self, seed: int) -> None:
        self._assert_equivalent(_random_dag(seed))

    @staticmethod
    def _count_sort_key_calls(graph: Graph, monkeypatch: pytest.MonkeyPatch) -> int:
        calls: list[uuid.UUID] = []
        original = ResolveLinks._parent_sort_key

        def spy(self: ResolveLinks, parent: uuid.UUID) -> tuple[str, str, str]:
            calls.append(parent)
            return original(self, parent)

        monkeypatch.setattr(ResolveLinks, "_parent_sort_key", spy)
        ResolveLinks(graph, {_pair_link()}).match_links()
        return len(calls)

    def test_pure_chain_makes_no_sort_key_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        n = 80
        nodes = [(f"n{i}", _A if i % 2 else _B) for i in range(n)]
        graph = _build_graph(nodes, [(i, i + 1) for i in range(n - 1)])
        assert self._count_sort_key_calls(graph, monkeypatch) == 0

    def test_chain_with_shared_root_is_linear_in_sort_key_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        n = 80
        nodes = [("root", _A)] + [(f"n{i}", _A if i % 2 else _B) for i in range(n)]
        edges = [(0, i) for i in range(1, n + 1)] + [(i, i + 1) for i in range(1, n)]
        graph = _build_graph(nodes, edges)
        assert self._count_sort_key_calls(graph, monkeypatch) <= 4 * (n + 1)
