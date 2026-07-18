"""Traversal-behavior tests for mloda.core.prepare.graph.graph.Graph.

These tests pin the observable outputs of the Graph traversal helpers and
define three requirements the current recursive implementation does not meet:

1. dfs / iterate_nodes_and_edges must survive deep (long) feature chains
   without RecursionError.
2. set_direct_parents_for_each_child + set_all_parents_for_each_child must
   survive deep chains without RecursionError and must not be exponential on
   diamond-shaped DAGs.
3. The existing outputs (roots, queue order, parents_by_direct_,
   parent_to_children_mapping, child_with_root) must stay byte-for-byte the
   same as they are today (regression guard).

Graphs are built manually so node-insertion order and edge order are fully
controlled. BuildGraph is intentionally avoided because it stores nodes/edges
in sets and therefore yields nondeterministic traversal order.
"""

from uuid import UUID

from mloda.core.prepare.graph.graph import Graph
from mloda.core.prepare.graph.properties import EdgeProperties, NodeProperties
from mloda.provider import FeatureGroup
from mloda.user import Feature


def _u(i: int) -> UUID:
    return UUID(int=i)


def build_graph(node_ids: list[int], edges: list[tuple[int, int]]) -> Graph:
    """Build a Graph with deterministic node-insertion and edge order."""
    g = Graph()
    for i in node_ids:
        f = Feature(f"f{i}")
        f.uuid = _u(i)
        g.add_node(_u(i), NodeProperties(f, FeatureGroup))
    for p, c in edges:
        g.add_edge(_u(p), _u(c), EdgeProperties(FeatureGroup, FeatureGroup))
    return g


class TestGraphTraversal:
    def test_regression_seven_node_golden_values(self) -> None:
        """Characterization / regression guard.

        Passes on the current implementation and must keep passing after any
        refactor. Every asserted value was captured from today's recursive
        implementation, including the load-bearing dfs pre-order in `queue`.
        """
        node_ids = [0, 1, 2, 3, 4, 5, 6]
        edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (0, 5), (5, 6)]
        g = build_graph(node_ids, edges)

        g.iterate_nodes_and_edges()
        assert g.roots == [_u(0)]
        # dfs pre-order, load-bearing downstream:
        assert g.queue == [_u(0), _u(1), _u(3), _u(4), _u(2), _u(5), _u(6)]

        g.set_direct_parents_for_each_child()
        # parents_by_direct_ (transpose of edges):
        #   1:{0}  2:{0}  3:{1,2}  4:{3}  5:{0}  6:{5}
        assert g.parents_by_direct_[_u(1)] == {_u(0)}
        assert g.parents_by_direct_[_u(2)] == {_u(0)}
        assert g.parents_by_direct_[_u(3)] == {_u(1), _u(2)}
        assert g.parents_by_direct_[_u(4)] == {_u(3)}
        assert g.parents_by_direct_[_u(5)] == {_u(0)}
        assert g.parents_by_direct_[_u(6)] == {_u(5)}

        g.set_all_parents_for_each_child()
        # parent_to_children_mapping (all transitive ancestors):
        #   1:{0} 2:{0} 3:{0,1,2} 4:{0,1,2,3} 5:{0} 6:{0,5}
        assert g.parent_to_children_mapping[_u(1)] == {_u(0)}
        assert g.parent_to_children_mapping[_u(2)] == {_u(0)}
        assert g.parent_to_children_mapping[_u(3)] == {_u(0), _u(1), _u(2)}
        assert g.parent_to_children_mapping[_u(4)] == {_u(0), _u(1), _u(2), _u(3)}
        assert g.parent_to_children_mapping[_u(5)] == {_u(0)}
        assert g.parent_to_children_mapping[_u(6)] == {_u(0), _u(5)}

        g.set_root_parents_by_direct_()
        # every child's only root ancestor is 0
        for i in (1, 2, 3, 4, 5, 6):
            assert g.child_with_root[_u(i)] == {_u(0)}

    def test_deep_chain_dfs_no_recursion_error(self) -> None:
        """Deep linear chain must not blow the Python recursion limit in dfs.

        RED: today `dfs` recurses once per chain link, so a 2000-node chain
        raises RecursionError inside iterate_nodes_and_edges.
        """
        node_ids = list(range(2000))
        edges = [(i, i + 1) for i in range(1999)]
        g = build_graph(node_ids, edges)

        g.iterate_nodes_and_edges()
        assert g.roots == [_u(0)]
        assert len(g.queue) == 2000
        assert g.queue[:3] == [_u(0), _u(1), _u(2)]

    def test_deep_chain_ancestors_no_recursion_error(self) -> None:
        """Ancestor computation must not blow the recursion limit on deep chains.

        RED: today `set_direct_parents_for_each_child` recurses once per chain
        link, so a 2000-node chain raises RecursionError before we ever reach
        set_all_parents_for_each_child.
        """
        node_ids = list(range(2000))
        edges = [(i, i + 1) for i in range(1999)]
        g = build_graph(node_ids, edges)

        g.set_direct_parents_for_each_child()
        g.set_all_parents_for_each_child()
        assert g.parent_to_children_mapping[_u(1)] == {_u(0)}
        assert g.parent_to_children_mapping[_u(500)] == {_u(i) for i in range(500)}
        assert len(g.parent_to_children_mapping[_u(1999)]) == 1999

    def test_diamond_dag_not_exponential(self) -> None:
        """Chained diamonds must be traversed in polynomial (not exponential) time.

        The graph is `entry -> {a, b} -> exit`, chained DIAMONDS times, so the
        number of distinct root-to-leaf paths is 2**DIAMONDS.

        RED: today `set_direct_parents_for_each_child` (and
        set_all_parents_for_each_child) re-walk every path with no memoization,
        making this O(2**DIAMONDS). With DIAMONDS=30 that is ~1e9 calls and the
        method never finishes; the global pytest `--timeout=10` (or the outer
        wall-clock cap used to validate the RED state) trips instead. After a
        memoized/iterative fix this completes in milliseconds.
        """
        diamonds = 30

        node_ids = list(range(3 * diamonds + 1))
        edges: list[tuple[int, int]] = []
        entry = 0
        n = 0
        for _ in range(diamonds):
            a = n + 1
            b = n + 2
            exit_node = n + 3
            edges.append((entry, a))
            edges.append((entry, b))
            edges.append((a, exit_node))
            edges.append((b, exit_node))
            entry = exit_node
            n += 3

        g = build_graph(node_ids, edges)

        # Shallow dfs (depth ~60) mirrors the production call order and is fine.
        g.iterate_nodes_and_edges()

        # Current recursive, unmemoized implementation is exponential here.
        g.set_direct_parents_for_each_child()
        g.set_all_parents_for_each_child()

        final_exit = _u(3 * diamonds)
        # every earlier node is a transitive ancestor of the final exit node
        assert len(g.parent_to_children_mapping[final_exit]) == 3 * diamonds
