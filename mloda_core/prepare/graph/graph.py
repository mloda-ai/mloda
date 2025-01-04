from collections import defaultdict
from copy import copy
from typing import Dict, List, Set
from uuid import UUID

from mloda_core.prepare.graph.properties import EdgeProperties, NodeProperties


class Graph:
    def __init__(self) -> None:
        self.nodes: Dict[UUID, NodeProperties] = defaultdict(lambda: NodeProperties(None, None))  # type: ignore
        self.edges: Dict[tuple[UUID, UUID], EdgeProperties] = defaultdict(lambda: EdgeProperties(None, None))  # type: ignore

        self.adjacency_list: Dict[UUID, list[UUID]] = defaultdict(list)

        self.roots: List[UUID] = []
        self.queue: List[UUID] = []

        # track parent children relations easier
        self.parents_by_direct_: Dict[UUID, Set[UUID]] = defaultdict(set)

        self.parent_to_children_mapping: Dict[UUID, Set[UUID]] = defaultdict(set)
        self.child_with_root: Dict[UUID, Set[UUID]] = defaultdict(set)

    def add_node(self, node: UUID, node_properties: NodeProperties) -> None:
        self.nodes[node] = node_properties

    def add_edge(self, parent: UUID, child: UUID, edge_properties: EdgeProperties) -> None:
        self.edges[(parent, child)] = edge_properties

        self.adjacency_list[parent].append(child)

    def get_nodes(self) -> Dict[UUID, NodeProperties]:
        return self.nodes

    def get_edges(self) -> Dict[tuple[UUID, UUID], EdgeProperties]:
        return self.edges

    def dfs(self, node: UUID) -> None:
        if node in self.visited:
            return
        self.visited.add(node)

        # Iterate through dependent edges
        for child in self.adjacency_list[node]:
            if child not in self.visited:
                self.queue.append(child)

            self.dfs(child)

    def iterate_nodes_and_edges(self) -> None:
        self.visited: Set[UUID] = set()

        # Start DFS from each node with in-degree 0
        in_degree = self.create_in_degree()

        self.roots = [node for node in self.nodes if in_degree[node] == 0]
        self.queue = copy(self.roots)
        for root in self.roots:
            self.dfs(root)

    def create_in_degree(self) -> Dict[UUID, int]:
        in_degree: Dict[UUID, int] = defaultdict(int)

        for parent, children in self.adjacency_list.items():
            for child in children:
                in_degree[child] += 1
        return in_degree

    def get_direct_parents_for_each_child(self, parent: UUID, children: List[UUID]) -> None:
        for child in children:
            self.parents_by_direct_[child].add(parent)
            self.get_direct_parents_for_each_child(child, self.adjacency_list[child])

    def set_direct_parents_for_each_child(self) -> None:
        for parent, children in self.adjacency_list.items():
            self.get_direct_parents_for_each_child(parent, children)

    def get_all_parents_for_each_child(self, child: UUID, parents: Set[UUID]) -> Set[UUID]:
        if not parents:
            return parents

        result_set: Set[UUID] = set()

        for parent in parents:
            parents_of_parent = self.parents_by_direct_[parent]
            result_set = result_set.union(self.get_all_parents_for_each_child(child, parents_of_parent))

        return parents.union(result_set)

    def set_all_parents_for_each_child(self) -> None:
        for child, parents in self.parents_by_direct_.copy().items():
            result_set = self.get_all_parents_for_each_child(child, parents)
            self.parent_to_children_mapping[child] = result_set.union(parents)

    def set_root_parents_by_direct_(self) -> None:
        for child, parents in self.parent_to_children_mapping.items():
            for parent in parents:
                if parent in self.roots:
                    self.child_with_root[child].add(parent)
