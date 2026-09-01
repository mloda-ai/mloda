from collections import defaultdict
from collections.abc import Iterator
from copy import copy
from uuid import UUID

from mloda.core.prepare.graph.properties import EdgeProperties, NodeProperties


class Graph:
    def __init__(self) -> None:
        self.nodes: defaultdict[UUID, NodeProperties] = defaultdict(lambda: NodeProperties(None, None))  # type: ignore[arg-type]
        self.edges: defaultdict[tuple[UUID, UUID], EdgeProperties] = defaultdict(lambda: EdgeProperties(None, None))  # type: ignore[arg-type]

        self.adjacency_list: dict[UUID, list[UUID]] = defaultdict(list)

        self.roots: list[UUID] = []
        self.queue: list[UUID] = []

        # track parent children relations easier
        self.parents_by_direct_: dict[UUID, set[UUID]] = defaultdict(set)

        self.parent_to_children_mapping: dict[UUID, set[UUID]] = defaultdict(set)
        self.child_with_root: dict[UUID, set[UUID]] = defaultdict(set)

    def add_node(self, node: UUID, node_properties: NodeProperties) -> None:
        self.nodes[node] = node_properties

    def add_edge(self, parent: UUID, child: UUID, edge_properties: EdgeProperties) -> None:
        self.edges[(parent, child)] = edge_properties

        self.adjacency_list[parent].append(child)

    def get_nodes(self) -> dict[UUID, NodeProperties]:
        return self.nodes

    def get_edges(self) -> dict[tuple[UUID, UUID], EdgeProperties]:
        return self.edges

    def dfs(self, node: UUID) -> None:
        if node in self.visited:
            return
        self.visited.add(node)
        stack: list[tuple[UUID, Iterator[UUID]]] = [(node, iter(self.adjacency_list[node]))]
        while stack:
            _, children = stack[-1]
            for child in children:
                if child in self.visited:
                    continue
                self.queue.append(child)
                self.visited.add(child)
                stack.append((child, iter(self.adjacency_list[child])))
                break
            else:
                stack.pop()

    def iterate_nodes_and_edges(self) -> None:
        self.visited: set[UUID] = set()

        # Start DFS from each node with in-degree 0
        in_degree = self.create_in_degree()

        self.roots = [node for node in self.nodes if in_degree[node] == 0]
        self.queue = copy(self.roots)
        for root in self.roots:
            self.dfs(root)

    def create_in_degree(self) -> dict[UUID, int]:
        in_degree: dict[UUID, int] = defaultdict(int)

        for parent, children in self.adjacency_list.items():
            for child in children:
                in_degree[child] += 1
        return in_degree

    def set_direct_parents_for_each_child(self) -> None:
        for parent, children in self.adjacency_list.items():
            for child in children:
                self.parents_by_direct_[child].add(parent)

    def set_all_parents_for_each_child(self) -> None:
        memo: dict[UUID, set[UUID]] = {}
        direct = self.parents_by_direct_
        for start in list(direct):
            if start in memo:
                continue
            stack: list[UUID] = [start]
            on_path: set[UUID] = set()
            while stack:
                cur = stack[-1]
                if cur in memo:
                    stack.pop()
                    continue
                # .get, not direct[cur]: indexing the defaultdict would insert spurious root keys
                parents = direct.get(cur, set())
                unresolved = [p for p in parents if p not in memo]
                if unresolved:
                    if cur in on_path:
                        raise ValueError(f"cycle detected in feature graph at {cur}")
                    on_path.add(cur)
                    stack.extend(unresolved)
                    continue
                ancestors: set[UUID] = set()
                for p in parents:
                    ancestors.add(p)
                    ancestors |= memo[p]
                memo[cur] = ancestors
                on_path.discard(cur)
                stack.pop()
        for child in direct:
            self.parent_to_children_mapping[child] = memo[child]

    def set_root_parents_by_direct_(self) -> None:
        for child, parents in self.parent_to_children_mapping.items():
            for parent in parents:
                if parent in self.roots:
                    self.child_with_root[child].add(parent)
