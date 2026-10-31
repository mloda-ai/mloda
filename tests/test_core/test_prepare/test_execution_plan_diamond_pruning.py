"""Graph reduction helpers (reduce_children_to_one_level, direct_ancestors) on diamonds and redundant edges."""

import random
from collections import defaultdict
from collections.abc import Callable
from uuid import UUID, uuid4

import pytest

from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.core.prepare.graph.graph import Graph


def test_reduce_children_to_one_level_prunes_a_grandchild_shared_by_two_children() -> None:
    plan = ExecutionPlan()
    graph = Graph()

    child_1 = uuid4()
    child_2 = uuid4()
    shared_grandchild = uuid4()

    graph.adjacency_list[child_1] = [shared_grandchild]
    graph.adjacency_list[child_2] = [shared_grandchild]

    children_uuids = {child_1, child_2, shared_grandchild}

    result = plan.reduce_children_to_one_level(children_uuids, graph)

    assert result == {child_1, child_2}


def _closed(parents: dict[UUID, set[UUID]]) -> dict[UUID, set[UUID]]:
    """Transitively close a direct-parent mapping into node -> all ancestors."""
    closed: dict[UUID, set[UUID]] = {}

    def ancestors(node: UUID) -> set[UUID]:
        if node not in closed:
            acc: set[UUID] = set()
            for p in parents.get(node, set()):
                acc |= {p} | ancestors(p)
            closed[node] = acc
        return closed[node]

    for node in parents:
        ancestors(node)
    return {k: v for k, v in closed.items() if v}


def _redundant_edge() -> tuple[dict[UUID, set[UUID]], UUID, set[UUID]]:
    a, b, c = uuid4(), uuid4(), uuid4()
    return _closed({b: {a}, c: {a, b}}), c, {b}


def _diamond() -> tuple[dict[UUID, set[UUID]], UUID, set[UUID]]:
    a, b, c, d = uuid4(), uuid4(), uuid4(), uuid4()
    return _closed({b: {a}, c: {a}, d: {b, c}}), d, {b, c}


def _chain() -> tuple[dict[UUID, set[UUID]], UUID, set[UUID]]:
    nodes = [uuid4() for _ in range(5)]
    return _closed({nodes[i]: {nodes[i - 1]} for i in range(1, 5)}), nodes[4], {nodes[3]}


@pytest.mark.parametrize("build", [_redundant_edge, _diamond, _chain], ids=["redundant_edge", "diamond", "chain"])
def test_direct_ancestors_matches_the_transitive_reduction_formula(
    build: Callable[[], tuple[dict[UUID, set[UUID]], UUID, set[UUID]]],
) -> None:
    mapping, node, expected = build()
    assert ExecutionPlan.direct_ancestors(node, mapping) == expected


def test_direct_ancestors_of_a_root_is_empty_and_does_not_mutate_the_mapping() -> None:
    mapping: defaultdict[UUID, set[UUID]] = defaultdict(set)
    assert ExecutionPlan.direct_ancestors(uuid4(), mapping) == set()
    assert len(mapping) == 0


def test_direct_ancestors_equals_the_formula_on_a_seeded_random_closed_dag() -> None:
    rng = random.Random(7)  # nosec B311
    nodes = [uuid4() for _ in range(25)]
    parents = {n: {p for p in nodes[:i] if rng.random() < 0.25} for i, n in enumerate(nodes)}
    mapping = _closed(parents)
    for node in nodes:
        anc = mapping.get(node, set())
        assert ExecutionPlan.direct_ancestors(node, mapping) == anc - {p for a in anc for p in mapping.get(a, set())}
