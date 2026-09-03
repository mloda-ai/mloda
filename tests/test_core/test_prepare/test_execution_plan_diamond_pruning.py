"""reduce_children_to_one_level must survive a shared grandchild reached from two children."""

from uuid import uuid4

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
