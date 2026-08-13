"""Prints one json line with the resolved join record a fresh interpreter builds.
No test_ prefix, so pytest never collects it; the record tests run it as a script.
"""

import json
from typing import Any

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.index.index import Index
from mloda.core.abstract_plugins.components.link import JoinSpec, Link
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.core.prepare.graph.graph import Graph
from mloda.core.prepare.graph.properties import NodeProperties
from mloda.core.prepare.resolve_links import LinkTrekker, ResolveLinks
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


PROBE_LEFT_INDEX = Index(("resolved_join_probe_left_key",))
PROBE_RIGHT_INDEX = Index(("resolved_join_probe_right_key",))


class ResolvedJoinProbeLeft(FeatureGroup):
    pass


class ResolvedJoinProbeRight(FeatureGroup):
    pass


class ResolvedJoinProbeChild(FeatureGroup):
    pass


def _feature(name: str, frameworks: set[type[ComputeFramework]], index: Index | None = None) -> Feature:
    feature = Feature(name, index=index)
    feature.compute_frameworks = frameworks
    return feature


def collect() -> dict[str, str]:
    """Each side reduces from a framework set, and the record has to name the same sides the reduction picked."""
    link = Link.inner(
        JoinSpec(ResolvedJoinProbeLeft, PROBE_LEFT_INDEX), JoinSpec(ResolvedJoinProbeRight, PROBE_RIGHT_INDEX)
    )
    left = _feature("resolved_join_probe_left_payload", {PandasDataFrame, PyArrowTable}, PROBE_LEFT_INDEX)
    right = _feature("resolved_join_probe_right_payload", {PythonDictFramework, PyArrowTable}, PROBE_RIGHT_INDEX)
    child = _feature("resolved_join_probe_child_payload", {PandasDataFrame})
    key = ResolveLinks(Graph()).create_link_trekker_key(link, left.compute_frameworks, right.compute_frameworks)

    graph = Graph()
    graph.add_node(left.uuid, NodeProperties(left, ResolvedJoinProbeLeft))
    graph.add_node(right.uuid, NodeProperties(right, ResolvedJoinProbeRight))
    graph.add_node(child.uuid, NodeProperties(child, ResolvedJoinProbeChild))
    graph.adjacency_list[left.uuid] = [child.uuid]
    graph.adjacency_list[right.uuid] = [child.uuid]
    graph.adjacency_list[child.uuid] = []
    graph.parent_to_children_mapping[child.uuid] = {left.uuid, right.uuid}

    trekker = LinkTrekker()
    trekked = {child.uuid}
    trekker.data[key] = trekked
    trekker.data_ordered[key] = trekked

    queue: list[Any] = [
        (ResolvedJoinProbeLeft, {left}),
        (ResolvedJoinProbeRight, {right}),
        key,
        (ResolvedJoinProbeChild, {child}),
    ]

    plan = ExecutionPlan()
    plan.create_execution_plan(queue, graph, trekker)

    resolved = plan.resolved_join_plan
    signature = resolved.records[0].signature(resolved.link_of_token())

    return {
        "declined_count": str(len(resolved.declined)),
        "depends_on_count": str(len(signature.depends_on_links)),
        "destination_framework": signature.destination_framework,
        "destination_is_declared_left": str(signature.destination_uuids == (str(left.uuid),)),
        "destination_side": signature.destination_side,
        "jointype": signature.jointype,
        "record_count": str(len(resolved.records)),
        "source_framework": signature.source_framework,
        "source_is_declared_right": str(signature.source_uuids == (str(right.uuid),)),
        "trekker_left": key[1].get_class_name(),
        "trekker_right": key[2].get_class_name(),
    }


if __name__ == "__main__":
    print(json.dumps(collect(), sort_keys=True))
