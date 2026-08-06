"""Prints one json line with the link orientation a fresh interpreter plans.
No test_ prefix, so pytest never collects it; the link planner characterization runs it as a script.
"""

import json
from typing import Any

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.link import JoinSpec, Link
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.graph.graph import Graph
from mloda.core.prepare.resolve_compute_frameworks import ResolveComputeFrameworks
from mloda.core.prepare.resolve_links import LinkTrekker, ResolveLinks
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


class LinkPlannerProbeLeft(FeatureGroup):
    pass


class LinkPlannerProbeRight(FeatureGroup):
    pass


class LinkPlannerProbeChild(FeatureGroup):
    pass


def collect() -> dict[str, str]:
    """Each side reduces from a framework set; the child then decides which reduction survives."""
    left_side: set[type[ComputeFramework]] = {PandasDataFrame, PyArrowTable}
    right_side: set[type[ComputeFramework]] = {PythonDictFramework, PyArrowTable}

    link = Link.inner(JoinSpec(LinkPlannerProbeLeft, "idx"), JoinSpec(LinkPlannerProbeRight, "idx"))
    key = ResolveLinks(Graph()).create_link_trekker_key(link, left_side, right_side)

    child = Feature("link_planner_probe_child")
    child.compute_frameworks = {PyArrowTable}

    trekker = LinkTrekker()
    trekked = {child.uuid}
    trekker.data[key] = trekked
    trekker.data_ordered[key] = trekked

    planned_queue: list[Any] = [key, (LinkPlannerProbeChild, {child})]
    ResolveComputeFrameworks(Graph()).links(planned_queue, trekker)

    orientations = [orientation for orientation in trekker.data if orientation[0] is link]
    planned = orientations[0] if len(orientations) == 1 else (link, ComputeFramework, ComputeFramework)

    return {
        "child": child.get_compute_framework().get_class_name(),
        "orientation_count": str(len(orientations)),
        "planned_left": planned[1].get_class_name(),
        "planned_right": planned[2].get_class_name(),
        "trekker_left": key[1].get_class_name(),
        "trekker_right": key[2].get_class_name(),
    }


if __name__ == "__main__":
    print(json.dumps(collect(), sort_keys=True))
