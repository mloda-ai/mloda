"""Prints one json line with the frameworks ChooseComputeFrameworks picks in a fresh interpreter.
No test_ prefix, so pytest never collects it; the chooser determinism test runs it as a script.
"""

import json

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.choose_compute_frameworks import ChooseComputeFrameworks
from mloda.core.prepare.graph.graph import Graph
from mloda.core.prepare.graph.properties import EdgeProperties, NodeProperties
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


class ProbeXFG(FeatureGroup):
    pass


class ProbeYFG(FeatureGroup):
    pass


class ProbeKpFG(FeatureGroup):
    pass


class ProbeKaFG(FeatureGroup):
    pass


def collect() -> dict[str, str]:
    """x feeds y and kp, y feeds ka; x and y are free, kp is Pandas-only, ka is PyArrow-only (a tied optimum)."""
    both: set[type[ComputeFramework]] = {PandasDataFrame, PyArrowTable}
    graph = Graph()
    nodes: dict[type[FeatureGroup], set[Feature]] = {}
    built: dict[str, tuple[Feature, type[FeatureGroup]]] = {}
    specs: list[tuple[str, type[FeatureGroup], set[type[ComputeFramework]]]] = [
        ("x", ProbeXFG, both),
        ("y", ProbeYFG, both),
        ("kp", ProbeKpFG, {PandasDataFrame}),
        ("ka", ProbeKaFG, {PyArrowTable}),
    ]
    for name, fg, allowed in specs:
        feature = Feature(f"probe_{name}")
        feature.compute_frameworks = set(allowed)
        graph.add_node(feature.uuid, NodeProperties(feature, fg))
        nodes.setdefault(fg, set()).add(feature)
        built[name] = (feature, fg)
    for parent, child in [("x", "y"), ("x", "kp"), ("y", "ka")]:
        graph.add_edge(built[parent][0].uuid, built[child][0].uuid, EdgeProperties(built[parent][1], built[child][1]))

    ChooseComputeFrameworks(graph, nodes, [], [], {}).choose()

    result: dict[str, str] = {}
    for name, (feature, _) in built.items():
        assert feature.chosen_compute_framework is not None
        result[name] = feature.chosen_compute_framework.get_class_name()
    return result


if __name__ == "__main__":
    print(json.dumps(collect(), sort_keys=True))
