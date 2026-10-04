"""Prints one json line with the frameworks and reasons ChooseComputeFrameworks picks in a fresh interpreter.
No test_ prefix, so pytest never collects it; the chooser determinism test runs it as a script.
"""

import itertools
import json
import os

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.choose_compute_frameworks import ChooseComputeFrameworks
from mloda.core.prepare.graph.graph import Graph
from mloda.core.prepare.graph.properties import EdgeProperties, NodeProperties
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


class ProbeObjFG(FeatureGroup):
    pass


class ProbeSrcFG(FeatureGroup):
    pass


class ProbeSnkFG(FeatureGroup):
    pass


class ProbeLeafFG(FeatureGroup):
    pass


class Opaque:
    """Default object repr, so its text carries an address that differs between interpreters."""

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


class ProbeXFG(FeatureGroup):
    pass


class ProbeYFG(FeatureGroup):
    pass


class ProbeKpFG(FeatureGroup):
    pass


class ProbeKaFG(FeatureGroup):
    pass


def collect_address_options() -> dict[str, str]:
    """Three blocks equal but for an option object's address; the seed permutes their allocation order."""
    seed = int(os.environ.get("PYTHONHASHSEED", "0"))
    permutation = list(itertools.permutations(range(3)))[seed % 6]
    objects = {index: Opaque() for index in permutation}
    allowed: set[type[ComputeFramework]] = {PandasDataFrame, PyArrowTable, PythonDictFramework}
    graph = Graph()
    nodes: dict[type[FeatureGroup], set[Feature]] = {}
    built: dict[str, tuple[Feature, type[FeatureGroup]]] = {}
    specs: list[tuple[str, type[FeatureGroup], set[type[ComputeFramework]], dict[str, Opaque] | None]] = [
        ("o0", ProbeObjFG, allowed, {"o": objects[0]}),
        ("o1", ProbeObjFG, allowed, {"o": objects[1]}),
        ("o2", ProbeObjFG, allowed, {"o": objects[2]}),
        ("src", ProbeSrcFG, {PandasDataFrame}, None),
        ("snk", ProbeSnkFG, {PythonDictFramework}, None),
        ("leaf", ProbeLeafFG, {PyArrowTable}, None),
    ]
    for name, fg, domain, options in specs:
        feature = Feature(f"probe_{name}", options=options)
        feature.compute_frameworks = set(domain)
        graph.add_node(feature.uuid, NodeProperties(feature, fg))
        nodes.setdefault(fg, set()).add(feature)
        built[name] = (feature, fg)
    edges = [("o0", "o1"), ("o0", "src"), ("o0", "leaf"), ("o1", "o2"), ("o1", "snk"), ("o2", "snk"), ("o2", "leaf")]
    for parent, child in edges:
        graph.add_edge(built[parent][0].uuid, built[child][0].uuid, EdgeProperties(built[parent][1], built[child][1]))

    ChooseComputeFrameworks(graph, nodes, [], [], {}).choose()

    result: dict[str, str] = {}
    for name, (feature, _) in built.items():
        assert feature.chosen_compute_framework is not None
        result[name] = feature.chosen_compute_framework.get_class_name()
        result[f"reason_{name}"] = str(feature.chosen_compute_framework_reason)
    return result


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
        result[f"reason_{name}"] = str(feature.chosen_compute_framework_reason)
    return result


if __name__ == "__main__":
    merged = {**collect(), **{f"addr_{name}": value for name, value in collect_address_options().items()}}
    print(json.dumps(merged, sort_keys=True))
