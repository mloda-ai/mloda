"""Prints one json line with the compute frameworks a fresh interpreter reduces to.
No test_ prefix, so pytest never collects it; the determinism test runs it as a script.
"""

import json

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.link import JoinSpec, Link
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.graph.graph import Graph
from mloda.core.prepare.resolve_links import ResolveLinks
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


class ProbeLeftFeatureGroup(FeatureGroup):
    pass


class ProbeRightFeatureGroup(FeatureGroup):
    pass


def collect() -> dict[str, str]:
    """Both frameworks on both sides, so the reduction is the only thing deciding."""
    # This pair is the one whose id order actually moves between interpreters, so it is the one worth probing.
    both: set[type[ComputeFramework]] = {PythonDictFramework, PyArrowTable}

    link = Link.inner(JoinSpec(ProbeLeftFeatureGroup, "idx"), JoinSpec(ProbeRightFeatureGroup, "idx"))
    _, trekker_left, trekker_right = ResolveLinks(Graph()).create_link_trekker_key(link, set(both), set(both))

    feature = Feature("determinism_probe_feature")
    feature.compute_frameworks = set(both)

    return {
        "feature": feature.get_compute_framework().get_class_name(),
        "trekker_left": trekker_left.get_class_name(),
        "trekker_right": trekker_right.get_class_name(),
    }


if __name__ == "__main__":
    print(json.dumps(collect(), sort_keys=True))
