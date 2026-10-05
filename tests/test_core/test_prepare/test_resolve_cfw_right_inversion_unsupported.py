"""A RIGHT join whose child cannot run on the right side is infeasible for the chooser."""

import pytest

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.link import JoinSpec, Link
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.choose_compute_frameworks import ChooseComputeFrameworks
from mloda.core.prepare.graph.graph import Graph
from mloda.core.prepare.graph.properties import EdgeProperties, NodeProperties
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


class RightInversionLeftFG(FeatureGroup):
    pass


class RightInversionRightFG(FeatureGroup):
    pass


class RightInversionChildFG(FeatureGroup):
    pass


def test_right_join_onto_an_undeclared_framework_is_infeasible() -> None:
    """The child declares only PandasDataFrame, but a RIGHT join lands on its right side, PyArrowTable."""
    left = Feature("right_inversion_left")
    right = Feature("right_inversion_right")
    child = Feature("right_inversion_feature")
    left.compute_frameworks = {PandasDataFrame}
    right.compute_frameworks = {PyArrowTable}
    child.compute_frameworks = {PandasDataFrame}

    graph = Graph()
    graph.add_node(left.uuid, NodeProperties(left, RightInversionLeftFG))
    graph.add_node(right.uuid, NodeProperties(right, RightInversionRightFG))
    graph.add_node(child.uuid, NodeProperties(child, RightInversionChildFG))
    graph.add_edge(left.uuid, child.uuid, EdgeProperties(RightInversionLeftFG, RightInversionChildFG))
    graph.add_edge(right.uuid, child.uuid, EdgeProperties(RightInversionRightFG, RightInversionChildFG))

    link = Link.right(JoinSpec(RightInversionLeftFG, "idx"), JoinSpec(RightInversionRightFG, "idx"))
    nodes: dict[type[FeatureGroup], set[Feature]] = {
        RightInversionLeftFG: {left},
        RightInversionRightFG: {right},
        RightInversionChildFG: {child},
    }
    occurrences = [(link, left.uuid, right.uuid, child.uuid)]

    with pytest.raises(ValueError, match="No compute framework assignment") as excinfo:
        ChooseComputeFrameworks(graph, nodes, occurrences, [], {}).choose()

    message = str(excinfo.value)
    assert str(child.name) in message, message
    assert PandasDataFrame.get_class_name() in message, message
    assert "join right" in message, message
