"""An APPEND/UNION link scheduled in an inverted orientation is a configuration error, not an internal bug.

Frameworks that agree still plan a JoinStep.
"""

from typing import Any, Callable, NamedTuple
from uuid import UUID

import pytest

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.index.index import Index
from mloda.core.abstract_plugins.components.link import JoinSpec, Link
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.core.step.join_step import JoinStep
from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.core.prepare.graph.graph import Graph
from mloda.core.prepare.graph.properties import NodeProperties
from mloda.core.prepare.resolve_links import LinkTrekker
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


LEFT_INDEX = Index(("stack_left_key",))
RIGHT_INDEX = Index(("stack_right_key",))

STACK_LINK_FACTORIES: list[Callable[[JoinSpec, JoinSpec], Link]] = [Link.append, Link.union]

# The message must name the orientation concept and say that it is not supported.
ORIENTATION_WORDS = ("invert", "revers", "swap", "orientation")
UNSUPPORTED_WORDS = ("not support", "unsupported", "cannot", "can't", "not allowed")


class StackSource(FeatureGroup):
    pass


class StackOtherSource(FeatureGroup):
    pass


class StackConsumer(FeatureGroup):
    pass


class Planned(NamedTuple):
    plan: ExecutionPlan
    link_fw: tuple[Link, type[ComputeFramework], type[ComputeFramework]]
    link_trekker: LinkTrekker
    graph: Graph
    pre_execution_plan: list[Any]
    left_uuid: UUID
    right_uuid: UUID


def _feature(name: str, cfw: type[ComputeFramework], index: Index | None = None) -> Feature:
    feature = Feature(name, index=index)
    feature.compute_frameworks = {cfw}
    return feature


def _step(fg: type[FeatureGroup], feature: Feature, cfw: type[ComputeFramework]) -> FeatureGroupStep:
    feature_set = FeatureSet()
    feature_set.add(feature)
    return FeatureGroupStep(fg, feature_set, set(), cfw)


def _plan_link(
    link_factory: Callable[[JoinSpec, JoinSpec], Link],
    *,
    right_feature_group: type[FeatureGroup] = StackSource,
    left_cfw: type[ComputeFramework] = PyArrowTable,
    right_cfw: type[ComputeFramework] = PandasDataFrame,
    child_cfw: type[ComputeFramework] = PandasDataFrame,
    orientation: tuple[type[ComputeFramework], type[ComputeFramework]] = (PandasDataFrame, PyArrowTable),
) -> Planned:
    """`orientation` is the trekked and queued left/right framework pair."""
    link = link_factory(JoinSpec(StackSource, LEFT_INDEX), JoinSpec(right_feature_group, RIGHT_INDEX))

    left = _feature("stack_left_payload", left_cfw, LEFT_INDEX)
    right = _feature("stack_right_payload", right_cfw, RIGHT_INDEX)
    child = _feature("stack_consumer", child_cfw)

    graph = Graph()
    graph.add_node(left.uuid, NodeProperties(left, StackSource))
    graph.add_node(right.uuid, NodeProperties(right, right_feature_group))
    graph.add_node(child.uuid, NodeProperties(child, StackConsumer))
    graph.adjacency_list[left.uuid] = [child.uuid]
    graph.adjacency_list[right.uuid] = [child.uuid]
    graph.adjacency_list[child.uuid] = []
    graph.parent_to_children_mapping[child.uuid] = {left.uuid, right.uuid}

    trekker_key = (link, orientation[0], orientation[1])
    link_trekker = LinkTrekker()
    link_trekker.data[trekker_key] = {child.uuid}

    pre_execution_plan: list[Any] = [
        _step(StackSource, left, left_cfw),
        _step(right_feature_group, right, right_cfw),
        _step(StackConsumer, child, child_cfw),
    ]

    plan = ExecutionPlan()
    plan.feature_set_collections = [{left.uuid}, {right.uuid}, {child.uuid}]

    return Planned(plan, trekker_key, link_trekker, graph, pre_execution_plan, left.uuid, right.uuid)


def _run(planned: Planned) -> JoinStep | None:
    return planned.plan.run_link(planned.link_fw, planned.link_trekker, planned.graph, planned.pre_execution_plan)


def _assert_is_orientation_configuration_error(message: str, link: Link) -> None:
    """The error reads as a configuration problem, not an internal bug report."""
    assert not message.startswith("Internal error:")
    assert "report this issue" not in message.lower()
    assert "sanity check" not in message
    assert str(link) in message
    assert PyArrowTable.get_class_name() in message
    assert PandasDataFrame.get_class_name() in message
    assert link.jointype.value in message.lower()
    assert any(word in message.lower() for word in ORIENTATION_WORDS)
    assert any(word in message.lower() for word in UNSUPPORTED_WORDS)


@pytest.mark.parametrize("link_factory", STACK_LINK_FACTORIES)
def test_left_framework_mismatch_is_rejected_as_a_configuration_error(
    link_factory: Callable[[JoinSpec, JoinSpec], Link],
) -> None:
    with pytest.raises(ValueError) as excinfo:
        _run(_plan_link(link_factory))

    assert not str(excinfo.value).startswith("Internal error:")


@pytest.mark.parametrize("link_factory", STACK_LINK_FACTORIES)
def test_left_framework_mismatch_names_the_link_and_both_frameworks(
    link_factory: Callable[[JoinSpec, JoinSpec], Link],
) -> None:
    planned = _plan_link(link_factory)

    with pytest.raises(ValueError) as excinfo:
        _run(planned)

    _assert_is_orientation_configuration_error(str(excinfo.value), planned.link_fw[0])


@pytest.mark.parametrize("link_factory", STACK_LINK_FACTORIES)
def test_right_framework_mismatch_names_the_link_and_both_frameworks(
    link_factory: Callable[[JoinSpec, JoinSpec], Link],
) -> None:
    planned = _plan_link(
        link_factory,
        right_feature_group=StackOtherSource,
        orientation=(PyArrowTable, PyArrowTable),
    )

    with pytest.raises(ValueError) as excinfo:
        _run(planned)

    _assert_is_orientation_configuration_error(str(excinfo.value), planned.link_fw[0])


@pytest.mark.parametrize("link_factory", STACK_LINK_FACTORIES)
def test_agreeing_frameworks_keep_planning_a_joinstep(
    link_factory: Callable[[JoinSpec, JoinSpec], Link],
) -> None:
    planned = _plan_link(
        link_factory,
        right_feature_group=StackOtherSource,
        child_cfw=PyArrowTable,
        orientation=(PyArrowTable, PandasDataFrame),
    )

    join_step = _run(planned)

    assert isinstance(join_step, JoinStep)
    assert join_step.destination_framework is PyArrowTable
    assert join_step.source_framework is PandasDataFrame
    assert join_step.destination_framework_uuids == {planned.left_uuid}
    assert join_step.source_framework_uuids == {planned.right_uuid}


@pytest.mark.parametrize("link_factory", STACK_LINK_FACTORIES)
def test_agreeing_frameworks_keep_planning_a_joinstep_for_one_feature_group(
    link_factory: Callable[[JoinSpec, JoinSpec], Link],
) -> None:
    planned = _plan_link(
        link_factory,
        right_cfw=PyArrowTable,
        child_cfw=PyArrowTable,
        orientation=(PyArrowTable, PyArrowTable),
    )

    join_step = _run(planned)

    assert isinstance(join_step, JoinStep)
    assert join_step.destination_framework is PyArrowTable
    assert join_step.source_framework is PyArrowTable
    assert join_step.destination_framework_uuids == {planned.left_uuid}
    assert join_step.source_framework_uuids == {planned.right_uuid}
