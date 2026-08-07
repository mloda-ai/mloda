"""Characterizes what the link planner plans, and what it declines to plan."""

from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional
from uuid import UUID

import pyarrow as pa
import pytest

from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.core.step.join_step import JoinStep
from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.core.prepare.graph.graph import Graph
from mloda.core.prepare.graph.properties import NodeProperties
from mloda.core.prepare.resolve_compute_frameworks import ResolveComputeFrameworks
from mloda.core.prepare.resolve_links import LinkFrameworkTrekker, LinkTrekker
from mloda.provider import BaseInputData
from mloda.provider import ComputeFramework
from mloda.provider import DataCreator
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Index
from mloda.user import JoinSpec, JoinType, Link
from mloda.user import Options
from mloda.user import ParallelizationMode
from mloda.user import PluginCollector
from mloda.user import mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from tests.helpers.probe_runner import run_probes


SHARED_LEFT_KEY = "link_plan_shared_left_key"
SHARED_LEFT_PAYLOAD = "link_plan_shared_left_payload"
SHARED_RIGHT_KEY = "link_plan_shared_right_key"
SHARED_RIGHT_PAYLOAD = "link_plan_shared_right_payload"

SELF_SIDE = "link_plan_self_side"
SELF_LEFT_KEY = "link_plan_self_left_key"
SELF_LEFT_PAYLOAD = "link_plan_self_left_payload"
SELF_RIGHT_KEY = "link_plan_self_right_key"
SELF_RIGHT_PAYLOAD = "link_plan_self_right_payload"

STACK_LEFT_INDEX = Index(("link_plan_stack_left_key",))
STACK_RIGHT_INDEX = Index(("link_plan_stack_right_key",))

PAIR_LEFT_INDEX = Index(("link_plan_pair_left_key",))
PAIR_RIGHT_INDEX = Index(("link_plan_pair_right_key",))

STACK_FACTORIES: list[Callable[[JoinSpec, JoinSpec], Link]] = [Link.append, Link.union]
LEFT_FRAMEWORK_INVARIANT = "APPEND/UNION left link framework must match"

STACK_INVERSION_REASON = (
    "the inverted orientation is dropped before the JoinStep is built; a fix has to move the left-framework "
    "invariant in create_joinstep_in_case_of_append_or_union with it, since neither append nor union commutes"
)

MODES = pytest.mark.parametrize("modes", [{ParallelizationMode.SYNC}, {ParallelizationMode.THREADING}])

_PROBE = Path(__file__).with_name("link_planner_probe.py")
# The reduction is deterministic within a process, so a second cold interpreter is the whole cross-process signal.
_PROBE_PROCESSES = 2
_PROBE_EXPECTED = {
    "child": "PyArrowTable",
    "orientation_count": "1",
    "planned_left": "PyArrowTable",
    "planned_right": "PandasDataFrame",
    "trekker_left": "PandasDataFrame",
    "trekker_right": "PyArrowTable",
}


def _column_names(data: Any) -> list[str]:
    if isinstance(data, pa.Table):
        return list(data.column_names)
    return list(data.columns)


def _column_values(data: Any, column: str) -> list[Any]:
    if isinstance(data, pa.Table):
        return list(data.column(column).to_pylist())
    return list(data[column])


def _with_column(data: Any, column: str, values: list[Any]) -> Any:
    if isinstance(data, pa.Table):
        return data.append_column(column, pa.array(values))
    data[column] = values
    return data


def _paired_payloads(data: Any) -> list[str]:
    """One string per joined row, so a dropped parent or a swapped merge changes the value."""
    left = _column_values(data, SHARED_LEFT_PAYLOAD)
    right = _column_values(data, SHARED_RIGHT_PAYLOAD)
    return [f"{a}|{b}" for a, b in zip(left, right)]


class LinkPlanSharedLeft(FeatureGroup):
    """Pinned to the framework the link declares as its left side, with descending keys as an order oracle."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={SHARED_LEFT_PAYLOAD})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {SHARED_LEFT_KEY: [4, 3, 2, 1], SHARED_LEFT_PAYLOAD: ["l4", "l3", "l2", "l1"]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class LinkPlanSharedRight(FeatureGroup):
    """Pinned to the right framework, with keys that only partly overlap the left side."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={SHARED_RIGHT_PAYLOAD})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {SHARED_RIGHT_KEY: [3, 4, 5], SHARED_RIGHT_PAYLOAD: ["r3", "r4", "r5"]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


def _shared_parents() -> set[Feature]:
    return {Feature(name=SHARED_LEFT_PAYLOAD), Feature(name=SHARED_RIGHT_PAYLOAD)}


class LinkPlanSharedFlexibleChild(FeatureGroup):
    """Takes either framework, so on its own it would keep the declared orientation."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return _shared_parents()

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return _with_column(data, cls.get_class_name(), _paired_payloads(data))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable, PandasDataFrame}


class LinkPlanSharedPinnedChild(FeatureGroup):
    """Takes the right framework only, so it forces the shared link into its inverted orientation."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return _shared_parents()

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return _with_column(data, cls.get_class_name(), _paired_payloads(data))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class LinkPlanSelfSource(FeatureGroup):
    """Serves both sides of the self join; the requested feature name picks the side."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={SELF_LEFT_PAYLOAD, SELF_RIGHT_PAYLOAD})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if SELF_LEFT_PAYLOAD in {str(feature.name) for feature in features.features}:
            return {SELF_LEFT_KEY: [1, 2, 3, 4], SELF_LEFT_PAYLOAD: ["l1", "l2", "l3", "l4"]}
        return {SELF_RIGHT_KEY: [3, 4, 5], SELF_RIGHT_PAYLOAD: ["r3", "r4", "r5"]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class LinkPlanSelfConsumer(FeatureGroup):
    """Consumes both sides; the options are what the discriminators match on."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {
            Feature(name=SELF_LEFT_PAYLOAD, options={SELF_SIDE: "left"}),
            Feature(name=SELF_RIGHT_PAYLOAD, options={SELF_SIDE: "right"}),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        keys = _column_values(data, SELF_LEFT_KEY)
        payloads = _column_values(data, SELF_RIGHT_PAYLOAD)
        return _with_column(data, cls.get_class_name(), [f"{key}|{payload}" for key, payload in zip(keys, payloads)])

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class LinkPlanStackLeft(FeatureGroup):
    pass


class LinkPlanStackRight(FeatureGroup):
    pass


class LinkPlanStackConsumer(FeatureGroup):
    pass


class LinkPlanPairLeft(FeatureGroup):
    pass


class LinkPlanPairRight(FeatureGroup):
    pass


class LinkPlanPairChild(FeatureGroup):
    pass


class Planned(NamedTuple):
    plan: ExecutionPlan
    link: Link
    link_trekker: LinkTrekker
    graph: Graph
    pre_execution_plan: list[Any]
    left_uuid: UUID
    right_uuid: UUID
    child_uuid: UUID


def _feature(name: str, cfw: type[ComputeFramework], index: Index | None = None, **kwargs: Any) -> Feature:
    feature = Feature(name, index=index, **kwargs)
    feature.compute_frameworks = {cfw}
    return feature


def _step(fg: type[FeatureGroup], feature: Feature, cfw: type[ComputeFramework]) -> FeatureGroupStep:
    feature_set = FeatureSet()
    feature_set.add(feature)
    return FeatureGroupStep(fg, feature_set, set(), cfw)


def _plan(
    link: Link,
    left: Feature,
    right: Feature,
    child: Feature,
    left_fg: type[FeatureGroup],
    right_fg: type[FeatureGroup],
    child_fg: type[FeatureGroup],
) -> Planned:
    """Hand-built two-parent graph, the smallest shape run_link accepts."""
    graph = Graph()
    graph.add_node(left.uuid, NodeProperties(left, left_fg))
    graph.add_node(right.uuid, NodeProperties(right, right_fg))
    graph.add_node(child.uuid, NodeProperties(child, child_fg))
    graph.adjacency_list[left.uuid] = [child.uuid]
    graph.adjacency_list[right.uuid] = [child.uuid]
    graph.adjacency_list[child.uuid] = []
    graph.parent_to_children_mapping[child.uuid] = {left.uuid, right.uuid}

    pre_execution_plan: list[Any] = [
        _step(left_fg, left, left.get_compute_framework()),
        _step(right_fg, right, right.get_compute_framework()),
        _step(child_fg, child, child.get_compute_framework()),
    ]

    plan = ExecutionPlan()
    plan.feature_set_collections = [{left.uuid}, {right.uuid}, {child.uuid}]

    return Planned(plan, link, LinkTrekker(), graph, pre_execution_plan, left.uuid, right.uuid, child.uuid)


def _trek(planned: Planned, left_cfw: type[ComputeFramework], right_cfw: type[ComputeFramework]) -> None:
    """Production shares one set object between data and data_ordered, and invert_link relies on that."""
    trekked = {planned.child_uuid}
    planned.link_trekker.data[(planned.link, left_cfw, right_cfw)] = trekked
    planned.link_trekker.data_ordered[(planned.link, left_cfw, right_cfw)] = trekked


def _run(planned: Planned, left_cfw: type[ComputeFramework], right_cfw: type[ComputeFramework]) -> Optional[JoinStep]:
    link_fw: LinkFrameworkTrekker = (planned.link, left_cfw, right_cfw)
    return planned.plan.run_link(link_fw, planned.link_trekker, planned.graph, planned.pre_execution_plan)


def _pair_scenario(
    *,
    link_factory: Callable[[JoinSpec, JoinSpec], Link] = Link.inner,
    left_cfw: type[ComputeFramework] = PyArrowTable,
    right_cfw: type[ComputeFramework] = PandasDataFrame,
    child_cfw: type[ComputeFramework] = PyArrowTable,
) -> Planned:
    link = link_factory(JoinSpec(LinkPlanPairLeft, PAIR_LEFT_INDEX), JoinSpec(LinkPlanPairRight, PAIR_RIGHT_INDEX))
    return _plan(
        link,
        _feature("link_plan_pair_left_payload", left_cfw, PAIR_LEFT_INDEX),
        _feature("link_plan_pair_right_payload", right_cfw, PAIR_RIGHT_INDEX),
        _feature("link_plan_pair_child", child_cfw),
        LinkPlanPairLeft,
        LinkPlanPairRight,
        LinkPlanPairChild,
    )


def _self_join_scenario(left_side: str, right_side: str) -> Planned:
    link = Link.left(
        JoinSpec(LinkPlanSelfSource, Index((SELF_LEFT_KEY,))),
        JoinSpec(LinkPlanSelfSource, Index((SELF_RIGHT_KEY,))),
        left_discriminator={SELF_SIDE: left_side},
        right_discriminator={SELF_SIDE: right_side},
    )
    return _plan(
        link,
        _feature(SELF_LEFT_PAYLOAD, PyArrowTable, options={SELF_SIDE: "left"}),
        _feature(SELF_RIGHT_PAYLOAD, PyArrowTable, options={SELF_SIDE: "right"}),
        _feature("link_plan_self_child", PyArrowTable),
        LinkPlanSelfSource,
        LinkPlanSelfSource,
        LinkPlanSelfConsumer,
    )


def _stack_scenario(link_factory: Callable[[JoinSpec, JoinSpec], Link], *, same_feature_group: bool) -> Planned:
    """Declared orientation PyArrow to Pandas, consumer on Pandas only, so the trekker inverts."""
    right_fg: type[FeatureGroup] = LinkPlanStackLeft if same_feature_group else LinkPlanStackRight
    link = link_factory(JoinSpec(LinkPlanStackLeft, STACK_LEFT_INDEX), JoinSpec(right_fg, STACK_RIGHT_INDEX))

    left = _feature("link_plan_stack_left_payload", PyArrowTable, STACK_LEFT_INDEX)
    right = _feature("link_plan_stack_right_payload", PandasDataFrame, STACK_RIGHT_INDEX)
    child = _feature("link_plan_stack_consumer", PandasDataFrame)

    planned = _plan(link, left, right, child, LinkPlanStackLeft, right_fg, LinkPlanStackConsumer)
    _trek(planned, PyArrowTable, PandasDataFrame)

    planned_queue: list[Any] = [
        (LinkPlanStackLeft, {left}),
        (right_fg, {right}),
        (link, PyArrowTable, PandasDataFrame),
        (LinkPlanStackConsumer, {child}),
    ]
    ResolveComputeFrameworks(Graph()).links(planned_queue, planned.link_trekker)
    return planned


def _orientations(planned: Planned) -> list[tuple[str, str]]:
    return [
        (key[1].get_class_name(), key[2].get_class_name())
        for key in planned.link_trekker.data
        if key[0] is planned.link
    ]


@MODES
class TestSharedLinkServingTwoChildren:
    """One link between one parent pair, consumed by a flexible child and a pinned child."""

    def _run_both_children(self, modes: set[ParallelizationMode], flight_server: Any) -> dict[str, list[str]]:
        link = Link.inner(
            left=JoinSpec(LinkPlanSharedLeft, Index((SHARED_LEFT_KEY,))),
            right=JoinSpec(LinkPlanSharedRight, Index((SHARED_RIGHT_KEY,))),
        )
        children = (LinkPlanSharedFlexibleChild, LinkPlanSharedPinnedChild)

        results = mloda.run_all(
            [Feature(name=child.get_class_name()) for child in children],
            links={link},
            compute_frameworks=["PyArrowTable", "PandasDataFrame"],
            plugin_collector=PluginCollector.enabled_feature_groups(
                {LinkPlanSharedLeft, LinkPlanSharedRight, *children}
            ),
            flight_server=flight_server,
            parallelization_modes=modes,
        )

        collected: dict[str, list[str]] = {}
        for result in results:
            for child in children:
                name = child.get_class_name()
                if name in _column_names(result):
                    collected[name] = _column_values(result, name)
        return collected

    def test_each_child_pairs_every_shared_key_with_both_parent_payloads(
        self, modes: set[ParallelizationMode], flight_server: Any
    ) -> None:
        collected = self._run_both_children(modes, flight_server)

        assert collected == {
            LinkPlanSharedFlexibleChild.get_class_name(): ["l4|r4", "l3|r3"],
            LinkPlanSharedPinnedChild.get_class_name(): ["l4|r4", "l3|r3"],
        }


def _run_self_join(left_side: str, right_side: str, modes: set[ParallelizationMode], flight_server: Any) -> list[Any]:
    link = Link.left(
        JoinSpec(LinkPlanSelfSource, Index((SELF_LEFT_KEY,))),
        JoinSpec(LinkPlanSelfSource, Index((SELF_RIGHT_KEY,))),
        left_discriminator={SELF_SIDE: left_side},
        right_discriminator={SELF_SIDE: right_side},
    )
    return list(
        mloda.run_all(
            [Feature(name=LinkPlanSelfConsumer.get_class_name())],
            links={link},
            compute_frameworks=["PyArrowTable"],
            plugin_collector=PluginCollector.enabled_feature_groups({LinkPlanSelfSource, LinkPlanSelfConsumer}),
            flight_server=flight_server,
            parallelization_modes=modes,
        )
    )


@MODES
def test_self_join_keeps_every_row_of_the_node_the_left_discriminator_names(
    modes: set[ParallelizationMode], flight_server: Any
) -> None:
    results = _run_self_join("left", "right", modes, flight_server)

    joined = sorted(_column_values(results[0], LinkPlanSelfConsumer.get_class_name()))
    assert joined == ["1|None", "2|None", "3|r3", "4|r4"]


@MODES
def test_swapped_discriminators_bind_the_other_node_as_the_left_side(
    modes: set[ParallelizationMode], flight_server: Any
) -> None:
    """The bound left node then lacks the left index column, and the run says so."""
    # The exception type is incidental: the column-semantics guard reaches the key column before the merge does.
    with pytest.raises((KeyError, ValueError), match=SELF_LEFT_KEY):
        _run_self_join("right", "left", modes, flight_server)


def test_planner_binds_the_left_discriminator_node_as_the_join_destination() -> None:
    planned = _self_join_scenario("left", "right")
    _trek(planned, PyArrowTable, PyArrowTable)

    join_step = _run(planned, PyArrowTable, PyArrowTable)

    assert isinstance(join_step, JoinStep)
    assert join_step.destination_framework_uuids == {planned.left_uuid}
    assert join_step.source_framework_uuids == {planned.right_uuid}


def test_planner_follows_swapped_discriminators_without_complaint() -> None:
    planned = _self_join_scenario("right", "left")
    _trek(planned, PyArrowTable, PyArrowTable)

    join_step = _run(planned, PyArrowTable, PyArrowTable)

    assert isinstance(join_step, JoinStep)
    assert join_step.destination_framework_uuids == {planned.right_uuid}
    assert join_step.source_framework_uuids == {planned.left_uuid}


@pytest.mark.parametrize("link_factory", STACK_FACTORIES)
def test_a_pinned_consumer_inverts_the_stack_link_orientation(
    link_factory: Callable[[JoinSpec, JoinSpec], Link],
) -> None:
    planned = _stack_scenario(link_factory, same_feature_group=False)

    assert _orientations(planned) == [("PandasDataFrame", "PyArrowTable")]


@pytest.mark.parametrize("link_factory", STACK_FACTORIES)
def test_an_inverted_cross_group_stack_link_still_plans_the_declared_orientation(
    link_factory: Callable[[JoinSpec, JoinSpec], Link],
) -> None:
    """The inversion reaches the trekker but not the JoinStep, so the declared sides survive."""
    planned = _stack_scenario(link_factory, same_feature_group=False)

    join_step = _run(planned, PyArrowTable, PandasDataFrame)

    assert isinstance(join_step, JoinStep)
    assert join_step.link.jointype in (JoinType.APPEND, JoinType.UNION)
    assert join_step.destination_framework is PyArrowTable
    assert join_step.source_framework is PandasDataFrame
    assert join_step.destination_framework_uuids == {planned.left_uuid}
    assert join_step.source_framework_uuids == {planned.right_uuid}
    assert join_step.swap_merge_sides is False


@pytest.mark.xfail(strict=True, reason=STACK_INVERSION_REASON)
@pytest.mark.parametrize("link_factory", STACK_FACTORIES)
def test_an_inverted_cross_group_stack_link_joins_where_its_consumer_runs(
    link_factory: Callable[[JoinSpec, JoinSpec], Link],
) -> None:
    """The consumer resolved to the right framework, so the join it feeds should run there too."""
    planned = _stack_scenario(link_factory, same_feature_group=False)

    join_step = _run(planned, PyArrowTable, PandasDataFrame)

    assert isinstance(join_step, JoinStep)
    assert join_step.destination_framework is PandasDataFrame
    assert join_step.source_framework is PyArrowTable
    assert join_step.destination_framework_uuids == {planned.right_uuid}
    assert join_step.source_framework_uuids == {planned.left_uuid}
    assert join_step.swap_merge_sides is True


@pytest.mark.parametrize("link_factory", STACK_FACTORIES)
def test_an_inverted_self_group_stack_link_plans_nothing_for_the_declared_orientation(
    link_factory: Callable[[JoinSpec, JoinSpec], Link],
) -> None:
    planned = _stack_scenario(link_factory, same_feature_group=True)

    assert _run(planned, PyArrowTable, PandasDataFrame) is None


@pytest.mark.parametrize("link_factory", STACK_FACTORIES)
def test_an_inverted_self_group_stack_link_is_rejected_naming_link_and_frameworks(
    link_factory: Callable[[JoinSpec, JoinSpec], Link],
) -> None:
    planned = _stack_scenario(link_factory, same_feature_group=True)

    with pytest.raises(ValueError) as excinfo:
        _run(planned, PandasDataFrame, PyArrowTable)

    message = str(excinfo.value)
    assert LEFT_FRAMEWORK_INVARIANT in message
    assert str(planned.link) in message
    assert PyArrowTable.get_class_name() in message
    assert PandasDataFrame.get_class_name() in message


def test_the_declared_orientation_plans_the_declared_joinstep() -> None:
    planned = _pair_scenario()
    _trek(planned, PyArrowTable, PandasDataFrame)

    join_step = _run(planned, PyArrowTable, PandasDataFrame)

    assert isinstance(join_step, JoinStep)
    assert join_step.destination_framework is PyArrowTable
    assert join_step.source_framework is PandasDataFrame
    assert join_step.destination_framework_uuids == {planned.left_uuid}
    assert join_step.source_framework_uuids == {planned.right_uuid}
    assert join_step.swap_merge_sides is False


def test_an_orientation_without_a_valid_pairing_plans_no_joinstep() -> None:
    """No parent is both a left-side feature group and on the left framework, so nothing pairs up."""
    planned = _pair_scenario(child_cfw=PandasDataFrame)
    _trek(planned, PandasDataFrame, PyArrowTable)

    assert _run(planned, PandasDataFrame, PyArrowTable) is None


def test_a_trekker_inverted_after_queueing_swaps_the_merge_sides() -> None:
    """The queue keeps the declared orientation, so run_link rediscovers the inverted one."""
    planned = _pair_scenario(child_cfw=PandasDataFrame)
    _trek(planned, PandasDataFrame, PyArrowTable)

    join_step = _run(planned, PyArrowTable, PandasDataFrame)

    assert isinstance(join_step, JoinStep)
    assert join_step.destination_framework is PandasDataFrame
    assert join_step.source_framework is PyArrowTable
    assert join_step.destination_framework_uuids == {planned.right_uuid}
    assert join_step.source_framework_uuids == {planned.left_uuid}
    assert join_step.swap_merge_sides is True


def test_a_hand_built_inconsistent_trekker_key_yields_an_inconsistent_joinstep() -> None:
    """This trekker key contradicts the graph, so create_link_trekker_key never reaches this state."""
    planned = _pair_scenario()
    _trek(planned, PandasDataFrame, PyArrowTable)

    join_step = _run(planned, PyArrowTable, PandasDataFrame)

    # The frameworks and the uuid sets disagree: the destination is Pandas while its uuid runs in PyArrow.
    # A rewrite that carries one record per join decision is free to answer differently here.
    assert isinstance(join_step, JoinStep)
    assert join_step.destination_framework is PandasDataFrame
    assert join_step.destination_framework_uuids == {planned.left_uuid}
    assert join_step.source_framework_uuids == {planned.right_uuid}
    assert join_step.swap_merge_sides is True


def test_a_right_join_plans_the_joinstep_where_the_declared_right_side_runs() -> None:
    """RIGHT swaps the frameworks, so the destination holds the declared right side and the merge sides swap back."""
    declared = _pair_scenario(link_factory=Link.right, child_cfw=PandasDataFrame)
    _trek(declared, PyArrowTable, PandasDataFrame)
    inverted = _pair_scenario(link_factory=Link.right, child_cfw=PandasDataFrame)
    _trek(inverted, PandasDataFrame, PyArrowTable)

    declared_step = _run(declared, PyArrowTable, PandasDataFrame)
    inverted_step = _run(inverted, PyArrowTable, PandasDataFrame)

    assert isinstance(declared_step, JoinStep)
    assert declared_step.destination_framework is PandasDataFrame
    assert declared_step.source_framework is PyArrowTable
    assert declared_step.destination_framework_uuids == {declared.right_uuid}
    assert declared_step.source_framework_uuids == {declared.left_uuid}
    assert declared_step.swap_merge_sides is True

    assert isinstance(inverted_step, JoinStep)
    assert inverted_step.destination_framework is PandasDataFrame
    assert inverted_step.source_framework is PyArrowTable
    assert inverted_step.destination_framework_uuids == {inverted.right_uuid}
    assert inverted_step.source_framework_uuids == {inverted.left_uuid}
    assert inverted_step.swap_merge_sides is True


def test_a_right_join_reached_through_a_reversed_key_keeps_the_declared_merge_sides() -> None:
    """The reversed key already puts the declared left side in the destination, so the merge sides must not swap."""
    planned = _pair_scenario(link_factory=Link.right)
    _trek(planned, PandasDataFrame, PyArrowTable)

    join_step = _run(planned, PandasDataFrame, PyArrowTable)

    assert isinstance(join_step, JoinStep)
    assert join_step.destination_framework is PyArrowTable
    assert join_step.source_framework is PandasDataFrame
    assert join_step.destination_framework_uuids == {planned.left_uuid}
    assert join_step.source_framework_uuids == {planned.right_uuid}
    assert join_step.swap_merge_sides is False


def test_a_left_join_inverted_after_queueing_swaps_the_merge_sides() -> None:
    planned = _pair_scenario(link_factory=Link.left, child_cfw=PandasDataFrame)
    _trek(planned, PandasDataFrame, PyArrowTable)

    join_step = _run(planned, PyArrowTable, PandasDataFrame)

    assert isinstance(join_step, JoinStep)
    assert join_step.destination_framework is PandasDataFrame
    assert join_step.source_framework is PyArrowTable
    assert join_step.destination_framework_uuids == {planned.right_uuid}
    assert join_step.source_framework_uuids == {planned.left_uuid}
    assert join_step.swap_merge_sides is True


# Fresh interpreters are slow to start, so this one needs more than the suite-wide per-test budget.
@pytest.mark.timeout(60)
def test_fresh_interpreters_plan_the_same_link_orientation() -> None:
    outputs = run_probes(_PROBE, _PROBE_PROCESSES)

    assert len(outputs) == _PROBE_PROCESSES, f"expected {_PROBE_PROCESSES} probe results, got {len(outputs)}"
    for position, output in enumerate(outputs):
        assert output == _PROBE_EXPECTED, f"probe {position} planned {output}, expected {_PROBE_EXPECTED}"
