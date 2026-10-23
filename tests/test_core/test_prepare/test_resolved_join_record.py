"""One record per join decision, built next to the join steps and signing the same joins they do."""

import cProfile
import pickle  # nosec B403
import pstats
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any, Callable, Iterable, NamedTuple
from uuid import UUID

import pytest

from mloda.core.core.engine import Engine
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.core.step.join_step import JoinStep
from mloda.core.core.step.transform_frame_work_step import TransformFrameworkStep
from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.core.prepare.graph.graph import Graph
from mloda.core.prepare.graph.properties import NodeProperties
from mloda.core.prepare.resolve_compute_frameworks import ResolveComputeFrameworks
from mloda.core.prepare.resolve_links import LinkTrekker
from mloda.core.prepare.resolved_join import (
    DeclinedOrientation,
    JoinSide,
    JoinSignature,
    ResolvedJoin,
    ResolvedJoinPlan,
)
from mloda.core.prepare.join_decision import DeclinedJoin, JoinDecision
from mloda.core.prepare.resolved_join_builder import joinstep_signatures
from mloda.provider import BaseInputData
from mloda.provider import ComputeFramework
from mloda.provider import DataCreator
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Features
from mloda.user import Index
from mloda.user import JoinSpec, JoinType, Link
from mloda.user import Options
from mloda.user import PluginCollector
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from tests.helpers.probe_runner import run_probes
from tests.test_core.test_prepare.join_plan_helpers import feature, join_tokens, trek


PAIR_LEFT_INDEX = Index(("resolved_join_pair_left_key",))
PAIR_RIGHT_INDEX = Index(("resolved_join_pair_right_key",))
OTHER_LEFT_INDEX = Index(("resolved_join_other_left_key",))
OTHER_RIGHT_INDEX = Index(("resolved_join_other_right_key",))
STACK_LEFT_INDEX = Index(("resolved_join_stack_left_key",))
STACK_RIGHT_INDEX = Index(("resolved_join_stack_right_key",))

SELF_SIDE = "resolved_join_self_side"
SELF_LEFT_KEY = "resolved_join_self_left_key"
SELF_RIGHT_KEY = "resolved_join_self_right_key"
SELF_LEFT_CANDIDATES = frozenset({PyArrowTable, PandasDataFrame})
SELF_RIGHT_CANDIDATES = frozenset({PyArrowTable})

END_LEFT_KEY = "resolved_join_end_left_key"
END_LEFT_PAYLOAD = "resolved_join_end_left_payload"
END_RIGHT_KEY = "resolved_join_end_right_key"
END_RIGHT_PAYLOAD = "resolved_join_end_right_payload"

_PROBE = Path(__file__).with_name("resolved_join_probe.py")
# Each side reduces from a framework set, so a second cold interpreter is the cross-process signal.
_PROBE_PROCESSES = 2
_PROBE_EXPECTED = {
    "declined_count": "0",
    "depends_on_count": "0",
    "destination_framework": "PandasDataFrame",
    "destination_is_declared_left": "True",
    "destination_side": "left",
    "jointype": "inner",
    "record_count": "1",
    "source_framework": "PyArrowTable",
    "source_is_declared_right": "True",
    "trekker_left": "PandasDataFrame",
    "trekker_right": "PyArrowTable",
}


class ResolvedJoinPairLeft(FeatureGroup):
    pass


class ResolvedJoinPairRight(FeatureGroup):
    pass


class ResolvedJoinPairLeftDescendant(ResolvedJoinPairLeft):
    """Matches the declared left side polymorphically, at inheritance distance one."""


class ResolvedJoinPairRightDescendant(ResolvedJoinPairRight):
    """Matches the declared right side polymorphically, at inheritance distance one."""


class ResolvedJoinOtherLeft(FeatureGroup):
    pass


class ResolvedJoinOtherRight(FeatureGroup):
    pass


class ResolvedJoinStackLeft(FeatureGroup):
    pass


class ResolvedJoinStackRight(FeatureGroup):
    pass


class ResolvedJoinSelfSource(FeatureGroup):
    pass


class ResolvedJoinUnlinked(FeatureGroup):
    """Feeds a child of a link without being named by it."""


class ResolvedJoinChild(FeatureGroup):
    pass


class ResolvedJoinEndLeft(FeatureGroup):
    """Pinned to the framework the end to end link declares as its left side."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={END_LEFT_PAYLOAD})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {END_LEFT_KEY: [1, 2], END_LEFT_PAYLOAD: ["l1", "l2"]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class ResolvedJoinEndRight(FeatureGroup):
    """Pinned to the other framework, so the join needs a transform hop."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={END_RIGHT_PAYLOAD})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {END_RIGHT_KEY: [1, 2], END_RIGHT_PAYLOAD: ["r1", "r2"]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class ResolvedJoinEndChild(FeatureGroup):
    """Takes either framework, so it keeps the declared orientation."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature(name=END_LEFT_PAYLOAD), Feature(name=END_RIGHT_PAYLOAD)}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable, PandasDataFrame}


Orientation = tuple[type[ComputeFramework], type[ComputeFramework]]
DeclaredFrameworks = dict[UUID, frozenset[type[ComputeFramework]]]


class Planned(NamedTuple):
    plan: ExecutionPlan
    graph: Graph
    link_trekker: LinkTrekker
    queue: list[Any]


class Sides(NamedTuple):
    left_uuid: UUID
    right_uuid: UUID
    child_uuid: UUID
    extra_right_uuid: UUID | None = None


class Built(NamedTuple):
    plan: ExecutionPlan
    link: Link
    sides: Sides
    graph: Graph
    declared_frameworks: DeclaredFrameworks


class ThirdParent(NamedTuple):
    plan: ExecutionPlan
    link: Link
    left_uuid: UUID
    right_uuid: UUID
    unlinked_uuid: UUID


class FrameworkCollision(NamedTuple):
    plan: ExecutionPlan
    link: Link
    left_pandas_uuid: UUID
    left_pyarrow_uuid: UUID
    right_uuid: UUID
    unrelated_uuid: UUID


class FallbackBranchCollision(NamedTuple):
    plan: ExecutionPlan
    link: Link
    left_uuid: UUID
    far_right_uuid: UUID
    unrelated_uuid: UUID


class Chain(NamedTuple):
    plan: ExecutionPlan
    producer: Link
    consumer: Link


def _planned() -> Planned:
    return Planned(ExecutionPlan(), Graph(), LinkTrekker(), [])


def _add_parents(planned: Planned, link: Link, left: Feature, right: Feature) -> None:
    planned.graph.add_node(left.uuid, NodeProperties(left, link.left_feature_group))
    planned.graph.add_node(right.uuid, NodeProperties(right, link.right_feature_group))
    planned.queue.append((link.left_feature_group, {left}))
    planned.queue.append((link.right_feature_group, {right}))


def _add_child(planned: Planned, child: Feature, *parents: Feature) -> None:
    planned.graph.add_node(child.uuid, NodeProperties(child, ResolvedJoinChild))
    for parent in parents:
        planned.graph.adjacency_list[parent.uuid].append(child.uuid)
    planned.graph.adjacency_list[child.uuid] = []
    planned.graph.parent_to_children_mapping[child.uuid] = {parent.uuid for parent in parents}
    planned.queue.append((ResolvedJoinChild, {child}))


def _branch(
    planned: Planned,
    link: Link,
    name: str,
    *,
    left_cfw: type[ComputeFramework] = PyArrowTable,
    right_cfw: type[ComputeFramework] = PandasDataFrame,
    child_cfw: type[ComputeFramework] = PyArrowTable,
    trekked: Orientation | None = None,
    left_options: dict[str, Any] | None = None,
    right_options: dict[str, Any] | None = None,
) -> Sides:
    """Two parents joined by ``link`` plus the consumer, the smallest shape run_link accepts."""
    left = feature(f"{name}_left", left_cfw, link.left_index, left_options)
    right = feature(f"{name}_right", right_cfw, link.right_index, right_options)
    child = feature(f"{name}_child", child_cfw)

    _add_parents(planned, link, left, right)
    planned.queue.append((link, left_cfw, right_cfw))
    _add_child(planned, child, left, right)

    trek(planned.link_trekker, link, trekked or (left_cfw, right_cfw), child.uuid)
    return Sides(left.uuid, right.uuid, child.uuid)


def _finish(
    planned: Planned,
    link: Link,
    sides: Sides,
    declared_frameworks: DeclaredFrameworks | None = None,
) -> Built:
    declared = declared_frameworks or {}
    planned.plan.create_execution_plan(planned.queue, planned.graph, planned.link_trekker, declared)
    return Built(planned.plan, link, sides, planned.graph, declared)


def _pair_link(link_factory: Callable[[JoinSpec, JoinSpec], Link] = Link.inner) -> Link:
    return link_factory(
        JoinSpec(ResolvedJoinPairLeft, PAIR_LEFT_INDEX), JoinSpec(ResolvedJoinPairRight, PAIR_RIGHT_INDEX)
    )


def _other_link() -> Link:
    return Link.inner(
        JoinSpec(ResolvedJoinOtherLeft, OTHER_LEFT_INDEX), JoinSpec(ResolvedJoinOtherRight, OTHER_RIGHT_INDEX)
    )


def _declared_pair() -> Built:
    planned = _planned()
    link = _pair_link()
    return _finish(planned, link, _branch(planned, link, "resolved_join_declared"))


def _inverted_pair() -> Built:
    """The queue keeps the declared orientation, so run_link rediscovers the inverted one."""
    planned = _planned()
    link = _pair_link()
    sides = _branch(
        planned, link, "resolved_join_inverted", child_cfw=PandasDataFrame, trekked=(PandasDataFrame, PyArrowTable)
    )
    return _finish(planned, link, sides)


def _flipped_decline_parts() -> tuple[Planned, Link]:
    """The queue key finds no children and run_link flips it to the key that declines."""
    planned = _planned()
    link = Link.left(
        JoinSpec(ResolvedJoinSelfSource, Index((SELF_LEFT_KEY,))),
        JoinSpec(ResolvedJoinSelfSource, Index((SELF_RIGHT_KEY,))),
        left_discriminator={SELF_SIDE: "left"},
        right_discriminator={SELF_SIDE: "right"},
    )

    left = feature("resolved_join_flip_decline_left", PyArrowTable, link.left_index, {SELF_SIDE: "left"})
    right = feature("resolved_join_flip_decline_right", PandasDataFrame, link.right_index, {SELF_SIDE: "right"})
    child = feature("resolved_join_flip_decline_child", PyArrowTable)

    _add_parents(planned, link, left, right)
    planned.queue.append((link, PyArrowTable, PandasDataFrame))
    planned.queue.append((link, PandasDataFrame, PyArrowTable))
    _add_child(planned, child, left, right)
    trek(planned.link_trekker, link, (PyArrowTable, PandasDataFrame), child.uuid)
    return planned, link


def _right_join() -> Built:
    planned = _planned()
    link = _pair_link(Link.right)
    return _finish(planned, link, _branch(planned, link, "resolved_join_right", child_cfw=PandasDataFrame))


def _inverted_left_join() -> Built:
    planned = _planned()
    link = _pair_link(Link.left)
    sides = _branch(
        planned, link, "resolved_join_left", child_cfw=PandasDataFrame, trekked=(PandasDataFrame, PyArrowTable)
    )
    return _finish(planned, link, sides)


def _self_join_parts() -> tuple[Planned, Link, Sides]:
    """One feature group on both sides, so only the discriminators tell the two parents apart."""
    planned = _planned()
    link = Link.left(
        JoinSpec(ResolvedJoinSelfSource, Index((SELF_LEFT_KEY,))),
        JoinSpec(ResolvedJoinSelfSource, Index((SELF_RIGHT_KEY,))),
        left_discriminator={SELF_SIDE: "left"},
        right_discriminator={SELF_SIDE: "right"},
    )
    sides = _branch(
        planned,
        link,
        "resolved_join_self",
        right_cfw=PyArrowTable,
        left_options={SELF_SIDE: "left"},
        right_options={SELF_SIDE: "right"},
    )
    return planned, link, sides


def _self_join() -> Built:
    planned, link, sides = _self_join_parts()
    return _finish(planned, link, sides)


def _self_join_with_split_declarations() -> Built:
    """The nearest subclass rule cannot split one feature group over two sides; the resolved sets separate them."""
    planned, link, sides = _self_join_parts()
    return _finish(
        planned,
        link,
        sides,
        {sides.left_uuid: SELF_LEFT_CANDIDATES, sides.right_uuid: SELF_RIGHT_CANDIDATES},
    )


def _append_pair() -> Built:
    planned = _planned()
    link = Link.append(
        JoinSpec(ResolvedJoinStackLeft, STACK_LEFT_INDEX), JoinSpec(ResolvedJoinStackRight, STACK_RIGHT_INDEX)
    )
    return _finish(planned, link, _branch(planned, link, "resolved_join_append"))


def _two_links() -> Built:
    """Two links that share nothing but the plan they are planned into."""
    planned = _planned()
    first = _pair_link()
    second = _other_link()
    sides = _branch(planned, first, "resolved_join_two_first")
    _branch(planned, second, "resolved_join_two_second")
    return _finish(planned, first, sides)


def _declined_orientation_parts() -> tuple[Planned, Link, Sides]:
    """Two children of one link, and only the PyArrow one pairs a left side with a right side."""
    planned = _planned()
    link = _pair_link()

    left = feature("resolved_join_declined_left", PyArrowTable, link.left_index)
    right = feature("resolved_join_declined_right", PandasDataFrame, link.right_index)
    kept = feature("resolved_join_declined_kept_child", PyArrowTable)
    dropped = feature("resolved_join_declined_dropped_child", PandasDataFrame)

    _add_parents(planned, link, left, right)
    planned.queue.append((link, PyArrowTable, PandasDataFrame))
    planned.queue.append((link, PandasDataFrame, PyArrowTable))
    _add_child(planned, kept, left, right)
    _add_child(planned, dropped, left, right)

    trek(planned.link_trekker, link, (PyArrowTable, PandasDataFrame), kept.uuid)
    trek(planned.link_trekker, link, (PandasDataFrame, PyArrowTable), dropped.uuid)

    return planned, link, Sides(left.uuid, right.uuid, kept.uuid)


def _pair_with_declined_orientation() -> Built:
    planned, link, sides = _declined_orientation_parts()
    return _finish(planned, link, sides)


def _third_parent_the_link_never_mentions(third_cfw: type[ComputeFramework], descends_from_right: bool) -> ThirdParent:
    """A right join whose child also has a same-framework third parent, descending from one side, the link never mentions."""
    planned = _planned()
    link = _pair_link(Link.right)

    left = feature("resolved_join_unlinked_left", PyArrowTable, link.left_index)
    right = feature("resolved_join_unlinked_right", PandasDataFrame, link.right_index)
    unlinked = feature("resolved_join_unlinked_third", third_cfw)
    child = feature("resolved_join_unlinked_child", PandasDataFrame)
    ancestor = right if descends_from_right else left

    _add_parents(planned, link, left, right)
    planned.graph.add_node(unlinked.uuid, NodeProperties(unlinked, ResolvedJoinUnlinked))
    planned.queue.append((ResolvedJoinUnlinked, {unlinked}))
    planned.queue.append((link, PyArrowTable, PandasDataFrame))
    # The third parent descends from a side on the same framework, so lineage counts through it.
    planned.graph.adjacency_list[ancestor.uuid].append(unlinked.uuid)
    planned.graph.adjacency_list[unlinked.uuid] = []
    planned.graph.parent_to_children_mapping[unlinked.uuid] = {ancestor.uuid}
    _add_child(planned, child, left, right, unlinked)
    trek(planned.link_trekker, link, (PyArrowTable, PandasDataFrame), child.uuid)

    declared: DeclaredFrameworks = {
        left.uuid: frozenset({PyArrowTable}),
        right.uuid: frozenset({PandasDataFrame}),
        unlinked.uuid: frozenset({third_cfw, PythonDictFramework}),
    }
    planned.plan.create_execution_plan(planned.queue, planned.graph, planned.link_trekker, declared)
    return ThirdParent(planned.plan, link, left.uuid, right.uuid, unlinked.uuid)


def _link_with_a_third_parent_it_never_mentions() -> ThirdParent:
    return _third_parent_the_link_never_mentions(PyArrowTable, descends_from_right=False)


def _link_with_a_pandas_third_parent_descending_from_the_right_side() -> ThirdParent:
    return _third_parent_the_link_never_mentions(PandasDataFrame, descends_from_right=True)


def _link_with_a_declared_left_split_across_frameworks_and_a_colliding_third_parent() -> FrameworkCollision:
    """The declared left side has two nearest parents on different frameworks, so the full containment
    check (both sides at once) fails; an unrelated third parent shares the destination framework with one
    of them, so the unfiltered per-framework fallback hands it to record.left along with that parent."""
    planned = _planned()
    link = _pair_link()

    left_pandas = feature("resolved_join_fw_collision_left_pandas", PandasDataFrame, link.left_index)
    left_pyarrow = feature("resolved_join_fw_collision_left_pyarrow", PyArrowTable, link.left_index)
    right = feature("resolved_join_fw_collision_right", PyArrowTable, link.right_index)
    unrelated = feature("resolved_join_fw_collision_unrelated", PandasDataFrame)
    child = feature("resolved_join_fw_collision_child", PyArrowTable)

    planned.graph.add_node(left_pandas.uuid, NodeProperties(left_pandas, link.left_feature_group))
    planned.graph.add_node(left_pyarrow.uuid, NodeProperties(left_pyarrow, link.left_feature_group))
    planned.graph.add_node(right.uuid, NodeProperties(right, link.right_feature_group))
    planned.graph.add_node(unrelated.uuid, NodeProperties(unrelated, ResolvedJoinUnlinked))
    planned.queue.append((link.left_feature_group, {left_pandas, left_pyarrow}))
    planned.queue.append((link.right_feature_group, {right}))
    planned.queue.append((ResolvedJoinUnlinked, {unrelated}))
    planned.queue.append((link, PandasDataFrame, PyArrowTable))
    _add_child(planned, child, left_pandas, left_pyarrow, right, unrelated)
    trek(planned.link_trekker, link, (PandasDataFrame, PyArrowTable), child.uuid)

    declared_frameworks: DeclaredFrameworks = {
        left_pandas.uuid: frozenset({PandasDataFrame}),
        left_pyarrow.uuid: frozenset({PyArrowTable}),
        right.uuid: frozenset({PyArrowTable}),
        unrelated.uuid: frozenset({PandasDataFrame, PythonDictFramework}),
    }
    planned.plan.create_execution_plan(planned.queue, planned.graph, planned.link_trekker, declared_frameworks)
    return FrameworkCollision(planned.plan, link, left_pandas.uuid, left_pyarrow.uuid, right.uuid, unrelated.uuid)


def _right_join_farther_right_parent_and_a_colliding_third_parent() -> FallbackBranchCollision:
    """Lands run_link's final fallback branch; an unrelated third parent shares the source framework."""
    planned = _planned()
    link = _pair_link(Link.right)

    left = feature("resolved_join_fallback_collision_left", PyArrowTable, link.left_index)
    nearest_right = feature("resolved_join_fallback_collision_nearest_right", PythonDictFramework, link.right_index)
    far_right = feature("resolved_join_fallback_collision_far_right", PandasDataFrame, link.right_index)
    unrelated = feature("resolved_join_fallback_collision_unrelated", PyArrowTable)
    child = feature("resolved_join_fallback_collision_child", PandasDataFrame)

    planned.graph.add_node(left.uuid, NodeProperties(left, link.left_feature_group))
    planned.graph.add_node(nearest_right.uuid, NodeProperties(nearest_right, ResolvedJoinPairRight))
    planned.graph.add_node(far_right.uuid, NodeProperties(far_right, ResolvedJoinPairRightDescendant))
    planned.graph.add_node(unrelated.uuid, NodeProperties(unrelated, ResolvedJoinUnlinked))
    planned.queue.append((link.left_feature_group, {left}))
    planned.queue.append((ResolvedJoinPairRight, {nearest_right}))
    planned.queue.append((ResolvedJoinPairRightDescendant, {far_right}))
    planned.queue.append((ResolvedJoinUnlinked, {unrelated}))
    planned.queue.append((link, PyArrowTable, PandasDataFrame))
    _add_child(planned, child, left, nearest_right, far_right, unrelated)
    # The trekker key matches the queued key directly, so run_link never flips here.
    trek(planned.link_trekker, link, (PyArrowTable, PandasDataFrame), child.uuid)

    planned.plan.create_execution_plan(planned.queue, planned.graph, planned.link_trekker)
    return FallbackBranchCollision(planned.plan, link, left.uuid, far_right.uuid, unrelated.uuid)


def _ordered_chain() -> Chain:
    planned = _planned()
    producer = _pair_link()
    consumer = _other_link()
    _branch(planned, producer, "resolved_join_chain_producer")
    _branch(planned, consumer, "resolved_join_chain_consumer")
    # The value side of an order entry lists the links that have to wait for the key.
    planned.link_trekker.order[producer.uuid] = {consumer.uuid}

    planned.plan.create_execution_plan(planned.queue, planned.graph, planned.link_trekker)
    return Chain(planned.plan, producer, consumer)


def _case_override_inverted() -> Built:
    """The queue keeps the declared orientation, so run_link rediscovers the inverted one; the child sits on the
    declared left framework too, so is_valid_join_step's case helper resolves left/right before the remap runs."""
    planned = _planned()
    link = _pair_link()
    sides = _branch(planned, link, "resolved_join_case_override", trekked=(PandasDataFrame, PyArrowTable))
    return _finish(planned, link, sides)


def _case_override_beats_nearer_wrong_framework_left() -> Built:
    """far_left is farther than nearest_left but is the only one on the framework the join needs; the case helper
    filters candidates by framework, not distance. destination_side turns RIGHT via that check here, not inversion."""
    planned = _planned()
    link = _pair_link()

    nearest_left = feature("resolved_join_nearer_wrong_fw_left", PandasDataFrame, link.left_index)
    far_left = feature("resolved_join_nearer_wrong_fw_far_left", PyArrowTable, link.left_index)
    right = feature("resolved_join_nearer_wrong_fw_right", PyArrowTable, link.right_index)
    child = feature("resolved_join_nearer_wrong_fw_child", PyArrowTable)

    planned.graph.add_node(nearest_left.uuid, NodeProperties(nearest_left, ResolvedJoinPairLeft))
    planned.graph.add_node(far_left.uuid, NodeProperties(far_left, ResolvedJoinPairLeftDescendant))
    planned.graph.add_node(right.uuid, NodeProperties(right, ResolvedJoinPairRight))
    planned.queue.append((ResolvedJoinPairLeft, {nearest_left}))
    planned.queue.append((ResolvedJoinPairLeftDescendant, {far_left}))
    planned.queue.append((ResolvedJoinPairRight, {right}))
    planned.queue.append((link, PyArrowTable, PyArrowTable))
    _add_child(planned, child, nearest_left, far_left, right)
    # The trekker key matches the queued key directly, so run_link never inverts here.
    trek(planned.link_trekker, link, (PyArrowTable, PyArrowTable), child.uuid)

    return _finish(planned, link, Sides(far_left.uuid, right.uuid, child.uuid))


def _right_join_consumer_shares_declared_left_framework() -> Built:
    """The consumer sits on the same framework as link_fw[1] (PyArrowTable), which triggers
    case_link_fw_is_equal_to_children_fw purely on framework equality, even though the link's two
    feature groups (ResolvedJoinPairLeft / ResolvedJoinPairRight) are unrelated: not equal, not
    polymorphic. This must still plan successfully."""
    planned = _planned()
    link = _pair_link(Link.right)
    return _finish(planned, link, _branch(planned, link, "resolved_join_right_shares_left_fw", child_cfw=PyArrowTable))


def _right_join_ambiguous_polymorphic_left_candidates() -> Built:
    """Two left-side candidates (the exact declared class and a subclass) both sit on link_fw[1]'s
    framework, so case_link_fw_is_equal_to_children_fw finds two valid pairs sharing the same right
    side; right_index disambiguation cannot break the tie since it only inspects the shared right side.
    This genuine polymorphic ambiguity must still be rejected for a RIGHT join."""
    planned = _planned()
    link = _pair_link(Link.right)

    left_exact = feature("resolved_join_ambiguous_left_exact", PyArrowTable, link.left_index)
    left_sub = feature("resolved_join_ambiguous_left_sub", PyArrowTable, link.left_index)
    right = feature("resolved_join_ambiguous_right", PandasDataFrame, link.right_index)
    child = feature("resolved_join_ambiguous_child", PyArrowTable)

    planned.graph.add_node(left_exact.uuid, NodeProperties(left_exact, ResolvedJoinPairLeft))
    planned.graph.add_node(left_sub.uuid, NodeProperties(left_sub, ResolvedJoinPairLeftDescendant))
    planned.graph.add_node(right.uuid, NodeProperties(right, ResolvedJoinPairRight))
    planned.queue.append((ResolvedJoinPairLeft, {left_exact}))
    planned.queue.append((ResolvedJoinPairLeftDescendant, {left_sub}))
    planned.queue.append((ResolvedJoinPairRight, {right}))
    planned.queue.append((link, PyArrowTable, PandasDataFrame))
    _add_child(planned, child, left_exact, left_sub, right)
    trek(planned.link_trekker, link, (PyArrowTable, PandasDataFrame), child.uuid)

    return _finish(planned, link, Sides(left_exact.uuid, right.uuid, child.uuid))


def _case_override_disagrees_with_the_nearest_split(*, with_declared_frameworks: bool = False) -> Built:
    """Both sides have a nearer, wrong-framework sibling and a farther, correct-framework one; the case helper
    selects the farther pair on both sides. destination_framework and source_framework differ here (unlike
    _case_override_beats_nearer_wrong_framework_left's shared framework), so a swap decision based on the
    nearest split's frameworks (which never sees the case-selected, farther parents) can disagree with the
    framework the case helper actually bound to each side."""
    planned = _planned()
    link = _pair_link()

    nearest_left = feature("resolved_join_split_disagree_nearest_left", PythonDictFramework, link.left_index)
    far_left = feature("resolved_join_split_disagree_far_left", PyArrowTable, link.left_index)
    nearest_right = feature("resolved_join_split_disagree_nearest_right", PyArrowTable, link.right_index)
    far_right = feature("resolved_join_split_disagree_far_right", PandasDataFrame, link.right_index)
    child = feature("resolved_join_split_disagree_child", PyArrowTable)

    planned.graph.add_node(nearest_left.uuid, NodeProperties(nearest_left, ResolvedJoinPairLeft))
    planned.graph.add_node(far_left.uuid, NodeProperties(far_left, ResolvedJoinPairLeftDescendant))
    planned.graph.add_node(nearest_right.uuid, NodeProperties(nearest_right, ResolvedJoinPairRight))
    planned.graph.add_node(far_right.uuid, NodeProperties(far_right, ResolvedJoinPairRightDescendant))
    planned.queue.append((ResolvedJoinPairLeft, {nearest_left}))
    planned.queue.append((ResolvedJoinPairLeftDescendant, {far_left}))
    planned.queue.append((ResolvedJoinPairRight, {nearest_right}))
    planned.queue.append((ResolvedJoinPairRightDescendant, {far_right}))
    planned.queue.append((link, PyArrowTable, PandasDataFrame))
    _add_child(planned, child, nearest_left, far_left, nearest_right, far_right)
    trek(planned.link_trekker, link, (PyArrowTable, PandasDataFrame), child.uuid)

    declared_frameworks: DeclaredFrameworks | None = None
    if with_declared_frameworks:
        declared_frameworks = {
            far_left.uuid: frozenset({PyArrowTable}),
            far_right.uuid: frozenset({PandasDataFrame}),
        }

    return _finish(planned, link, Sides(far_left.uuid, far_right.uuid, child.uuid), declared_frameworks)


def _right_join_farther_right_parent_holds_destination_framework() -> Built:
    """split_by_declared_side keeps only the nearest right parent (PythonDictFramework); the farther,
    subclass right parent that actually sits on the destination framework (PandasDataFrame) is excluded from
    declared_right_frameworks, so declared-side membership alone cannot decide. destination_framework and
    source_framework differ here, so the identity tiebreaker decides directly: destination_framework
    (PandasDataFrame) is not the trekker key's left framework (PyArrowTable), so the destination stays RIGHT."""
    planned = _planned()
    link = _pair_link(Link.right)

    left = feature("resolved_join_declared_split_left", PyArrowTable, link.left_index)
    nearest_right = feature("resolved_join_declared_split_nearest_right", PythonDictFramework, link.right_index)
    far_right = feature("resolved_join_declared_split_far_right", PandasDataFrame, link.right_index)
    child = feature("resolved_join_declared_split_child", PandasDataFrame)

    planned.graph.add_node(left.uuid, NodeProperties(left, link.left_feature_group))
    planned.graph.add_node(nearest_right.uuid, NodeProperties(nearest_right, ResolvedJoinPairRight))
    planned.graph.add_node(far_right.uuid, NodeProperties(far_right, ResolvedJoinPairRightDescendant))
    planned.queue.append((link.left_feature_group, {left}))
    planned.queue.append((ResolvedJoinPairRight, {nearest_right}))
    planned.queue.append((ResolvedJoinPairRightDescendant, {far_right}))
    planned.queue.append((link, PyArrowTable, PandasDataFrame))
    _add_child(planned, child, left, nearest_right, far_right)
    # The trekker key matches the queued key directly, so run_link never flips here.
    trek(planned.link_trekker, link, (PyArrowTable, PandasDataFrame), child.uuid)

    return _finish(planned, link, Sides(left.uuid, far_right.uuid, child.uuid))


def _right_join_both_sides_claim_destination_framework() -> Built:
    """The nearest left parent and the sole right parent both sit on the destination framework, so declared-side
    membership names both sides and cannot decide; the identity tiebreaker must still pick RIGHT."""
    planned = _planned()
    link = _pair_link(Link.right)

    nearest_left = feature("resolved_join_both_claim_nearest_left", PandasDataFrame, link.left_index)
    far_left = feature("resolved_join_both_claim_far_left", PyArrowTable, link.left_index)
    right = feature("resolved_join_both_claim_right", PandasDataFrame, link.right_index)
    child = feature("resolved_join_both_claim_child", PandasDataFrame)

    planned.graph.add_node(nearest_left.uuid, NodeProperties(nearest_left, ResolvedJoinPairLeft))
    planned.graph.add_node(far_left.uuid, NodeProperties(far_left, ResolvedJoinPairLeftDescendant))
    planned.graph.add_node(right.uuid, NodeProperties(right, ResolvedJoinPairRight))
    planned.queue.append((ResolvedJoinPairLeft, {nearest_left}))
    planned.queue.append((ResolvedJoinPairLeftDescendant, {far_left}))
    planned.queue.append((ResolvedJoinPairRight, {right}))
    planned.queue.append((link, PyArrowTable, PandasDataFrame))
    _add_child(planned, child, nearest_left, far_left, right)
    # The trekker key matches the queued key directly, so run_link never flips here.
    trek(planned.link_trekker, link, (PyArrowTable, PandasDataFrame), child.uuid)

    # far_left (PyArrowTable) is the sole source-side uuid; nearest_left shares the destination
    # framework but is declared left, so it belongs to neither join end.
    return _finish(planned, link, Sides(far_left.uuid, right.uuid, child.uuid, nearest_left.uuid))


def _right_join_nearest_left_and_farther_right_share_destination_framework() -> Built:
    """The nearest left parent (distance 0) legitimately sits on the destination framework
    (PandasDataFrame), but the nearest right parent (distance 0) sits on an unrelated third framework
    (PythonDictFramework) while a farther, subclass right parent (distance 1) is the one that actually
    shares the destination framework. Nearest-only declared-side membership sees only the left side on
    the destination framework and confidently (and wrongly) returns LEFT; the farther right parent must
    be enough to make membership ambiguous so the RIGHT-jointype tiebreak decides RIGHT."""
    planned = _planned()
    link = _pair_link(Link.right)

    nearest_left = feature("resolved_join_mixed_split_nearest_left", PandasDataFrame, link.left_index)
    far_left = feature("resolved_join_mixed_split_far_left", PyArrowTable, link.left_index)
    nearest_right = feature("resolved_join_mixed_split_nearest_right", PythonDictFramework, link.right_index)
    far_right = feature("resolved_join_mixed_split_far_right", PandasDataFrame, link.right_index)
    child = feature("resolved_join_mixed_split_child", PandasDataFrame)

    planned.graph.add_node(nearest_left.uuid, NodeProperties(nearest_left, ResolvedJoinPairLeft))
    planned.graph.add_node(far_left.uuid, NodeProperties(far_left, ResolvedJoinPairLeftDescendant))
    planned.graph.add_node(nearest_right.uuid, NodeProperties(nearest_right, ResolvedJoinPairRight))
    planned.graph.add_node(far_right.uuid, NodeProperties(far_right, ResolvedJoinPairRightDescendant))
    planned.queue.append((ResolvedJoinPairLeft, {nearest_left}))
    planned.queue.append((ResolvedJoinPairLeftDescendant, {far_left}))
    planned.queue.append((ResolvedJoinPairRight, {nearest_right}))
    planned.queue.append((ResolvedJoinPairRightDescendant, {far_right}))
    planned.queue.append((link, PyArrowTable, PandasDataFrame))
    _add_child(planned, child, nearest_left, far_left, nearest_right, far_right)
    trek(planned.link_trekker, link, (PyArrowTable, PandasDataFrame), child.uuid)

    return _finish(planned, link, Sides(far_left.uuid, far_right.uuid, child.uuid, nearest_left.uuid))


def _right_join_both_declared_sides_share_one_framework_child_on_a_third() -> Built:
    """Regression shape for issue #1137 finding #2: both declared sides sit on the same framework, and only
    the consumer names a third, unrelated framework, so no case-override branch can intervene."""
    planned = _planned()
    link = _pair_link(Link.right)
    sides = _branch(
        planned,
        link,
        "resolved_join_shared_fw_third_child",
        left_cfw=PandasDataFrame,
        right_cfw=PandasDataFrame,
        child_cfw=PyArrowTable,
    )
    return _finish(planned, link, sides)


def _right_join_declared_left_spans_frameworks_declared_right_is_pyarrow_only() -> Built:
    """Regression shape for issue #1137 finding #2 ('hop names PairRight -> PairLeft'): declared left has
    nearest parents on two frameworks, declared right is PyArrow-only, and the consumer sits on a third
    framework so it does not trip case_link_fw_is_equal_to_children_fw's RIGHT-join guard."""
    planned = _planned()
    link = _pair_link(Link.right)

    # One shared name: the scenario is about join-side resolution, and the child reads that name from either step.
    left_pandas = feature("resolved_join_right_only_pyarrow_left", PandasDataFrame, link.left_index)
    left_pyarrow = feature("resolved_join_right_only_pyarrow_left", PyArrowTable, link.left_index)
    right = feature("resolved_join_right_only_pyarrow_right", PyArrowTable, link.right_index)
    child = feature("resolved_join_right_only_pyarrow_child", PythonDictFramework)

    planned.graph.add_node(left_pandas.uuid, NodeProperties(left_pandas, link.left_feature_group))
    planned.graph.add_node(left_pyarrow.uuid, NodeProperties(left_pyarrow, link.left_feature_group))
    planned.graph.add_node(right.uuid, NodeProperties(right, link.right_feature_group))
    planned.queue.append((link.left_feature_group, {left_pandas, left_pyarrow}))
    planned.queue.append((link.right_feature_group, {right}))
    planned.queue.append((link, PyArrowTable, PyArrowTable))
    _add_child(planned, child, left_pandas, left_pyarrow, right)
    trek(planned.link_trekker, link, (PyArrowTable, PyArrowTable), child.uuid)

    return _finish(planned, link, Sides(left_pyarrow.uuid, right.uuid, child.uuid))


def _inner_join_ambiguous_split_stays_on_the_tiebreak_answer() -> Built:
    """Same shape as _case_override_disagrees_with_the_nearest_split, consumer on a third framework; planning it
    is refused as a dishonest record."""
    planned = _planned()
    link = _pair_link()

    nearest_left = feature("resolved_join_inner_escalation_nearest_left", PythonDictFramework, link.left_index)
    far_left = feature("resolved_join_inner_escalation_far_left", PyArrowTable, link.left_index)
    nearest_right = feature("resolved_join_inner_escalation_nearest_right", PyArrowTable, link.right_index)
    far_right = feature("resolved_join_inner_escalation_far_right", PandasDataFrame, link.right_index)
    child = feature("resolved_join_inner_escalation_child", PythonDictFramework)

    planned.graph.add_node(nearest_left.uuid, NodeProperties(nearest_left, ResolvedJoinPairLeft))
    planned.graph.add_node(far_left.uuid, NodeProperties(far_left, ResolvedJoinPairLeftDescendant))
    planned.graph.add_node(nearest_right.uuid, NodeProperties(nearest_right, ResolvedJoinPairRight))
    planned.graph.add_node(far_right.uuid, NodeProperties(far_right, ResolvedJoinPairRightDescendant))
    planned.queue.append((ResolvedJoinPairLeft, {nearest_left}))
    planned.queue.append((ResolvedJoinPairLeftDescendant, {far_left}))
    planned.queue.append((ResolvedJoinPairRight, {nearest_right}))
    planned.queue.append((ResolvedJoinPairRightDescendant, {far_right}))
    planned.queue.append((link, PyArrowTable, PandasDataFrame))
    # The farther left parent descends from the nearest one, so the two are linked by ancestry.
    planned.graph.adjacency_list[nearest_left.uuid].append(far_left.uuid)
    planned.graph.parent_to_children_mapping[far_left.uuid] = {nearest_left.uuid}
    _add_child(planned, child, nearest_left, far_left, nearest_right, far_right)
    trek(planned.link_trekker, link, (PyArrowTable, PandasDataFrame), child.uuid)

    return _finish(planned, link, Sides(nearest_left.uuid, nearest_right.uuid, child.uuid))


def _right_join_declared_right_subclass_also_matches_declared_left() -> Built:
    """The link's declared left feature group is itself a subclass of its declared right feature group
    (ResolvedJoinPairRightDescendant IS-A ResolvedJoinPairRight), so the single left-side parent also
    matches the right side's any-distance split at its own (farther) distance. The widened right-side
    check added for issue #1173 must not count that same parent as a competing right-side claim; the
    left-declared parent's own destination-framework membership must decide this cleanly."""
    planned = _planned()
    link = Link.right(
        JoinSpec(ResolvedJoinPairRightDescendant, PAIR_LEFT_INDEX), JoinSpec(ResolvedJoinPairRight, PAIR_RIGHT_INDEX)
    )

    left_node = feature("resolved_join_subclass_overlap_left", PandasDataFrame, link.left_index)
    right_node = feature("resolved_join_subclass_overlap_right", PyArrowTable, link.right_index)
    child = feature("resolved_join_subclass_overlap_child", PythonDictFramework)

    planned.graph.add_node(left_node.uuid, NodeProperties(left_node, ResolvedJoinPairRightDescendant))
    planned.graph.add_node(right_node.uuid, NodeProperties(right_node, ResolvedJoinPairRight))
    planned.queue.append((ResolvedJoinPairRightDescendant, {left_node}))
    planned.queue.append((ResolvedJoinPairRight, {right_node}))
    planned.queue.append((link, PyArrowTable, PandasDataFrame))
    _add_child(planned, child, left_node, right_node)
    trek(planned.link_trekker, link, (PyArrowTable, PandasDataFrame), child.uuid)

    return _finish(planned, link, Sides(left_node.uuid, right_node.uuid, child.uuid))


def _join_steps(plan: ExecutionPlan) -> list[JoinStep]:
    return [step for step in plan if isinstance(step, JoinStep)]


def _transform_steps(plan: ExecutionPlan, link: Link) -> list[TransformFrameworkStep]:
    return [
        step for step in plan if isinstance(step, TransformFrameworkStep) and step.link_id in join_tokens(plan, link)
    ]


def _records(plan: ExecutionPlan, link: Link) -> tuple[ResolvedJoin, ...]:
    return plan.resolved_join_plan.records_of_link(link.uuid)


def _one_record(plan: ExecutionPlan, link: Link) -> ResolvedJoin:
    records = _records(plan, link)
    assert len(records) == 1, f"the orientation must build exactly one record; got: {records}"
    return records[0]


def _without_depends(signatures: Iterable[JoinSignature]) -> frozenset[JoinSignature]:
    return frozenset(signature._replace(depends_on_links=()) for signature in signatures)


def test_a_record_is_inverted_exactly_when_its_destination_is_the_right_side() -> None:
    declared = _declared_pair()
    inverted = _inverted_pair()

    declared_record = _one_record(declared.plan, declared.link)
    inverted_record = _one_record(inverted.plan, inverted.link)

    assert declared_record.destination_side is JoinSide.LEFT
    assert declared_record.inverted is False
    assert inverted_record.destination_side is JoinSide.RIGHT
    assert inverted_record.inverted is True


def test_a_record_refuses_assignment_to_its_fields_and_to_inverted() -> None:
    built = _declared_pair()
    record = _one_record(built.plan, built.link)

    with pytest.raises(FrozenInstanceError):
        setattr(record, "destination_side", JoinSide.RIGHT)

    with pytest.raises(AttributeError):
        setattr(record, "inverted", True)


def test_destination_and_source_name_the_sides_the_destination_side_picks() -> None:
    declared = _declared_pair()
    inverted = _inverted_pair()

    declared_record = _one_record(declared.plan, declared.link)
    inverted_record = _one_record(inverted.plan, inverted.link)

    assert declared_record.destination is declared_record.left
    assert declared_record.source is declared_record.right
    assert inverted_record.destination is inverted_record.right
    assert inverted_record.source is inverted_record.left


def test_a_record_survives_the_round_trip_to_a_multiprocessing_worker() -> None:
    built = _declared_pair()
    record = _one_record(built.plan, built.link)

    restored = pickle.loads(pickle.dumps(record))  # nosec B301

    assert restored == record


def test_the_declared_orientation_of_an_inner_pair_builds_one_left_destination_record() -> None:
    built = _declared_pair()

    record = _one_record(built.plan, built.link)

    assert record.link_uuid == built.link.uuid
    assert record.jointype is JoinType.INNER
    assert record.destination_side is JoinSide.LEFT
    assert record.inverted is False
    assert record.left.uuids == {built.sides.left_uuid}
    assert record.right.uuids == {built.sides.right_uuid}
    assert record.destination_uuids == {built.sides.left_uuid}
    assert record.source_uuids == {built.sides.right_uuid}
    assert record.destination_framework is PyArrowTable
    assert record.source_framework is PandasDataFrame
    assert record.left.feature_group is ResolvedJoinPairLeft
    assert record.left.index == built.link.left_index


def test_an_orientation_inverted_after_queueing_still_names_the_declared_left_side() -> None:
    built = _inverted_pair()

    record = _one_record(built.plan, built.link)

    assert record.destination_side is JoinSide.RIGHT
    assert record.inverted is True
    assert record.left.uuids == {built.sides.left_uuid}
    assert record.right.uuids == {built.sides.right_uuid}
    assert record.destination_uuids == {built.sides.right_uuid}
    assert record.source_uuids == {built.sides.left_uuid}


def test_a_right_join_binds_the_destination_to_the_declared_right_side() -> None:
    built = _right_join()

    record = _one_record(built.plan, built.link)

    assert record.jointype is JoinType.RIGHT
    assert record.destination_side is JoinSide.RIGHT
    assert record.right.uuids == {built.sides.right_uuid}
    assert record.destination.uuids == {built.sides.right_uuid}
    assert record.destination_uuids == {built.sides.right_uuid}


def test_a_right_joins_destination_stays_right_when_the_nearest_right_parent_is_on_the_wrong_framework() -> None:
    """The nearest-only split must not blind the declared-side check to a farther, correct-framework
    right parent; the identity tiebreaker must decide this case directly, since declared-side
    membership sees neither side on the destination framework."""
    built = _right_join_farther_right_parent_holds_destination_framework()

    record = _one_record(built.plan, built.link)

    assert record.jointype is JoinType.RIGHT
    assert record.destination_side is JoinSide.RIGHT
    assert record.destination_framework is PandasDataFrame
    # Pre-fix, the trekker-flip fallback also swapped which uuids land on left/right and left
    # swap_merge_sides False, which would merge the arguments in the wrong order at execution time.
    assert record.left.uuids == {built.sides.left_uuid}
    assert record.right.uuids == {built.sides.right_uuid}
    join_steps = _join_steps(built.plan)
    assert len(join_steps) == 1, "the shape must plan exactly one JoinStep for this to say anything"
    assert join_steps[0].swap_merge_sides is True


def test_a_right_joins_destination_stays_right_when_both_declared_sides_claim_the_destination_framework() -> None:
    """holds_left and holds_right both come out True here; membership alone cannot decide, and the identity
    tiebreaker must still land the RIGHT join's destination on RIGHT."""
    built = _right_join_both_sides_claim_destination_framework()

    record = _one_record(built.plan, built.link)

    assert record.jointype is JoinType.RIGHT
    assert record.destination_side is JoinSide.RIGHT
    assert record.left.uuids == {built.sides.left_uuid}
    # The declared-left parent on the destination framework is not a member of the right side.
    assert record.right.uuids == {built.sides.right_uuid}
    assert record.destination_uuids == {built.sides.right_uuid}
    join_steps = _join_steps(built.plan)
    assert len(join_steps) == 1, "the shape must plan exactly one JoinStep for this to say anything"
    assert join_steps[0].swap_merge_sides is True


def test_a_right_joins_destination_stays_right_when_a_farther_right_parent_shares_the_destination_framework() -> None:
    built = _right_join_nearest_left_and_farther_right_share_destination_framework()

    record = _one_record(built.plan, built.link)

    assert record.jointype is JoinType.RIGHT
    assert record.destination_side is JoinSide.RIGHT
    assert record.destination_framework is PandasDataFrame
    assert record.left.uuids == {built.sides.left_uuid}
    # The declared-left parent on the destination framework is not a member of the right side.
    assert record.right.uuids == {built.sides.right_uuid}
    assert record.destination_uuids == {built.sides.right_uuid}
    join_steps = _join_steps(built.plan)
    assert len(join_steps) == 1, "the shape must plan exactly one JoinStep for this to say anything"
    assert join_steps[0].swap_merge_sides is True


def test_a_right_joins_destination_stays_right_when_both_declared_sides_share_one_framework() -> None:
    """Regression pin for issue #1137 finding #2: a same-framework declared pair with the consumer on a
    third framework must not knock a RIGHT join's destination onto LEFT."""
    built = _right_join_both_declared_sides_share_one_framework_child_on_a_third()

    record = _one_record(built.plan, built.link)

    assert record.jointype is JoinType.RIGHT
    assert record.destination_side is JoinSide.RIGHT
    assert record.destination_uuids == {built.sides.right_uuid}
    assert record.source_uuids == {built.sides.left_uuid}


def test_a_right_joins_destination_stays_right_when_declared_right_is_the_only_pyarrow_exclusive_side() -> None:
    """Regression pin for issue #1137 finding #2 ('hop names PairRight -> PairLeft'): declared right can
    only ever be PyArrow, so the destination must land there even though declared left also offers PyArrow."""
    built = _right_join_declared_left_spans_frameworks_declared_right_is_pyarrow_only()

    record = _one_record(built.plan, built.link)

    assert record.jointype is JoinType.RIGHT
    assert record.destination_side is JoinSide.RIGHT


_THIRD_PARENT_BUILDERS = [
    pytest.param(_link_with_a_third_parent_it_never_mentions, id="pyarrow_from_left"),
    pytest.param(_link_with_a_pandas_third_parent_descending_from_the_right_side, id="pandas_from_right"),
]


@pytest.mark.parametrize("build", _THIRD_PARENT_BUILDERS)
def test_a_parent_the_link_never_mentions_stays_out_of_the_declared_sides(build: Callable[[], ThirdParent]) -> None:
    unlinked = build()

    record = _one_record(unlinked.plan, unlinked.link)

    assert record.destination_side is JoinSide.RIGHT
    assert record.left.uuids == {unlinked.left_uuid}
    assert record.right.uuids == {unlinked.right_uuid}
    assert record.destination_uuids == {unlinked.right_uuid}
    assert record.source_uuids == {unlinked.left_uuid}


@pytest.mark.parametrize("build", _THIRD_PARENT_BUILDERS)
def test_a_declared_side_keeps_only_the_frameworks_its_own_parents_declared(build: Callable[[], ThirdParent]) -> None:
    unlinked = build()

    record = _one_record(unlinked.plan, unlinked.link)

    assert record.left.declared_frameworks == {PyArrowTable}
    assert record.right.declared_frameworks == {PandasDataFrame}
    assert PythonDictFramework not in record.left.declared_frameworks | record.right.declared_frameworks


def test_a_third_parent_sharing_the_destination_framework_now_raises_a_missing_links_error() -> None:
    with pytest.raises(ValueError) as exc_info:
        _link_with_a_declared_left_split_across_frameworks_and_a_colliding_third_parent()

    error_message = str(exc_info.value)
    assert "Link" in error_message, "Error should mention Links as the missing piece of guidance"
    # Either declared side can be the one a merged join-served group reports first, so accept either name.
    assert "ResolvedJoinPairLeft" in error_message or "ResolvedJoinPairRight" in error_message
    assert "ResolvedJoinUnlinked" in error_message


def test_a_third_parent_colliding_in_run_links_final_fallback_also_raises() -> None:
    """Same leak as above, in run_link's other (final fallback) branch."""
    with pytest.raises(ValueError) as exc_info:
        _right_join_farther_right_parent_and_a_colliding_third_parent()

    error_message = str(exc_info.value)
    assert "Link" in error_message, "Error should mention Links as the missing piece of guidance"
    assert "ResolvedJoinPairRight" in error_message
    assert "ResolvedJoinUnlinked" in error_message


def test_a_self_join_gives_each_declared_side_only_its_own_parent() -> None:
    built = _self_join_with_split_declarations()

    record = _one_record(built.plan, built.link)

    assert record.left.uuids == {built.sides.left_uuid}
    assert record.right.uuids == {built.sides.right_uuid}
    assert record.destination.uuids == record.destination_uuids
    assert record.source.uuids == record.source_uuids


def test_a_self_join_keeps_each_parents_framework_candidates_on_its_own_side() -> None:
    built = _self_join_with_split_declarations()

    record = _one_record(built.plan, built.link)

    assert record.left.declared_frameworks == SELF_LEFT_CANDIDATES
    assert record.right.declared_frameworks == SELF_RIGHT_CANDIDATES


def test_a_side_keeps_the_framework_candidates_its_feature_declared_before_the_rewrite() -> None:
    """The graph node carries the one framework the rewrite left; the snapshot carries what was declared."""
    planned = _planned()
    link = _pair_link()
    sides = _branch(planned, link, "resolved_join_candidates")
    built = _finish(
        planned,
        link,
        sides,
        {
            sides.left_uuid: frozenset({PyArrowTable, PandasDataFrame}),
            sides.right_uuid: frozenset({PandasDataFrame}),
        },
    )

    record = _one_record(built.plan, built.link)

    assert record.left.declared_frameworks == {PyArrowTable, PandasDataFrame}
    assert record.right.declared_frameworks == {PandasDataFrame}


def test_consumers_name_the_children_the_orientation_serves() -> None:
    built = _declared_pair()

    record = _one_record(built.plan, built.link)

    assert record.consumers == {built.sides.child_uuid}


def test_a_declined_orientation_builds_no_record_and_one_declined_entry() -> None:
    built = _pair_with_declined_orientation()
    resolved = built.plan.resolved_join_plan

    record = _one_record(built.plan, built.link)

    assert record.destination_framework is PyArrowTable
    assert record.consumers == {built.sides.child_uuid}
    assert resolved.declined == (DeclinedOrientation(built.link.uuid, PandasDataFrame, PyArrowTable),)


def test_a_decline_reached_through_the_inversion_branch_records_the_orientation_it_attempted() -> None:
    """The queue key finds no children, run_link flips it, and the declined entry must name the flipped key."""
    planned, link = _flipped_decline_parts()

    planned.plan.create_execution_plan(planned.queue, planned.graph, planned.link_trekker)

    resolved = planned.plan.resolved_join_plan
    assert len(resolved.records) == 1, "the kept orientation must plan a record for the decline to say anything"
    assert resolved.declined == (DeclinedOrientation(link.uuid, PyArrowTable, PandasDataFrame),)


class Pending(NamedTuple):
    plan: ExecutionPlan
    link: Link
    graph: Graph
    link_trekker: LinkTrekker
    pre_execution_plan: list[Any]


def _pending(planned: Planned, link: Link, declared: DeclaredFrameworks | None = None) -> Pending:
    """The plan state right before add_joinstep: feature groups stepped, no join built."""
    plan = planned.plan
    plan.declared_frameworks = declared or {}
    child_links = plan.invert_link_trekker(planned.link_trekker)
    pre_execution_plan = plan.add_feature_group_step(
        planned.queue, planned.graph.parent_to_children_mapping, child_links, planned.graph.get_nodes()
    )
    return Pending(plan, link, planned.graph, planned.link_trekker, pre_execution_plan)


def _plan_state(plan: ExecutionPlan) -> tuple[list[ResolvedJoin], dict[JoinStep, set[UUID]], list[Any]]:
    return list(plan.planned_records), dict(plan.joinstep_collection.collection), list(plan.declined_orientations)


def _decision_view(decision: JoinDecision) -> tuple[Any, ...]:
    return (
        decision.destination_framework,
        decision.source_framework,
        decision.destination_side,
        decision.carriers,
        decision.shared_source,
        decision.shared_destination,
    )


DECIDE_SHAPES = [
    pytest.param(Link.inner, {}, (PyArrowTable, PandasDataFrame), id="declared"),
    pytest.param(
        Link.inner,
        {"child_cfw": PandasDataFrame, "trekked": (PandasDataFrame, PyArrowTable)},
        (PyArrowTable, PandasDataFrame),
        id="inverted",
    ),
    pytest.param(
        Link.right,
        {"child_cfw": PandasDataFrame},
        (PyArrowTable, PandasDataFrame),
        id="right",
    ),
    pytest.param(
        Link.inner,
        {"trekked": (PandasDataFrame, PyArrowTable)},
        (PyArrowTable, PandasDataFrame),
        id="case_override_inverted",
    ),
]


@pytest.mark.parametrize(("link_factory", "branch_kwargs", "orientation"), DECIDE_SHAPES)
def test_deciding_a_key_reports_what_planning_builds_and_writes_nothing(
    link_factory: Callable[[JoinSpec, JoinSpec], Link],
    branch_kwargs: dict[str, Any],
    orientation: Orientation,
) -> None:
    built_planned = _planned()
    built_link = _pair_link(link_factory)
    built_sides = _branch(built_planned, built_link, "resolved_join_decide_built", **branch_kwargs)
    built = _finish(built_planned, built_link, built_sides)
    record = _one_record(built.plan, built.link)
    step = next(s for s in _join_steps(built.plan) if s.link.uuid == built.link.uuid)

    planned = _planned()
    link = _pair_link(link_factory)
    _branch(planned, link, "resolved_join_decide", **branch_kwargs)
    pending = _pending(planned, link)
    before = _plan_state(pending.plan)
    key = (link, orientation[0], orientation[1])

    decided = pending.plan.decide_link_joins(
        link, [key], pending.link_trekker, pending.graph, pending.pre_execution_plan
    )

    assert list(decided) == [key]
    assert len(decided[key]) == 1
    decision = decided[key][0]
    assert isinstance(decision, JoinDecision)
    assert decision.destination_framework is record.destination_framework
    assert decision.source_framework is record.source_framework
    assert decision.destination_side is record.destination_side
    assert decision.carriers == step.carriers
    assert decision.shared_source == step.shared_source
    assert decision.shared_destination == step.shared_destination
    assert not hasattr(decision, "carried_left")
    assert not hasattr(decision, "bound_buckets")
    assert isinstance(decision.buckets, tuple)
    assert len(decision.buckets) == 2
    assert all(isinstance(bucket, frozenset) for bucket in decision.buckets)
    assert _plan_state(pending.plan) == before

    # Uuids and tokens differ between plans, so the decision is built in its own plan.
    [own_step] = pending.plan._build_decisions([decision], pending.graph)
    own_record = pending.plan.planned_records[-1]
    assert decision.token == own_record.token == own_step.uuid
    assert decision.buckets[0] == own_record.destination_uuids
    assert decision.buckets[1] == own_record.source_uuids


def test_deciding_a_declined_orientation_returns_a_declined_join_and_records_nothing() -> None:
    planned, link, _ = _declined_orientation_parts()
    pending = _pending(planned, link)
    key = (link, PandasDataFrame, PyArrowTable)

    decided = pending.plan.decide_link_joins(
        link, [key], pending.link_trekker, pending.graph, pending.pre_execution_plan
    )

    assert decided[key] == [DeclinedJoin(key)]
    assert pending.plan.declined_orientations == []
    assert pending.plan.planned_records == []


def test_deciding_a_flipped_key_declines_with_the_orientation_it_attempted() -> None:
    planned, link = _flipped_decline_parts()
    pending = _pending(planned, link)
    queued_key = (link, PandasDataFrame, PyArrowTable)

    decided = pending.plan.decide_link_joins(
        link, [queued_key], pending.link_trekker, pending.graph, pending.pre_execution_plan
    )

    assert decided[queued_key] == [DeclinedJoin((link, PyArrowTable, PandasDataFrame))]
    assert pending.plan.declined_orientations == []


def _polymorphic_sides_on_two_frameworks(
    between_keys: Callable[[Planned], None] | None = None,
) -> tuple[Planned, Link, DeclaredFrameworks]:
    """Both link sides have a PyArrow and a Pandas member, so the link carries two trekker keys."""
    planned = _planned()
    link = _pair_link()
    left_arrow = feature("resolved_join_poly_left_arrow", PyArrowTable, link.left_index)
    left_pandas = feature("resolved_join_poly_left_pandas", PandasDataFrame, link.left_index)
    right_arrow = feature("resolved_join_poly_right_arrow", PyArrowTable, link.right_index)
    right_pandas = feature("resolved_join_poly_right_pandas", PandasDataFrame, link.right_index)
    child_arrow = feature("resolved_join_poly_child_arrow", PyArrowTable)
    child_pandas = feature("resolved_join_poly_child_pandas", PandasDataFrame)

    for node, group in (
        (left_arrow, link.left_feature_group),
        (left_pandas, link.left_feature_group),
        (right_arrow, link.right_feature_group),
        (right_pandas, link.right_feature_group),
    ):
        planned.graph.add_node(node.uuid, NodeProperties(node, group))
    planned.queue.append((link.left_feature_group, {left_arrow, left_pandas}))
    planned.queue.append((link.right_feature_group, {right_arrow, right_pandas}))
    planned.queue.append((link, PyArrowTable, PandasDataFrame))
    if between_keys is not None:
        between_keys(planned)
    planned.queue.append((link, PandasDataFrame, PyArrowTable))
    _add_child(planned, child_arrow, left_arrow, right_pandas)
    _add_child(planned, child_pandas, left_pandas, right_arrow)
    trek(planned.link_trekker, link, (PyArrowTable, PandasDataFrame), child_arrow.uuid)
    trek(planned.link_trekker, link, (PandasDataFrame, PyArrowTable), child_pandas.uuid)

    declared: DeclaredFrameworks = {
        left_arrow.uuid: frozenset({PyArrowTable}),
        left_pandas.uuid: frozenset({PandasDataFrame}),
        right_arrow.uuid: frozenset({PyArrowTable}),
        right_pandas.uuid: frozenset({PandasDataFrame}),
    }
    return planned, link, declared


def _link_two_between_the_keys_of_a_polymorphic_link() -> tuple[Planned, Link, Link, DeclaredFrameworks]:
    """A second link on the same two frameworks whose key is queued between the first link's two keys."""
    other = _other_link()

    def queue_other_link(planned: Planned) -> None:
        _branch(planned, other, "resolved_join_interleaved_other", child_cfw=PyArrowTable)

    planned, link, declared = _polymorphic_sides_on_two_frameworks(queue_other_link)
    return planned, link, other, declared


def _edge_view(plan: ExecutionPlan, graph: Graph) -> dict[str, tuple[str, ...]]:
    """Every step's waited-on tokens under stable labels: feature names, joins by link side and orientation."""
    names = {uuid: f"feature:{node.feature.name}" for uuid, node in graph.get_nodes().items()}
    labels: dict[UUID, str] = {}
    for step in plan:
        if isinstance(step, JoinStep):
            labels[step.uuid] = (
                f"join:{step.link.left_feature_group.__name__}:"
                f"{step.destination_framework.__name__}>{step.source_framework.__name__}"
                f"{':carried' if step.carriers else ''}"
            )
    for step in plan:
        if isinstance(step, FeatureGroupStep):
            labels[step.uuid] = "step:" + ",".join(sorted(names[u] for u in step.get_uuids()))
    for step in plan:
        if isinstance(step, TransformFrameworkStep):
            if step.link_id is not None:
                labels[step.uuid] = f"hop:{labels[step.link_id]}"
            else:
                assert step.source_step_uuid is not None
                labels[step.uuid] = f"hop:{labels[step.source_step_uuid]}>{step.to_feature_group.__name__}"
    assert len(set(labels.values())) == len(labels), f"edge labels must be unique: {sorted(labels.values())}"

    def label(uuid: UUID) -> str:
        return labels[uuid] if uuid in labels else names[uuid]

    return {labels[step.uuid]: tuple(sorted(label(u) for u in step.get_wait_uuids())) for step in plan}


def _reachable(edges: dict[str, tuple[str, ...]]) -> dict[str, frozenset[str]]:
    """Each step's transitive waits, so a redundant direct edge does not change the answer."""

    def closure(start: str) -> frozenset[str]:
        seen: set[str] = set()
        stack = [start]
        while stack:
            for waited in edges.get(stack.pop(), ()):
                if waited not in seen:
                    seen.add(waited)
                    stack.append(waited)
        return frozenset(seen)

    return {step: closure(step) for step in edges}


def test_deciding_both_keys_of_a_link_together_equals_deciding_each_alone() -> None:
    planned, link, declared = _polymorphic_sides_on_two_frameworks()
    pending = _pending(planned, link, declared)
    keys = list(pending.link_trekker.data)
    assert len(keys) == 2

    def decide(wanted: list[Any]) -> dict[Any, list[Any]]:
        return pending.plan.decide_link_joins(
            link, wanted, pending.link_trekker, pending.graph, pending.pre_execution_plan
        )

    together = decide(keys)
    assert list(together) == keys
    for key in keys:
        alone = decide([key])[key]
        assert [_decision_view(d) for d in together[key]] == [_decision_view(d) for d in alone]
        assert len(alone) == 1
        assert alone[0].order == 0
        assert alone[0].runs_after == frozenset()
    assert together[keys[1]][0].runs_after


def test_a_link_with_two_trekker_keys_still_plans_one_record_per_key() -> None:
    planned, link, declared = _polymorphic_sides_on_two_frameworks()
    pending = _pending(planned, link, declared)
    keys = list(pending.link_trekker.data)
    decided = pending.plan.decide_link_joins(
        link, keys, pending.link_trekker, pending.graph, pending.pre_execution_plan
    )

    fresh, fresh_link, fresh_declared = _polymorphic_sides_on_two_frameworks()
    fresh.plan.create_execution_plan(fresh.queue, fresh.graph, fresh.link_trekker, fresh_declared)
    records = _records(fresh.plan, fresh_link)

    decisions = [decision for key in keys for decision in decided[key]]
    assert all(isinstance(decision, JoinDecision) for decision in decisions)
    pairs = [
        (decision.destination_framework, decision.source_framework)
        for decision in decisions
        if isinstance(decision, JoinDecision)
    ]
    assert len(records) == 2
    assert [(r.destination_framework, r.source_framework) for r in records] == pairs

    joins = [decision for decision in decisions if isinstance(decision, JoinDecision)]
    step_uuids = {step.uuid for step in pending.pre_execution_plan if isinstance(step, FeatureGroupStep)}
    assert [decision.order for decision in joins] == [0, 1]
    assert joins[0].runs_after == frozenset()
    assert joins[1].runs_after == frozenset({joins[0].token})
    for decision in joins:
        assert isinstance(decision.destination_owner, frozenset)
        assert decision.destination_owner
        assert decision.destination_owner <= step_uuids

    fresh_joins = [step for step in _join_steps(fresh.plan) if step.link.uuid == fresh_link.uuid]
    assert len(fresh_joins) == 2
    first, second = fresh_joins
    assert [record.token for record in records] == [first.uuid, second.uuid]
    assert first.runs_after_joins == frozenset()
    assert second.runs_after_joins == frozenset({first.uuid})
    assert fresh.plan.joinstep_collection.get_required_join_uuids(second) == set()
    assert first.uuid in second.required_uuids


_ARROW_TO_PANDAS = "join:ResolvedJoinPairLeft:PyArrowTable>PandasDataFrame"
_PANDAS_TO_ARROW = "join:ResolvedJoinPairLeft:PandasDataFrame>PyArrowTable"
_OTHER_ARROW_TO_PANDAS = "join:ResolvedJoinOtherLeft:PyArrowTable>PandasDataFrame"


def _waits(*labels: str) -> tuple[str, ...]:
    return tuple(sorted(labels))


def _poly(name: str) -> str:
    return f"feature:resolved_join_poly_{name}"


_POLY_EDGES = {
    f"step:{_poly('left_arrow')}": _waits(),
    f"step:{_poly('left_pandas')}": _waits(),
    f"step:{_poly('right_arrow')}": _waits(),
    f"step:{_poly('right_pandas')}": _waits(),
    f"hop:{_ARROW_TO_PANDAS}": _waits(_poly("left_arrow"), _poly("right_pandas")),
    f"hop:{_PANDAS_TO_ARROW}": _waits(_poly("left_pandas"), _poly("right_arrow")),
    _ARROW_TO_PANDAS: _waits(_poly("left_arrow"), _poly("right_pandas"), f"hop:{_ARROW_TO_PANDAS}"),
    _PANDAS_TO_ARROW: _waits(_poly("left_pandas"), _poly("right_arrow"), f"hop:{_PANDAS_TO_ARROW}", _ARROW_TO_PANDAS),
    f"step:{_poly('child_arrow')}": _waits(_poly("left_arrow"), _poly("right_pandas"), _ARROW_TO_PANDAS),
    f"step:{_poly('child_pandas')}": _waits(_poly("left_pandas"), _poly("right_arrow"), _PANDAS_TO_ARROW),
}


def test_the_two_joins_of_a_polymorphic_link_keep_their_planned_edges() -> None:
    planned, _, declared = _polymorphic_sides_on_two_frameworks()
    planned.plan.create_execution_plan(planned.queue, planned.graph, planned.link_trekker, declared)

    assert _edge_view(planned.plan, planned.graph) == _POLY_EDGES


def _other(name: str) -> str:
    return f"feature:resolved_join_interleaved_other_{name}"


_INTERLEAVED_EDGES = {
    **_POLY_EDGES,
    f"step:{_other('left')}": _waits(),
    f"step:{_other('right')}": _waits(),
    f"hop:{_OTHER_ARROW_TO_PANDAS}": _waits(_other("left"), _other("right")),
    _OTHER_ARROW_TO_PANDAS: _waits(_other("left"), _other("right"), f"hop:{_OTHER_ARROW_TO_PANDAS}", _ARROW_TO_PANDAS),
    f"step:{_other('child')}": _waits(_other("left"), _other("right"), _OTHER_ARROW_TO_PANDAS),
    _PANDAS_TO_ARROW: _waits(
        _poly("left_pandas"), _poly("right_arrow"), f"hop:{_PANDAS_TO_ARROW}", _OTHER_ARROW_TO_PANDAS
    ),
}


def test_a_join_of_another_link_between_the_keys_of_one_link_keeps_the_planned_edges() -> None:
    """Build order is link one's first join, the other link's join, link one's second join."""
    planned, _, _, declared = _link_two_between_the_keys_of_a_polymorphic_link()
    planned.plan.create_execution_plan(planned.queue, planned.graph, planned.link_trekker, declared)

    edges = _edge_view(planned.plan, planned.graph)

    assert [step.link.left_feature_group.__name__ for step in _join_steps(planned.plan)] == [
        "ResolvedJoinPairLeft",
        "ResolvedJoinOtherLeft",
        "ResolvedJoinPairLeft",
    ]
    assert edges == _INTERLEAVED_EDGES
    assert _ARROW_TO_PANDAS not in edges[_PANDAS_TO_ARROW]
    assert _reachable(edges) == _reachable(_INTERLEAVED_EDGES)


def _carrier_split_planned(carried_first: bool, after_carried: bool = False) -> Planned:
    """A consumer reading the left side through a Pandas mid and a direct one: two same-framework joins, one frame.

    The consumers are decided in uuid order, so their uuids are fixed to pin which join is planned first.
    With after_carried the direct consumer also reads a step below the carried consumer.
    """
    planned = _planned()
    link = _pair_link()
    left = feature("resolved_join_split_left", PyArrowTable, link.left_index)
    right = feature("resolved_join_split_right", PyArrowTable, link.right_index)
    mid = feature("resolved_join_split_mid", PandasDataFrame)
    carried = feature("resolved_join_split_carried", PyArrowTable)
    direct = feature("resolved_join_split_direct", PyArrowTable)
    carried.uuid, direct.uuid = (UUID(int=1), UUID(int=2)) if carried_first else (UUID(int=2), UUID(int=1))
    _add_parents(planned, link, left, right)
    planned.graph.add_node(mid.uuid, NodeProperties(mid, ResolvedJoinUnlinked))
    planned.graph.adjacency_list[left.uuid].append(mid.uuid)
    planned.graph.adjacency_list[mid.uuid] = []
    planned.graph.parent_to_children_mapping[mid.uuid] = {left.uuid}
    planned.queue.append((ResolvedJoinUnlinked, {mid}))
    planned.queue.append((link, PyArrowTable, PyArrowTable))
    _add_child(planned, carried, mid, left, right)
    if after_carried:
        after = feature("resolved_join_split_after", PyArrowTable)
        planned.graph.add_node(after.uuid, NodeProperties(after, ResolvedJoinUnlinked))
        planned.graph.adjacency_list[carried.uuid].append(after.uuid)
        planned.graph.adjacency_list[after.uuid] = []
        planned.graph.parent_to_children_mapping[after.uuid] = {carried.uuid}
        planned.queue.append((ResolvedJoinUnlinked, {after}))
        _add_child(planned, direct, after, left, right)
        planned.graph.adjacency_list[after.uuid].append(direct.uuid)
    else:
        _add_child(planned, direct, left, right)
    for consumer in (carried, direct):
        trek(planned.link_trekker, link, (PyArrowTable, PyArrowTable), consumer.uuid)
    return planned


def _declared_by_own_framework(planned: Planned) -> DeclaredFrameworks:
    return {uuid: frozenset({node.feature.get_compute_framework()}) for uuid, node in planned.graph.get_nodes().items()}


_SPLIT_DIRECT = "join:ResolvedJoinPairLeft:PyArrowTable>PyArrowTable"
_SPLIT_CARRIED = f"{_SPLIT_DIRECT}:carried"


def _split(name: str) -> str:
    return f"feature:resolved_join_split_{name}"


_SPLIT_HOP_LEFT = f"hop:step:{_split('left')}>ResolvedJoinPairLeft"
_SPLIT_HOP_MID = f"hop:step:{_split('mid')}>ResolvedJoinPairLeft"
_SPLIT_HOP_UNLINKED = f"hop:step:{_split('left')}>ResolvedJoinUnlinked"
_SPLIT_BASE_EDGES = {
    f"step:{_split('left')}": _waits(),
    f"step:{_split('right')}": _waits(),
    f"step:{_split('mid')}": _waits(_split("left"), _SPLIT_HOP_UNLINKED),
    _SPLIT_HOP_UNLINKED: _waits(_split("left")),
    _SPLIT_HOP_LEFT: _waits(_split("left")),
    _SPLIT_HOP_MID: _waits(_split("mid")),
    f"step:{_split('carried')}": _waits(_split("left"), _split("mid"), _split("right"), _SPLIT_CARRIED),
    f"step:{_split('direct')}": _waits(_split("left"), _split("right"), _SPLIT_DIRECT),
}
_SPLIT_DIRECT_WAITS = (_split("left"), _split("right"), _SPLIT_HOP_LEFT)
_SPLIT_CARRIED_WAITS = (_split("left"), _split("mid"), _split("right"), _SPLIT_HOP_MID)
# Each case pins which join is planned first and so waits for nothing of its link; the other waits for it.
_SPLIT_EDGES = {
    False: {
        **_SPLIT_BASE_EDGES,
        _SPLIT_DIRECT: _waits(*_SPLIT_DIRECT_WAITS),
        _SPLIT_CARRIED: _waits(*_SPLIT_CARRIED_WAITS, _SPLIT_DIRECT),
    },
    True: {
        **_SPLIT_BASE_EDGES,
        _SPLIT_DIRECT: _waits(*_SPLIT_DIRECT_WAITS, _SPLIT_CARRIED),
        _SPLIT_CARRIED: _waits(*_SPLIT_CARRIED_WAITS),
    },
}


# The direct join also reaches the carried join through a step below the carried consumer.
_SPLIT_CHAINED_EDGES = {
    **_SPLIT_EDGES[True],
    f"step:{_split('after')}": _waits(_split("carried")),
    _SPLIT_HOP_LEFT: _waits(_split("after"), _split("left")),
    _SPLIT_DIRECT: _waits(_split("after"), *_SPLIT_DIRECT_WAITS, _SPLIT_CARRIED),
    f"step:{_split('direct')}": _waits(_split("after"), _split("left"), _split("right"), _SPLIT_DIRECT),
}


@pytest.mark.parametrize(
    ("carried_first", "after_carried"),
    [(False, False), (True, False), (True, True)],
    ids=["direct_first", "carried_first", "carried_first_direct_reads_below_carried"],
)
def test_the_two_same_framework_joins_of_a_carrier_split_keep_their_planned_edges(
    carried_first: bool, after_carried: bool
) -> None:
    """Two joins of one link into one frame stay ordered by a direct edge, as in the planner before ordering moved."""
    expected = _SPLIT_CHAINED_EDGES if after_carried else _SPLIT_EDGES[carried_first]
    planned = _carrier_split_planned(carried_first, after_carried)
    planned.plan.create_execution_plan(
        planned.queue, planned.graph, planned.link_trekker, _declared_by_own_framework(planned)
    )

    edges = _edge_view(planned.plan, planned.graph)
    first, second = (_SPLIT_CARRIED, _SPLIT_DIRECT) if carried_first else (_SPLIT_DIRECT, _SPLIT_CARRIED)

    assert [bool(step.carriers) for step in _join_steps(planned.plan)] == [carried_first, not carried_first]
    assert edges == expected
    assert first in edges[second]
    assert _reachable(edges) == _reachable(expected)


@pytest.mark.parametrize(
    "build",
    [
        _declared_pair,
        _inverted_pair,
        _right_join,
        _inverted_left_join,
        _self_join,
        _self_join_with_split_declarations,
        _append_pair,
        _two_links,
        _link_with_a_third_parent_it_never_mentions,
        _link_with_a_pandas_third_parent_descending_from_the_right_side,
        _case_override_inverted,
        _case_override_beats_nearer_wrong_framework_left,
        _case_override_disagrees_with_the_nearest_split,
        _right_join_farther_right_parent_holds_destination_framework,
        _right_join_both_sides_claim_destination_framework,
        _right_join_both_declared_sides_share_one_framework_child_on_a_third,
        _right_join_declared_left_spans_frameworks_declared_right_is_pyarrow_only,
    ],
    ids=[
        "inner",
        "inverted_inner",
        "right",
        "inverted_left",
        "self_join",
        "self_join_split_declarations",
        "append",
        "two_links",
        "unlinked_third_parent",
        "unlinked_pandas_third_parent_from_right",
        "case_override_inverted",
        "case_override_beats_nearer_wrong_framework_left",
        "case_override_disagrees_with_nearest_split",
        "right_join_farther_right_parent_holds_destination_framework",
        "right_join_both_sides_claim_destination_framework",
        "right_join_both_declared_sides_share_one_framework",
        "right_join_declared_right_is_pyarrow_exclusive",
    ],
)
def test_the_records_sign_the_joins_the_join_steps_sign(build: Callable[[], Any]) -> None:
    built = build()

    join_steps = _join_steps(built.plan)
    resolved = built.plan.resolved_join_plan

    assert join_steps, "the shape must plan at least one JoinStep for the parity to say anything"
    assert len(resolved.records) == len(join_steps)
    assert resolved.signatures() == built.plan.join_signatures_at_build
    assert {record.token for record in resolved.records} == {step.uuid for step in join_steps}


def test_raise_on_join_plan_divergence_raises_on_a_mutated_step() -> None:
    from mloda.core.prepare.resolved_join_builder import raise_on_join_plan_divergence

    built = _declared_pair()
    join_steps = _join_steps(built.plan)

    assert raise_on_join_plan_divergence(built.plan.resolved_join_plan, join_steps) is None

    join_steps[0].swap_merge_sides = not join_steps[0].swap_merge_sides

    with pytest.raises(ValueError):
        raise_on_join_plan_divergence(built.plan.resolved_join_plan, join_steps)


def _with_record(built: Built, **changes: Any) -> ResolvedJoinPlan:
    record = _one_record(built.plan, built.link)
    return replace(built.plan.resolved_join_plan, records=(replace(record, **changes),))


def _wrong_side_membership() -> tuple[Built, dict[str, Any]]:
    built = _declared_pair()
    record = _one_record(built.plan, built.link)
    return built, {"right": replace(record.right, uuids=frozenset({built.sides.left_uuid}))}


def _cross_framework_overlap() -> tuple[Built, dict[str, Any]]:
    built = _declared_pair()
    return built, {
        "source_uuids": frozenset({built.sides.left_uuid}),
        "destination_uuids": frozenset({built.sides.left_uuid}),
    }


def _ends_swapped() -> tuple[Built, dict[str, Any]]:
    built = _declared_pair()
    return built, {
        "destination_uuids": frozenset({built.sides.right_uuid}),
        "source_uuids": frozenset({built.sides.left_uuid}),
    }


def _same_framework_overlap() -> tuple[Built, dict[str, Any]]:
    built = _right_join_both_declared_sides_share_one_framework_child_on_a_third()
    shared = frozenset({built.sides.left_uuid})
    return built, {"source_uuids": shared, "destination_uuids": shared}


def _destination_holds_declared_left_member() -> tuple[Built, dict[str, Any]]:
    built = _right_join_both_sides_claim_destination_framework()
    # extra_right_uuid holds nearest_left, a declared-left parent, in this shape.
    assert built.sides.extra_right_uuid is not None
    return built, {"destination_uuids": frozenset({built.sides.right_uuid, built.sides.extra_right_uuid})}


def _destination_shares_none_with_its_side() -> tuple[Built, dict[str, Any]]:
    built = _right_join_nearest_left_and_farther_right_share_destination_framework()
    nodes = built.graph.get_nodes()
    [nearest_right] = [uuid for uuid, node in nodes.items() if node.feature_group_class is ResolvedJoinPairRight]
    return built, {"destination_uuids": frozenset({nearest_right})}


def _append_overlap() -> tuple[Built, dict[str, Any]]:
    built = _append_pair()
    shared = frozenset({built.sides.left_uuid})
    return built, {"source_uuids": shared, "destination_uuids": shared}


@pytest.mark.parametrize(
    "build, message, carried",
    [
        pytest.param(
            _wrong_side_membership,
            "right side holds members of ['ResolvedJoinPairLeft']",
            False,
            id="right_side_holds_left_group_uuid",
        ),
        pytest.param(_cross_framework_overlap, "Both name [", False, id="cross_framework_overlap"),
        pytest.param(
            _ends_swapped,
            "destination uuids hold members of ['ResolvedJoinPairRight']",
            False,
            id="destination_holds_source_group_uuid",
        ),
        pytest.param(_same_framework_overlap, None, False, id="same_framework_overlap_exempt"),
        pytest.param(_same_framework_overlap, "Both name [", True, id="same_framework_overlap_with_carriers"),
        pytest.param(_append_overlap, None, False, id="append_overlap_exempt"),
        pytest.param(
            _destination_holds_declared_left_member,
            "destination uuids hold members of ['ResolvedJoinPairLeft']",
            False,
            id="destination_holds_declared_left_member",
        ),
        pytest.param(
            _destination_shares_none_with_its_side,
            "share none with the side",
            False,
            id="destination_shares_none_with_its_side",
        ),
    ],
)
def test_raise_on_dishonest_join_record(
    build: Callable[[], tuple[Built, dict[str, Any]]], message: str | None, carried: bool
) -> None:
    from mloda.core.prepare.validate_resolved_join import raise_on_dishonest_join_record

    built, changes = build()
    join_steps = _join_steps(built.plan)
    raise_on_dishonest_join_record(built.plan.resolved_join_plan, join_steps, built.graph)

    dishonest = _with_record(built, **changes)
    if carried:
        join_steps[0].carriers = frozenset({built.sides.left_uuid})

    if message is None:
        raise_on_dishonest_join_record(dishonest, join_steps, built.graph)
        return
    with pytest.raises(ValueError, match=r"Internal error") as exc_info:
        raise_on_dishonest_join_record(dishonest, join_steps, built.graph)
    assert message in str(exc_info.value)


def test_a_nearest_left_parent_on_a_third_framework_is_refused_as_a_dishonest_record() -> None:
    """The engine rejects this shape earlier; the record check refuses it if planning is reached."""
    planned = _planned()
    link = _pair_link()

    descendant = feature("resolved_join_third_fw_descendant", PandasDataFrame, link.left_index)
    nearest_left = feature("resolved_join_third_fw_nearest_left", PythonDictFramework, link.left_index)
    right = feature("resolved_join_third_fw_right", PyArrowTable, link.right_index)
    child = feature("resolved_join_third_fw_child", PandasDataFrame)

    planned.graph.add_node(descendant.uuid, NodeProperties(descendant, ResolvedJoinPairLeftDescendant))
    planned.graph.add_node(nearest_left.uuid, NodeProperties(nearest_left, ResolvedJoinPairLeft))
    planned.graph.add_node(right.uuid, NodeProperties(right, ResolvedJoinPairRight))
    planned.queue.append((ResolvedJoinPairLeftDescendant, {descendant}))
    planned.queue.append((ResolvedJoinPairLeft, {nearest_left}))
    planned.queue.append((ResolvedJoinPairRight, {right}))
    planned.queue.append((link, PyArrowTable, PandasDataFrame))
    _add_child(planned, child, descendant, nearest_left, right)
    trek(planned.link_trekker, link, (PandasDataFrame, PyArrowTable), child.uuid)

    with pytest.raises(ValueError, match="Internal error"):
        planned.plan.create_execution_plan(planned.queue, planned.graph, planned.link_trekker)


def test_a_case_override_survives_a_right_destination_side() -> None:
    """is_valid_join_step's case helper already resolves left/right in declared order; the inversion remap must
    not re-swap them once destination_side comes out RIGHT."""
    built = _case_override_inverted()

    assert len(_join_steps(built.plan)) == 1, "the shape must plan exactly one JoinStep for this to say anything"
    record = _one_record(built.plan, built.link)
    assert record.destination_side is JoinSide.RIGHT
    assert record.left.uuids == {built.sides.left_uuid}
    assert record.right.uuids == {built.sides.right_uuid}


def test_a_case_override_beats_a_nearer_wrong_framework_left() -> None:
    """split_by_declared_side's nearest-by-distance rule alone would pick nearest_left; the case helper's
    framework filter must override it and keep far_left, the correct-framework parent, on record.left."""
    built = _case_override_beats_nearer_wrong_framework_left()

    assert len(_join_steps(built.plan)) == 1, "the shape must plan exactly one JoinStep for this to say anything"
    record = _one_record(built.plan, built.link)
    assert record.destination_side is JoinSide.RIGHT
    assert record.left.uuids == {built.sides.left_uuid}
    assert record.right.uuids == {built.sides.right_uuid}
    assert record.destination.uuids <= record.destination_uuids
    assert record.source.uuids <= record.source_uuids


def test_a_right_join_plans_when_the_consumer_shares_link_fw1_and_the_groups_are_unrelated() -> None:
    built = _right_join_consumer_shares_declared_left_framework()

    record = _one_record(built.plan, built.link)

    assert record.jointype is JoinType.RIGHT
    assert record.destination_side is JoinSide.LEFT
    assert record.left.uuids == {built.sides.left_uuid}
    assert record.right.uuids == {built.sides.right_uuid}


def test_a_right_join_still_rejects_genuine_polymorphic_left_ambiguity() -> None:
    with pytest.raises(
        Exception,
        match=r"(?s)members of one link side.*one member of the link side per consumer.*declare one link per member",
    ):
        _right_join_ambiguous_polymorphic_left_candidates()


def test_a_case_override_right_destination_matches_the_destination_uuids() -> None:
    """A RIGHT-destination case override keeps destination_uuids in step with destination."""
    built = _case_override_inverted()

    record = _one_record(built.plan, built.link)

    assert record.destination_side is JoinSide.RIGHT
    assert record.destination.uuids <= record.destination_uuids
    assert record.source.uuids <= record.source_uuids


def test_a_case_override_disagreeing_with_the_nearest_split_still_binds_the_selected_framework() -> None:
    built = _case_override_disagrees_with_the_nearest_split()

    assert len(_join_steps(built.plan)) == 1, "the shape must plan exactly one JoinStep for this to say anything"
    record = _one_record(built.plan, built.link)
    assert record.destination_framework is PyArrowTable
    assert record.destination_uuids == {built.sides.left_uuid}
    assert record.source_uuids == {built.sides.right_uuid}
    assert record.destination.uuids <= record.destination_uuids
    assert record.source.uuids <= record.source_uuids


def test_a_case_override_binds_the_destination_side_from_the_selected_parents() -> None:
    """The case helper binds far_left to link.left and far_right to link.right; destination_side must follow
    that binding, not the nearest split's frameworks, which never saw the case-selected, farther parents."""
    built = _case_override_disagrees_with_the_nearest_split()

    join_steps = _join_steps(built.plan)
    assert len(join_steps) == 1, "the shape must plan exactly one JoinStep for this to say anything"
    record = _one_record(built.plan, built.link)

    assert record.destination_side is JoinSide.LEFT
    assert record.left.uuids == {built.sides.left_uuid}
    assert record.right.uuids == {built.sides.right_uuid}
    assert record.destination.uuids <= record.destination_uuids
    assert record.source.uuids <= record.source_uuids
    assert join_steps[0].swap_merge_sides is False


def test_a_case_override_hop_moves_the_right_group_into_the_left_groups_framework() -> None:
    """The destination holds the declared left side here, so the hop must move the right group's data into it."""
    built = _case_override_disagrees_with_the_nearest_split()

    transform_steps = _transform_steps(built.plan, built.link)

    assert len(transform_steps) == 1, "the shape must plan exactly one hop for this to say anything"
    assert transform_steps[0].from_feature_group is ResolvedJoinPairRight
    assert transform_steps[0].to_feature_group is ResolvedJoinPairLeft
    assert transform_steps[0].from_framework is PandasDataFrame
    assert transform_steps[0].to_framework is PyArrowTable


def test_a_case_override_keeps_each_sides_declared_frameworks_on_its_own_side() -> None:
    built = _case_override_disagrees_with_the_nearest_split(with_declared_frameworks=True)

    record = _one_record(built.plan, built.link)

    assert record.left.declared_frameworks == {PyArrowTable}
    assert record.right.declared_frameworks == {PandasDataFrame}


def test_the_record_leaves_out_the_write_serialization_edges_add_tfs_adds() -> None:
    built = _two_links()

    recorded = built.plan.resolved_join_plan.signatures()
    after_tfs = joinstep_signatures(_join_steps(built.plan))

    assert after_tfs != recorded, "add_tfs must add an edge here for this to say anything"
    assert _without_depends(after_tfs) == _without_depends(recorded)


def test_a_second_planning_pass_does_not_accumulate_records() -> None:
    planned = _planned()
    link = _pair_link()
    _branch(planned, link, "resolved_join_twice")
    planned.plan.create_execution_plan(planned.queue, planned.graph, planned.link_trekker)
    first = len(planned.plan.resolved_join_plan.records)
    assert len(_transform_steps(planned.plan, link)) == 1, "the cross-framework join must plan one hop on pass one"

    planned.plan.create_execution_plan(planned.queue, planned.graph, planned.link_trekker)

    assert len(planned.plan.resolved_join_plan.records) == first
    assert len(planned.plan.resolved_join_plan.records) == len(_join_steps(planned.plan))
    assert len(_transform_steps(planned.plan, link)) == 1


def test_an_order_edge_makes_the_consumer_records_depend_on_the_producer_record_tokens() -> None:
    chain = _ordered_chain()
    resolved = chain.plan.resolved_join_plan
    producer_records = resolved.records_of_link(chain.producer.uuid)
    consumer_records = resolved.records_of_link(chain.consumer.uuid)

    assert producer_records, "the producer link must build a record for the edge to point at"
    assert consumer_records, "the consumer link must build a record for the edge to hang off"
    for record in producer_records:
        assert not record.depends_on
    for record in consumer_records:
        assert record.depends_on == {produced.token for produced in producer_records}
        assert chain.producer.uuid not in record.depends_on, "a record depends on tokens, not on link uuids"
        assert record.signature(resolved.link_of_token()).depends_on_links == (str(chain.producer.uuid),)


@pytest.mark.parametrize(
    "build",
    [_declared_pair, _inverted_pair, _right_join, _inverted_left_join],
    ids=["inner", "inverted_inner", "right", "inverted_left"],
)
def test_the_transform_hop_moves_the_source_side_into_the_destination_side(build: Callable[[], Any]) -> None:
    """The hop names the record's own direction, not the one the jointype implies."""
    built = build()

    record = _one_record(built.plan, built.link)
    transform_steps = _transform_steps(built.plan, built.link)

    assert len(transform_steps) == 1, "the shape must plan exactly one hop for this to say anything"
    assert transform_steps[0].from_feature_group is record.transform_from_feature_group
    assert transform_steps[0].to_feature_group is record.transform_to_feature_group


def test_an_inverted_hop_carries_data_into_the_declared_right_side() -> None:
    """The declared right side holds the destination here, so the hop moves the left side into it."""
    built = _inverted_pair()

    record = _one_record(built.plan, built.link)
    transform_steps = _transform_steps(built.plan, built.link)

    assert record.destination_side is JoinSide.RIGHT
    assert len(transform_steps) == 1
    assert transform_steps[0].from_feature_group is ResolvedJoinPairLeft
    assert transform_steps[0].to_feature_group is ResolvedJoinPairRight


def test_a_real_engine_plan_carries_one_record_per_planned_join_step() -> None:
    link = Link.inner(
        JoinSpec(ResolvedJoinEndLeft, Index((END_LEFT_KEY,))),
        JoinSpec(ResolvedJoinEndRight, Index((END_RIGHT_KEY,))),
    )
    engine = Engine(
        Features([Feature(name=ResolvedJoinEndChild.get_class_name())]),
        {PyArrowTable, PandasDataFrame},
        {link},
        plugin_collector=PluginCollector.enabled_feature_groups(
            {ResolvedJoinEndLeft, ResolvedJoinEndRight, ResolvedJoinEndChild}
        ),
    )

    plan = engine.execution_planner
    join_steps = _join_steps(plan)
    resolved = plan.resolved_join_plan

    assert len(join_steps) == 1
    assert len(resolved.records) == len(join_steps)
    assert resolved.signatures() == plan.join_signatures_at_build
    for record in resolved.records:
        assert record.destination_framework in record.destination.declared_frameworks


def test_the_resolver_snapshots_the_frameworks_a_feature_declared_and_never_narrows_them() -> None:
    link = _pair_link()
    left = feature("resolved_join_snapshot_left", PyArrowTable, link.left_index)
    right = feature("resolved_join_snapshot_right", PandasDataFrame, link.right_index)
    child = Feature("resolved_join_snapshot_child")
    child.compute_frameworks = {PyArrowTable, PandasDataFrame}
    child.chosen_compute_framework = PyArrowTable

    link_trekker = LinkTrekker()
    trekked = {child.uuid}
    link_trekker.data[(link, PyArrowTable, PandasDataFrame)] = trekked
    link_trekker.data_ordered[(link, PyArrowTable, PandasDataFrame)] = trekked
    queue: list[Any] = [
        (ResolvedJoinPairLeft, {left}),
        (ResolvedJoinPairRight, {right}),
        (link, PyArrowTable, PandasDataFrame),
        (ResolvedJoinChild, {child}),
    ]

    resolver = ResolveComputeFrameworks(Graph())
    resolver.links(queue, link_trekker)

    assert child.compute_frameworks == {PyArrowTable, PandasDataFrame}, "resolution never narrows the allowed set"
    assert resolver.get_declared_frameworks()[child.uuid] == {PyArrowTable, PandasDataFrame}
    assert resolver.get_declared_frameworks()[left.uuid] == {PyArrowTable}


# Fresh interpreters are slow to start, so this needs more than the suite-wide timeout.
@pytest.mark.timeout(60)
def test_fresh_interpreters_build_the_same_record_signature() -> None:
    outputs = run_probes(_PROBE, _PROBE_PROCESSES)

    assert len(outputs) == _PROBE_PROCESSES, f"expected {_PROBE_PROCESSES} probe results, got {len(outputs)}"
    for position, output in enumerate(outputs):
        assert output == _PROBE_EXPECTED, f"probe {position} signed {output}, expected {_PROBE_EXPECTED}"


def test_an_inner_joins_ambiguous_split_is_refused_as_a_dishonest_record() -> None:
    """The engine rejects this shape earlier; the record check refuses it if planning is reached."""
    with pytest.raises(ValueError, match="Internal error"):
        _inner_join_ambiguous_split_stays_on_the_tiebreak_answer()


def test_a_right_joins_declared_left_side_is_not_double_counted_as_a_right_side_claim() -> None:
    built = _right_join_declared_right_subclass_also_matches_declared_left()

    record = _one_record(built.plan, built.link)

    assert record.jointype is JoinType.RIGHT
    assert record.destination_side is JoinSide.LEFT
    join_steps = _join_steps(built.plan)
    assert len(join_steps) == 1, "the shape must plan exactly one JoinStep for this to say anything"
    assert join_steps[0].swap_merge_sides is False


BASE_JOINS = 100
MAX_CALL_RATIO_FOR_DOUBLED_JOINS = 3
PAIR_SHAPES = ("cross_framework", "same_framework")


class IndependentPairs(NamedTuple):
    planned: Planned
    links: list[Link]
    sides: list[Sides]


def _independent_pairs(count: int, shape: str) -> IndependentPairs:
    """``count`` pairs sharing nothing; ``same_framework`` puts every feature on PyArrowTable."""
    planned = _planned()
    links: list[Link] = []
    all_sides: list[Sides] = []
    for i in range(count):
        if shape == "mixed":
            right_cfw: type[ComputeFramework] = PandasDataFrame if i % 2 == 0 else PyArrowTable
        else:
            right_cfw = PyArrowTable if shape == "same_framework" else PandasDataFrame
        left_fg = type(f"IndependentPairLeft{shape}{i}", (FeatureGroup,), {})
        right_fg = type(f"IndependentPairRight{shape}{i}", (FeatureGroup,), {})
        link = Link.inner(
            JoinSpec(left_fg, Index((f"independent_{shape}_left_key_{i}",))),
            JoinSpec(right_fg, Index((f"independent_{shape}_right_key_{i}",))),
        )
        links.append(link)
        all_sides.append(_branch(planned, link, f"independent_{shape}_{i}", right_cfw=right_cfw))
    return IndependentPairs(planned, links, all_sides)


def _total_calls(work: Callable[..., Any], *args: Any) -> int:
    profiler = cProfile.Profile()
    profiler.runcall(work, *args)
    return int(pstats.Stats(profiler).total_calls)  # type: ignore[attr-defined]


def _planning_calls(count: int, shape: str) -> tuple[int, IndependentPairs]:
    """Plan ``count`` independent joins under a profiler and return the total function calls made."""
    pairs = _independent_pairs(count, shape)
    planned = pairs.planned
    calls = _total_calls(planned.plan.create_execution_plan, planned.queue, planned.graph, planned.link_trekker, {})
    return calls, pairs


def _destination_lookup_calls(count: int) -> int:
    planned = _independent_pairs(count, "same_framework").planned
    planned.plan.create_execution_plan(planned.queue, planned.graph, planned.link_trekker, {})
    coll = planned.plan.joinstep_collection

    def lookups() -> None:
        index = coll.joins_by_destination_owner()
        for join in coll.collection:
            coll.earlier_joins_sharing_destination(join, None, index)

    return _total_calls(lookups)


def test_destination_lookups_grow_linearly_with_independent_joins() -> None:
    warm = _independent_pairs(2, "same_framework").planned
    warm.plan.create_execution_plan(warm.queue, warm.graph, warm.link_trekker, {})

    calls_at_base = _destination_lookup_calls(BASE_JOINS)
    calls_at_double = _destination_lookup_calls(2 * BASE_JOINS)

    assert calls_at_double < MAX_CALL_RATIO_FOR_DOUBLED_JOINS * calls_at_base, (
        f"calls at {BASE_JOINS} joins: {calls_at_base}, at {2 * BASE_JOINS}: {calls_at_double}"
    )


def test_every_cross_framework_join_in_a_mixed_plan_still_waits_for_every_earlier_join() -> None:
    """Guard: all joins share PyArrowTable, so each cross join must reach every earlier one."""
    pairs = _independent_pairs(6, "mixed")
    planned = pairs.planned
    planned.plan.create_execution_plan(planned.queue, planned.graph, planned.link_trekker, {})
    plan = planned.plan
    joins = _join_steps(plan)
    assert joins == list(plan.joinstep_collection.collection)
    steps_by_uuid = {step.uuid: step for step in plan}
    producer_of = {token: step.uuid for step in plan for token in step.get_uuids()}

    for index, join in enumerate(joins):
        if join.destination_framework == join.source_framework:
            continue
        for earlier in joins[:index]:
            assert ExecutionPlan._step_waits_for(join.uuid, earlier.uuid, steps_by_uuid, producer_of)


@pytest.mark.parametrize("shape", PAIR_SHAPES, ids=list(PAIR_SHAPES))
def test_planning_work_grows_linearly_with_independent_joins(shape: str) -> None:
    warm = _independent_pairs(2, shape).planned
    # The first plan in a process pays a one-time plugin import, which must not be counted.
    warm.plan.create_execution_plan(warm.queue, warm.graph, warm.link_trekker, {})

    calls_at_base, _ = _planning_calls(BASE_JOINS, shape)
    calls_at_double, pairs = _planning_calls(2 * BASE_JOINS, shape)

    ratio = calls_at_double / calls_at_base
    assert calls_at_double < MAX_CALL_RATIO_FOR_DOUBLED_JOINS * calls_at_base, (
        f"calls at {BASE_JOINS} joins: {calls_at_base}, at {2 * BASE_JOINS}: {calls_at_double}, ratio {ratio:.2f}, "
        f"bound {MAX_CALL_RATIO_FOR_DOUBLED_JOINS} (linear is 2, quadratic is 4)"
    )
    planned = pairs.planned
    join_steps = _join_steps(planned.plan)
    assert sorted(str(step.link.uuid) for step in join_steps) == sorted(str(link.uuid) for link in pairs.links)
    for link in pairs.links:
        assert len(_records(planned.plan, link)) == 1
    assert all(len(step.required_uuids) <= 4 for step in join_steps)


def _fg_step(plan: ExecutionPlan, uuid: UUID) -> FeatureGroupStep:
    matching = [step for step in plan if isinstance(step, FeatureGroupStep) and step.get_uuids() == {uuid}]
    assert len(matching) == 1, f"expected one step for {uuid}; got {matching}"
    return matching[0]


@pytest.mark.parametrize("shape", PAIR_SHAPES, ids=list(PAIR_SHAPES))
def test_independent_pairs_plan_the_same_structure_per_pair(shape: str) -> None:
    """Pins the plan shape the planner rewrite must keep, expressed relative to each pair."""
    pairs = _independent_pairs(4, shape)
    planned = pairs.planned
    planned.plan.create_execution_plan(planned.queue, planned.graph, planned.link_trekker, {})
    plan = planned.plan
    cross = shape == "cross_framework"

    join_steps = _join_steps(plan)
    join_of = {link.uuid: [step for step in join_steps if step.link.uuid == link.uuid] for link in pairs.links}
    assert all(len(steps) == 1 for steps in join_of.values())
    assert len(join_steps) == len(pairs.links)
    all_hops = [step for step in plan if isinstance(step, TransformFrameworkStep)]
    assert len(all_hops) == (len(pairs.links) if cross else 0)

    for index, (link, sides) in enumerate(zip(pairs.links, pairs.sides)):
        join = join_of[link.uuid][0]
        assert join.destination_framework_uuids == {sides.left_uuid}
        assert join.source_framework_uuids == {sides.right_uuid}

        hops = _transform_steps(plan, link)
        assert len(hops) == (1 if cross else 0)
        # Each join waits on its own sides and hop, and when the pairs cross frameworks on the join of
        # the previous pair only; same framework joins wait on nothing else.
        expected_required = {sides.left_uuid, sides.right_uuid} | {hop.uuid for hop in hops}
        if cross and index > 0:
            expected_required.add(join_of[pairs.links[index - 1].uuid][0].uuid)
        assert join.required_uuids == expected_required
        for hop in hops:
            assert hop.link_id == join.uuid
            assert hop.required_uuids == {sides.left_uuid, sides.right_uuid}
            assert hop.owed_tokens == {sides.child_uuid}

        consumer = _fg_step(plan, sides.child_uuid)
        # A same framework consumer is anchored on the destination parent, which also is its only tfs id.
        assert consumer.features.any_uuid == (sides.child_uuid if cross else sides.left_uuid)
        assert consumer.tfs_ids == (set() if cross else {sides.left_uuid})
        assert consumer.children_if_root == {sides.child_uuid}
        assert consumer.need_to_upload is False
        assert consumer.required_uuids == {join.uuid, sides.left_uuid, sides.right_uuid}
        assert set(consumer.features.link_index_columns) == set()

        pair_columns = {*link.left_index.index, *link.right_index.index}
        left_step = _fg_step(plan, sides.left_uuid)
        right_step = _fg_step(plan, sides.right_uuid)
        assert left_step.children_if_root == {sides.left_uuid, sides.child_uuid}
        assert right_step.children_if_root == {sides.right_uuid, sides.child_uuid} | (set() if cross else {join.uuid})
        assert left_step.need_to_upload is False
        assert right_step.need_to_upload is True
        for parent in (left_step, right_step):
            assert parent.features.any_uuid == (sides.left_uuid if parent is left_step else sides.right_uuid)
            assert parent.tfs_ids == set()
            assert parent.required_uuids == set()
            assert set(parent.features.link_index_columns) == pair_columns
