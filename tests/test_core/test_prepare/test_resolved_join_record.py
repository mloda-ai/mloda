"""One record per join decision, built next to the join steps and signing the same joins they do."""

import pickle  # nosec B403
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, Callable, Iterable, NamedTuple
from uuid import UUID

import pytest

from mloda.core.core.engine import Engine
from mloda.core.core.step.join_step import JoinStep
from mloda.core.core.step.transform_frame_work_step import TransformFrameworkStep
from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.core.prepare.graph.graph import Graph
from mloda.core.prepare.graph.properties import NodeProperties
from mloda.core.prepare.resolve_compute_frameworks import ResolveComputeFrameworks
from mloda.core.prepare.resolve_links import LinkTrekker
from mloda.core.prepare.resolved_join import DeclinedOrientation, JoinSide, JoinSignature, ResolvedJoin
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
from tests.test_core.test_prepare.join_plan_helpers import feature, trek


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


class Unlinked(NamedTuple):
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


def _pair_with_declined_orientation() -> Built:
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

    return _finish(planned, link, Sides(left.uuid, right.uuid, kept.uuid))


def _link_with_an_unlinked_third_parent() -> Unlinked:
    """A right join whose child also has a parent the link never mentions."""
    planned = _planned()
    link = _pair_link(Link.right)

    left = feature("resolved_join_unlinked_left", PyArrowTable, link.left_index)
    right = feature("resolved_join_unlinked_right", PandasDataFrame, link.right_index)
    unlinked = feature("resolved_join_unlinked_third", PandasDataFrame)
    child = feature("resolved_join_unlinked_child", PandasDataFrame)

    _add_parents(planned, link, left, right)
    planned.graph.add_node(unlinked.uuid, NodeProperties(unlinked, ResolvedJoinUnlinked))
    planned.queue.append((ResolvedJoinUnlinked, {unlinked}))
    planned.queue.append((link, PyArrowTable, PandasDataFrame))
    _add_child(planned, child, left, right, unlinked)
    trek(planned.link_trekker, link, (PyArrowTable, PandasDataFrame), child.uuid)

    declared: DeclaredFrameworks = {
        left.uuid: frozenset({PyArrowTable}),
        right.uuid: frozenset({PandasDataFrame}),
        unlinked.uuid: frozenset({PandasDataFrame, PythonDictFramework}),
    }
    planned.plan.create_execution_plan(planned.queue, planned.graph, planned.link_trekker, declared)
    return Unlinked(planned.plan, link, left.uuid, right.uuid, unlinked.uuid)


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

    # far_left (PyArrowTable) is the sole source-side uuid; nearest_left also sits on the
    # destination framework, so it lands on the destination side alongside right.uuid.
    return _finish(planned, link, Sides(far_left.uuid, right.uuid, child.uuid, nearest_left.uuid))


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

    left_pandas = feature("resolved_join_right_only_pyarrow_left_pandas", PandasDataFrame, link.left_index)
    left_pyarrow = feature("resolved_join_right_only_pyarrow_left_pyarrow", PyArrowTable, link.left_index)
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


def _join_steps(plan: ExecutionPlan) -> list[JoinStep]:
    return [step for step in plan if isinstance(step, JoinStep)]


def _transform_steps(plan: ExecutionPlan, link: Link) -> list[TransformFrameworkStep]:
    return [step for step in plan if isinstance(step, TransformFrameworkStep) and step.link_id == link.uuid]


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
    # destination_framework_uuids is a framework filter, not a side filter: the declared-left
    # parent that also sits on the destination framework lands here alongside the declared right.
    assert record.right.uuids == {built.sides.right_uuid, built.sides.extra_right_uuid}
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


def test_a_right_joins_destination_stays_right_when_declared_right_is_the_only_pyarrow_exclusive_side() -> None:
    """Regression pin for issue #1137 finding #2 ('hop names PairRight -> PairLeft'): declared right can
    only ever be PyArrow, so the destination must land there even though declared left also offers PyArrow."""
    built = _right_join_declared_left_spans_frameworks_declared_right_is_pyarrow_only()

    record = _one_record(built.plan, built.link)

    assert record.jointype is JoinType.RIGHT
    assert record.destination_side is JoinSide.RIGHT


def test_a_parent_the_link_never_mentions_stays_out_of_the_declared_sides() -> None:
    unlinked = _link_with_an_unlinked_third_parent()

    record = _one_record(unlinked.plan, unlinked.link)

    assert record.destination_side is JoinSide.RIGHT
    assert record.left.uuids == {unlinked.left_uuid}
    assert record.right.uuids == {unlinked.right_uuid}
    # The join writes into a parent that belongs to neither declared side, which a validation step should reject.
    assert record.destination_uuids == {unlinked.right_uuid, unlinked.unlinked_uuid}
    assert record.source_uuids == {unlinked.left_uuid}


def test_a_declared_side_keeps_only_the_frameworks_its_own_parents_declared() -> None:
    unlinked = _link_with_an_unlinked_third_parent()

    record = _one_record(unlinked.plan, unlinked.link)

    assert record.left.declared_frameworks == {PyArrowTable}
    assert record.right.declared_frameworks == {PandasDataFrame}
    assert PythonDictFramework not in record.left.declared_frameworks | record.right.declared_frameworks


def test_a_third_parent_sharing_the_destination_framework_must_not_leak_into_declared_left() -> None:
    """Issue #1137 finding #1: the declared left split across two frameworks defeats the full (both-sides)
    containment check, and the fallback must not hand an unrelated, same-framework parent to record.left,
    nor the declared-left parent that belongs to the other framework's joinstep."""
    built = _link_with_a_declared_left_split_across_frameworks_and_a_colliding_third_parent()

    record = _one_record(built.plan, built.link)

    assert built.unrelated_uuid not in record.left.uuids
    assert built.left_pyarrow_uuid not in record.left.uuids
    assert record.left.uuids == {built.left_pandas_uuid}
    assert record.right.uuids == {built.right_uuid}


def test_a_third_parent_sharing_the_destination_framework_must_not_leak_its_frameworks_into_declared_left() -> None:
    built = _link_with_a_declared_left_split_across_frameworks_and_a_colliding_third_parent()

    record = _one_record(built.plan, built.link)

    assert PandasDataFrame in record.left.declared_frameworks
    assert PythonDictFramework not in record.left.declared_frameworks


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

    planned.plan.create_execution_plan(planned.queue, planned.graph, planned.link_trekker)

    resolved = planned.plan.resolved_join_plan
    assert len(resolved.records) == 1, "the kept orientation must plan a record for the decline to say anything"
    assert resolved.declined == (DeclinedOrientation(link.uuid, PyArrowTable, PandasDataFrame),)


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
        _link_with_an_unlinked_third_parent,
        _case_override_inverted,
        _case_override_beats_nearer_wrong_framework_left,
        _case_override_disagrees_with_the_nearest_split,
        _right_join_farther_right_parent_holds_destination_framework,
        _right_join_both_sides_claim_destination_framework,
        _link_with_a_declared_left_split_across_frameworks_and_a_colliding_third_parent,
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
        "case_override_inverted",
        "case_override_beats_nearer_wrong_framework_left",
        "case_override_disagrees_with_nearest_split",
        "right_join_farther_right_parent_holds_destination_framework",
        "right_join_both_sides_claim_destination_framework",
        "declared_left_framework_collision_with_unrelated_third_parent",
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


def test_a_nearest_left_parent_on_a_third_framework_keeps_the_record_on_the_steps_side() -> None:
    """run_link ranks the left side over all required parents; the record must not re-rank over fewer of them."""
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

    planned.plan.create_execution_plan(planned.queue, planned.graph, planned.link_trekker)

    assert len(_join_steps(planned.plan)) == 1, "the shape must plan exactly one JoinStep for this to say anything"
    assert planned.plan.resolved_join_plan.signatures() == planned.plan.join_signatures_at_build
    record = _one_record(planned.plan, link)
    assert record.destination_side is JoinSide.RIGHT
    assert record.destination.uuids <= record.destination_uuids
    assert record.source.uuids <= record.source_uuids


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


def test_the_resolver_snapshots_the_frameworks_a_feature_declared_before_the_rewrite() -> None:
    link = _pair_link()
    left = feature("resolved_join_snapshot_left", PyArrowTable, link.left_index)
    right = feature("resolved_join_snapshot_right", PandasDataFrame, link.right_index)
    child = Feature("resolved_join_snapshot_child")
    child.compute_frameworks = {PyArrowTable, PandasDataFrame}

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

    assert child.compute_frameworks == {PyArrowTable}, "the rewrite has to collapse the child for this to say anything"
    assert resolver.get_declared_frameworks()[child.uuid] == {PyArrowTable, PandasDataFrame}
    assert resolver.get_declared_frameworks()[left.uuid] == {PyArrowTable}


# Fresh interpreters are slow to start, so this needs more than the suite-wide timeout.
@pytest.mark.timeout(60)
def test_fresh_interpreters_build_the_same_record_signature() -> None:
    outputs = run_probes(_PROBE, _PROBE_PROCESSES)

    assert len(outputs) == _PROBE_PROCESSES, f"expected {_PROBE_PROCESSES} probe results, got {len(outputs)}"
    for position, output in enumerate(outputs):
        assert output == _PROBE_EXPECTED, f"probe {position} signed {output}, expected {_PROBE_EXPECTED}"
