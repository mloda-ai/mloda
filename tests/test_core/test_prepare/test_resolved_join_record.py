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
from mloda.core.prepare.resolved_join_builder import build_resolved_join_plan, legacy_join_signatures
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
# Each side reduces from a framework set, so a second cold interpreter is the whole cross-process signal.
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
    """Two children of one link, and only the PyArrow one pairs a left side up with a right side."""
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


def test_each_record_names_the_join_step_it_shadows() -> None:
    built = _two_links()
    step_of_link = {step.link.uuid: step.uuid for step in _join_steps(built.plan)}

    records = built.plan.resolved_join_plan.records

    assert len(records) == len(step_of_link)
    for record in records:
        assert record.shadowed_step_uuid == step_of_link[record.link_uuid]
        assert record.shadowed_step_uuid != record.token


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
    ],
)
def test_the_records_sign_the_joins_the_legacy_join_steps_sign(build: Callable[[], Any]) -> None:
    built = build()

    join_steps = _join_steps(built.plan)
    resolved = built.plan.resolved_join_plan

    assert join_steps, "the shape must plan at least one JoinStep for the parity to say anything"
    assert len(resolved.records) == len(join_steps)
    assert resolved.signatures() == built.plan.join_signatures_at_build


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
    assert _one_record(planned.plan, link).destination_side is JoinSide.RIGHT


def test_flipping_the_merge_sides_of_a_join_step_breaks_the_parity() -> None:
    """The signatures only agree while the record derives its destination side from the planned orientation."""
    built = _declared_pair()
    join_step = _join_steps(built.plan)[0]

    join_step.swap_merge_sides = not join_step.swap_merge_sides
    rebuilt = build_resolved_join_plan(
        built.plan.planned_orientations, built.plan.declined_orientations, built.declared_frameworks
    )

    assert len(rebuilt.records) == len(_join_steps(built.plan)), "an empty rebuild makes the inequality meaningless"
    assert rebuilt.signatures(), "an empty rebuild makes the inequality meaningless"
    assert rebuilt.signatures() != legacy_join_signatures([join_step])


def test_the_record_leaves_out_the_write_serialization_edges_add_tfs_adds() -> None:
    built = _two_links()

    recorded = built.plan.resolved_join_plan.signatures()
    after_tfs = legacy_join_signatures(_join_steps(built.plan))

    assert after_tfs != recorded, "add_tfs must add an edge here for this to say anything"
    assert _without_depends(after_tfs) == _without_depends(recorded)


def test_a_second_planning_pass_does_not_accumulate_records() -> None:
    planned = _planned()
    link = _pair_link()
    _branch(planned, link, "resolved_join_twice")
    planned.plan.create_execution_plan(planned.queue, planned.graph, planned.link_trekker)
    first = len(planned.plan.resolved_join_plan.records)

    planned.plan.create_execution_plan(planned.queue, planned.graph, planned.link_trekker)

    assert len(planned.plan.resolved_join_plan.records) == first
    assert len(planned.plan.resolved_join_plan.records) == len(_join_steps(planned.plan))


def test_an_order_edge_makes_the_consumer_records_depend_on_the_producer_record_tokens() -> None:
    chain = _ordered_chain()
    resolved = chain.plan.resolved_join_plan
    producer_records = resolved.records_of_link(chain.producer.uuid)
    consumer_records = resolved.records_of_link(chain.consumer.uuid)
    step_uuids = {step.uuid for step in _join_steps(chain.plan)}

    assert producer_records, "the producer link must build a record for the edge to point at"
    assert consumer_records, "the consumer link must build a record for the edge to hang off"
    for record in producer_records:
        assert not record.depends_on
    for record in consumer_records:
        assert record.depends_on == {produced.token for produced in producer_records}
        assert chain.producer.uuid not in record.depends_on, "a record depends on tokens, not on link uuids"
        assert record.depends_on.isdisjoint(step_uuids), "a record depends on tokens, not on step uuids"
        assert record.signature(resolved.link_of_token()).depends_on_links == (str(chain.producer.uuid),)


def test_the_record_and_the_legacy_transform_hop_name_opposite_directions() -> None:
    built = _inverted_pair()

    record = _one_record(built.plan, built.link)
    transform_steps = _transform_steps(built.plan, built.link)

    assert len(transform_steps) == 1
    # The record binds the hop to the sides the join actually moves; a later lowering step should adopt its answer.
    assert record.transform_from_feature_group is ResolvedJoinPairLeft
    assert record.transform_to_feature_group is ResolvedJoinPairRight
    assert transform_steps[0].from_feature_group is ResolvedJoinPairRight
    assert transform_steps[0].to_feature_group is ResolvedJoinPairLeft


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


# Fresh interpreters are slow to start, so this one needs more than the suite-wide per-test budget.
@pytest.mark.timeout(60)
def test_fresh_interpreters_build_the_same_record_signature() -> None:
    outputs = run_probes(_PROBE, _PROBE_PROCESSES)

    assert len(outputs) == _PROBE_PROCESSES, f"expected {_PROBE_PROCESSES} probe results, got {len(outputs)}"
    for position, output in enumerate(outputs):
        assert output == _PROBE_EXPECTED, f"probe {position} signed {output}, expected {_PROBE_EXPECTED}"
