"""Builds the resolved join plan from the orientations run_link planned, and signs the legacy join steps.

The two signature sets are compared in shadow mode only; nothing here changes what the plan runs.
"""

import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import NamedTuple
from uuid import UUID, uuid4

from mloda.core.abstract_plugins.components.index.index import Index
from mloda.core.abstract_plugins.components.link import Link
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.core.step.join_step import JoinStep
from mloda.core.prepare.graph.graph import Graph
from mloda.core.prepare.resolve_links import LinkFrameworkTrekker, inheritance_distance
from mloda.core.prepare.resolved_join import (
    DeclinedOrientation,
    JoinSide,
    JoinSignature,
    PlannedOrientation,
    ResolvedJoin,
    ResolvedJoinPlan,
    ResolvedJoinSide,
    signed_uuids,
)


logger = logging.getLogger(__name__)

DeclaredFrameworks = Mapping[UUID, frozenset[type[ComputeFramework]]]


class _DeclaredSides(NamedTuple):
    """The side the destination lands in, plus the parents each declared side owns."""

    destination_side: JoinSide
    left_uuids: frozenset[UUID]
    right_uuids: frozenset[UUID]


def _nearest(uuids_by_distance: dict[int, set[UUID]]) -> frozenset[UUID]:
    """A declared side is held by its closest subclasses only; farther ones answer for a different side."""
    if not uuids_by_distance:
        return frozenset()
    return frozenset(uuids_by_distance[min(uuids_by_distance)])


def _nearest_side_uuids(link: Link, uuids: set[UUID], graph: Graph) -> tuple[frozenset[UUID], frozenset[UUID]]:
    """Split the join's parents into the nearest subclasses of each declared feature group."""
    left_by_distance: dict[int, set[UUID]] = defaultdict(set)
    right_by_distance: dict[int, set[UUID]] = defaultdict(set)
    nodes = graph.get_nodes()

    for uuid in uuids:
        if uuid not in nodes:
            continue
        feature_group_class = nodes[uuid].feature_group_class
        if issubclass(feature_group_class, link.left_feature_group):
            left_by_distance[inheritance_distance(feature_group_class, link.left_feature_group)].add(uuid)
        if issubclass(feature_group_class, link.right_feature_group):
            right_by_distance[inheritance_distance(feature_group_class, link.right_feature_group)].add(uuid)

    return _nearest(left_by_distance), _nearest(right_by_distance)


def _destination_side(link: Link, join_step: JoinStep, graph: Graph) -> _DeclaredSides:
    """Name the declared side that holds the destination, next to the parents each side owns."""
    left_uuids, right_uuids = _nearest_side_uuids(
        link, join_step.destination_framework_uuids | join_step.source_framework_uuids, graph
    )

    destination = join_step.destination_framework_uuids
    holds_left = bool(destination & left_uuids)
    holds_right = bool(destination & right_uuids)

    if holds_left and not holds_right:
        side = JoinSide.LEFT
    elif holds_right and not holds_left:
        side = JoinSide.RIGHT
    else:
        # Self links and sides sharing one feature group are not decidable from the declared groups.
        side = JoinSide.RIGHT if join_step.swap_merge_sides else JoinSide.LEFT

    resolved_left, resolved_right = (
        (destination, join_step.source_framework_uuids)
        if side is JoinSide.LEFT
        else (join_step.source_framework_uuids, destination)
    )
    if left_uuids != right_uuids and left_uuids <= resolved_left and right_uuids <= resolved_right:
        return _DeclaredSides(side, left_uuids, right_uuids)
    # One feature group on both sides is not separable by class, so the resolved sets separate the parents.
    return _DeclaredSides(side, frozenset(resolved_left), frozenset(resolved_right))


def _side(
    feature_group: type[FeatureGroup],
    index: Index,
    uuids: frozenset[UUID],
    declared_frameworks: DeclaredFrameworks,
) -> ResolvedJoinSide:
    frameworks: set[type[ComputeFramework]] = set()
    for uuid in uuids:
        frameworks |= declared_frameworks.get(uuid, frozenset())
    return ResolvedJoinSide(feature_group, index, uuids, frozenset(frameworks))


def build_resolved_join_plan(
    planned: Sequence[tuple[PlannedOrientation, JoinStep]],
    declined: Sequence[LinkFrameworkTrekker],
    graph: Graph,
    declared_frameworks: DeclaredFrameworks,
) -> ResolvedJoinPlan:
    """One record per planned orientation, ordered as the orientations were planned."""
    records: list[ResolvedJoin] = []
    token_by_step: dict[UUID, UUID] = {}

    for orientation, join_step in planned:
        link = orientation.key[0]
        sides = _destination_side(link, join_step, graph)

        record = ResolvedJoin(
            link_uuid=link.uuid,
            jointype=link.jointype,
            left=_side(link.left_feature_group, link.left_index, sides.left_uuids, declared_frameworks),
            right=_side(link.right_feature_group, link.right_index, sides.right_uuids, declared_frameworks),
            destination_side=sides.destination_side,
            destination_uuids=frozenset(join_step.destination_framework_uuids),
            source_uuids=frozenset(join_step.source_framework_uuids),
            destination_framework=join_step.destination_framework,
            source_framework=join_step.source_framework,
            consumers=orientation.consumers,
            depends_on=frozenset(),
            token=uuid4(),
            shadowed_step_uuid=join_step.uuid,
        )
        records.append(record)
        token_by_step[join_step.uuid] = record.token

    # An order edge is keyed by link uuid, so a producer link fans its tokens out over every record it built.
    resolved = tuple(
        replace(
            record,
            depends_on=frozenset(token_by_step[uuid] for uuid in step.required_uuids if uuid in token_by_step),
        )
        for record, (_, step) in zip(records, planned)
    )
    return ResolvedJoinPlan(resolved, tuple(DeclinedOrientation(key[0].uuid, key[1], key[2]) for key in declined))


def _legacy_signature(join_step: JoinStep, link_of_step: Mapping[UUID, UUID]) -> JoinSignature:
    """The join a legacy step names, with its destination side read off swap_merge_sides."""
    side = JoinSide.RIGHT if join_step.swap_merge_sides else JoinSide.LEFT
    return JoinSignature(
        link_uuid=join_step.link.uuid,
        jointype=join_step.link.jointype.value,
        destination_uuids=signed_uuids(join_step.destination_framework_uuids),
        source_uuids=signed_uuids(join_step.source_framework_uuids),
        destination_side=side.value,
        destination_framework=join_step.destination_framework.get_class_name(),
        source_framework=join_step.source_framework.get_class_name(),
        depends_on_links=tuple(
            sorted({str(link_of_step[uuid]) for uuid in join_step.required_uuids if uuid in link_of_step})
        ),
    )


def legacy_join_signatures(join_steps: Iterable[JoinStep]) -> frozenset[JoinSignature]:
    """The same signature read off the legacy join steps."""
    steps = list(join_steps)
    link_of_step = {step.uuid: step.link.uuid for step in steps}
    return frozenset(_legacy_signature(step, link_of_step) for step in steps)


def log_join_plan_divergence(plan: ResolvedJoinPlan, join_steps: Iterable[JoinStep]) -> None:
    """Report at DEBUG where the records and the join steps sign different joins."""
    if not logger.isEnabledFor(logging.DEBUG):
        return

    steps = list(join_steps)
    link_of_step = {step.uuid: step.link.uuid for step in steps}
    legacy = {_legacy_signature(step, link_of_step): step.uuid for step in steps}

    link_of_token = plan.link_of_token()
    recorded = {record.signature(link_of_token): record.shadowed_step_uuid for record in plan.records}

    if recorded.keys() == legacy.keys():
        return

    logger.debug(
        "The resolved join plan diverges from the join steps. Only in the records: %s. Only in the steps: %s.",
        sorted(
            f"{signature} (step {step_uuid})" for signature, step_uuid in recorded.items() if signature not in legacy
        ),
        sorted(
            f"{signature} (step {step_uuid})" for signature, step_uuid in legacy.items() if signature not in recorded
        ),
    )
