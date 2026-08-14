"""Materializes the join decision run_link made as records, and signs the legacy join steps the same way.

Shadow mode: the two signature sets must agree and divergence raises; nothing here changes what the plan runs.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from uuid import UUID, uuid4

from mloda.core.abstract_plugins.components.error_utils import internal_invariant_error
from mloda.core.abstract_plugins.components.index.index import Index
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.core.step.join_step import JoinStep
from mloda.core.prepare.resolve_links import LinkFrameworkTrekker
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


DeclaredFrameworks = Mapping[UUID, frozenset[type[ComputeFramework]]]


def _side(
    feature_group: type[FeatureGroup],
    index: Index,
    uuids: frozenset[UUID],
    declared_frameworks: DeclaredFrameworks,
) -> ResolvedJoinSide:
    # Unions the parents' declared candidates; a multi-parent side may claim a framework only one parent declared.
    frameworks: set[type[ComputeFramework]] = set()
    for uuid in uuids:
        frameworks |= declared_frameworks.get(uuid, frozenset())
    return ResolvedJoinSide(feature_group, index, uuids, frozenset(frameworks))


def build_resolved_join_plan(
    planned: Sequence[tuple[PlannedOrientation, JoinStep]],
    declined: Sequence[LinkFrameworkTrekker],
    declared_frameworks: DeclaredFrameworks,
) -> ResolvedJoinPlan:
    """One record per planned orientation, ordered as the orientations were planned."""
    records: list[ResolvedJoin] = []
    token_by_step: dict[UUID, UUID] = {}

    for orientation, join_step in planned:
        link = orientation.link
        side = orientation.destination_side
        destination = frozenset(join_step.destination_framework_uuids)
        source = frozenset(join_step.source_framework_uuids)
        resolved_left, resolved_right = (destination, source) if side is JoinSide.LEFT else (source, destination)
        if link.left_feature_group == link.right_feature_group:
            # run_link resolves a self link's left-discriminated parents into the destination set unconditionally.
            left_uuids, right_uuids = destination, source
        elif (
            orientation.left_uuids <= resolved_left
            and orientation.right_uuids <= resolved_right
            and orientation.left_uuids != orientation.right_uuids
        ):
            left_uuids, right_uuids = orientation.left_uuids, orientation.right_uuids
        else:
            # The step's sets, so a record can never name parents outside its own destination/source claim.
            left_uuids, right_uuids = resolved_left, resolved_right

        record = ResolvedJoin(
            link_uuid=link.uuid,
            jointype=link.jointype,
            left=_side(link.left_feature_group, link.left_index, left_uuids, declared_frameworks),
            right=_side(link.right_feature_group, link.right_index, right_uuids, declared_frameworks),
            destination_side=side,
            destination_uuids=destination,
            source_uuids=source,
            destination_framework=join_step.destination_framework,
            source_framework=join_step.source_framework,
            consumers=orientation.consumers,
            depends_on=frozenset(),
            token=uuid4(),
            shadowed_step_uuid=join_step.uuid,
        )
        records.append(record)
        token_by_step[join_step.uuid] = record.token

    # An order edge is keyed by link uuid, so a producer fans its tokens over every record it built.
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
    steps = list(join_steps)
    link_of_step = {step.uuid: step.link.uuid for step in steps}
    return frozenset(_legacy_signature(step, link_of_step) for step in steps)


def raise_on_join_plan_divergence(plan: ResolvedJoinPlan, join_steps: Iterable[JoinStep]) -> object:
    """Returns None when the records and the steps sign the same joins; a divergence is a planning bug."""
    recorded = plan.signatures()
    legacy = legacy_join_signatures(join_steps)
    if recorded == legacy:
        return None
    raise ValueError(
        internal_invariant_error(
            "the resolved join records and the planned join steps sign different joins.",
            f"only_in_records={sorted(recorded - legacy)}, only_in_steps={sorted(legacy - recorded)}",
        )
    )
