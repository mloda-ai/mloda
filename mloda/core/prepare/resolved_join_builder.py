"""Materializes the join decision run_link made as records, and signs the legacy join steps the same way.

Shadow mode: the two signature sets must agree and divergence raises; nothing here changes what the plan runs.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from uuid import UUID

from mloda.core.abstract_plugins.components.error_utils import internal_invariant_error
from mloda.core.abstract_plugins.components.index.index import Index
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.core.step.join_step import JoinStep
from mloda.core.prepare.resolved_join import (
    JoinSide,
    JoinSignature,
    ResolvedJoin,
    ResolvedJoinPlan,
    ResolvedJoinSide,
    signed_uuids,
)


DeclaredFrameworks = Mapping[UUID, frozenset[type[ComputeFramework]]]


def build_resolved_join_side(
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


def wire_join_dependencies(records: Sequence[ResolvedJoin], join_steps: Iterable[JoinStep]) -> tuple[ResolvedJoin, ...]:
    """Fill in each record's depends_on: which other records' tokens its own JoinStep waits for."""
    steps_by_uuid = {step.uuid: step for step in join_steps}
    all_tokens = {record.token for record in records}
    resolved = []
    for record in records:
        join_step = steps_by_uuid[record.token]
        depends_on = frozenset(uuid for uuid in join_step.required_uuids if uuid in all_tokens and uuid != record.token)
        resolved.append(replace(record, depends_on=depends_on))
    return tuple(resolved)


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
