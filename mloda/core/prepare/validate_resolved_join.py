"""Guards a resolved join plan against orphaned shared sources and against records that
disagree with the plan (foreign side members, overlapping destination and source)."""

from collections import defaultdict
from collections.abc import Iterable
from itertools import combinations
from uuid import UUID

from mloda.core.abstract_plugins.components.error_utils import internal_invariant_error
from mloda.core.abstract_plugins.components.link import JoinType
from mloda.core.core.step.join_step import JoinStep
from mloda.core.prepare.graph.graph import Graph
from mloda.core.prepare.resolved_join import ResolvedJoin, ResolvedJoinPlan, ResolvedJoinSide


def _stays_in_source_framework(record: ResolvedJoin) -> bool:
    """Same-framework joins reunite through add_tfs's children_if_root bookkeeping, not uuid-slot rewriting."""
    return record.destination_framework is record.source_framework


def _shared_consumer_writes_back(
    records: tuple[ResolvedJoin, ...], uuid: UUID, shared_consumers: frozenset[UUID]
) -> bool:
    """True iff some record writes uuid back for a consumer the competing pair actually shares."""
    return any(uuid in record.destination_uuids and record.consumers & shared_consumers for record in records)


def _orphaned_join_source_error(first: ResolvedJoin, second: ResolvedJoin) -> str:
    """Both records read the shared uuid as their source, so `.source`/`.destination` name it directly."""
    shared_name = first.source.feature_group.get_class_name()
    first_label = first.destination.feature_group.get_class_name()
    second_label = second.destination.feature_group.get_class_name()
    return (
        f"{shared_name} is read as the join source by two joins ({first_label} and {second_label}), and neither "
        f"writes its result back into {shared_name}: the two branches can never reunite, so a consumer needing "
        "both loses one of them.\n"
        f"Resolution: chain the joins so that one of them writes its result back into {shared_name}."
    )


def raise_on_orphaned_join_source(plan: ResolvedJoinPlan) -> None:
    """Raise when two joins share both a consumer and a source parent that no join writes back into for it."""
    readers_of: dict[UUID, list[ResolvedJoin]] = defaultdict(list)
    for record in plan.records:
        for uuid in record.source_uuids:
            readers_of[uuid].append(record)

    for uuid, readers in readers_of.items():
        if len(readers) < 2:
            continue
        for first, second in combinations(readers, 2):
            shared_consumers = first.consumers & second.consumers
            if not shared_consumers:
                continue
            if _stays_in_source_framework(first) and _stays_in_source_framework(second):
                continue
            if _shared_consumer_writes_back(plan.records, uuid, shared_consumers):
                continue
            raise ValueError(_orphaned_join_source_error(first, second))


def _foreign_members(side: ResolvedJoinSide, graph: Graph) -> list[str]:
    nodes = graph.get_nodes()
    return [
        nodes[uuid].feature_group_class.get_class_name()
        for uuid in side.uuids
        if not issubclass(nodes[uuid].feature_group_class, side.feature_group)
    ]


def raise_on_dishonest_join_record(plan: ResolvedJoinPlan, join_steps: Iterable[JoinStep], graph: Graph) -> None:
    """Raise when a record's sides hold parents outside their declared group or its two ends overlap."""
    step_of = {step.uuid: step for step in join_steps}
    for record in plan.records:
        for label, side in (("left", record.left), ("right", record.right)):
            foreign = _foreign_members(side, graph)
            if foreign:
                raise ValueError(
                    internal_invariant_error(
                        f"Every {label} member of link {record.link_uuid} is a {side.feature_group.get_class_name()}.",
                        f"{label} side holds members of {sorted(foreign)}.",
                    )
                )
        if record.jointype in (JoinType.APPEND, JoinType.UNION):
            continue
        step = step_of.get(record.token)
        if record.destination_framework is record.source_framework and not (step and step.carriers):
            continue
        overlap = record.destination_uuids & record.source_uuids
        if overlap:
            raise ValueError(
                internal_invariant_error(
                    f"Join destination and source of link {record.link_uuid} are disjoint.",
                    f"Both name {sorted(str(uuid) for uuid in overlap)}.",
                )
            )
