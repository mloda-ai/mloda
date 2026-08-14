"""Guards a resolved join plan against two joins draining a shared parent that no join writes
back into for the consumer(s) the two joins actually share."""

from collections import defaultdict
from itertools import combinations
from uuid import UUID

from mloda.core.prepare.resolved_join import ResolvedJoin, ResolvedJoinPlan


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
