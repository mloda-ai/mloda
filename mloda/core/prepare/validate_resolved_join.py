"""Guards a resolved join plan against two joins draining a shared parent that no join writes back into."""

from collections import defaultdict
from itertools import combinations
from uuid import UUID

from mloda.core.prepare.resolved_join import ResolvedJoin, ResolvedJoinPlan


def _side_feature_group_name(record: ResolvedJoin, uuid: UUID) -> str | None:
    if uuid in record.left.uuids:
        return record.left.feature_group.get_class_name()
    if uuid in record.right.uuids:
        return record.right.feature_group.get_class_name()
    return None


def _shared_feature_group_name(records: tuple[ResolvedJoin, ...], uuid: UUID) -> str:
    """The contested parent's class, read off whichever record declares it on a side."""
    for record in records:
        name = _side_feature_group_name(record, uuid)
        if name is not None:
            return name
    return str(uuid)


def _competing_join_label(record: ResolvedJoin, shared_uuid: UUID) -> str:
    """The declared side opposite the shared parent, falling back to the link uuid."""
    if shared_uuid in record.left.uuids:
        return record.right.feature_group.get_class_name()
    if shared_uuid in record.right.uuids:
        return record.left.feature_group.get_class_name()
    return str(record.link_uuid)


def _orphaned_join_source_error(
    shared_uuid: UUID, records: tuple[ResolvedJoin, ...], first: ResolvedJoin, second: ResolvedJoin
) -> str:
    shared_name = _shared_feature_group_name(records, shared_uuid)
    first_label = _competing_join_label(first, shared_uuid)
    second_label = _competing_join_label(second, shared_uuid)
    return (
        f"{shared_name} is read as the join source by two joins ({first_label} and {second_label}), and neither "
        f"writes its result back into {shared_name}: the two branches can never reunite, so a consumer needing "
        "both loses one of them.\n"
        f"Resolution: chain the joins so that one of them writes its result back into {shared_name}."
    )


def raise_on_orphaned_join_source(plan: ResolvedJoinPlan) -> None:
    """Raise when two joins share both a consumer and a source parent that no join writes back into."""
    all_destination_uuids: set[UUID] = set()
    for record in plan.records:
        all_destination_uuids |= record.destination_uuids

    readers_of: dict[UUID, list[ResolvedJoin]] = defaultdict(list)
    for record in plan.records:
        for uuid in record.source_uuids:
            readers_of[uuid].append(record)

    for uuid, readers in readers_of.items():
        if len(readers) < 2 or uuid in all_destination_uuids:
            continue
        for first, second in combinations(readers, 2):
            if first.consumers & second.consumers:
                raise ValueError(_orphaned_join_source_error(uuid, plan.records, first, second))
