"""Guards a resolved join plan against the confirmed silent-data-loss shape: two joins that both
read one parent's data as their source, with neither writing its result back into that parent's
slot, and both feeding a consumer that needs them together.
"""

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
    """The class of the contested parent, read off whichever record declares it on a side."""
    for record in records:
        name = _side_feature_group_name(record, uuid)
        if name is not None:
            return name
    return str(uuid)


def _competing_join_label(record: ResolvedJoin, shared_uuid: UUID) -> str:
    """The record's declared side opposite the shared parent, or its link uuid if neither side matches."""
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
        f"{shared_name}'s data is read as the join source by two joins ({first_label} and {second_label}), and "
        f"neither writes its result back into {shared_name}'s slot: the two branches can never reunite, so a "
        "consumer needing both loses whichever the runtime does not happen to expose to it.\n"
        f"Resolution: chain the joins so one of them writes its result back into {shared_name}'s slot, instead "
        "of both draining it independently."
    )


def raise_on_orphaned_join_source(plan: ResolvedJoinPlan) -> None:
    """Raise when two joins share a source parent that no join writes back into and share a consumer:
    the branches can never reunite, so whichever the runtime does not expose is silently dropped."""
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
