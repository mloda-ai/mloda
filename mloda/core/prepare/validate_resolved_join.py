"""Guards a resolved join plan against orphaned shared sources and against records that
disagree with the plan (foreign side members, overlapping destination and source)."""

from collections import defaultdict
from collections.abc import Iterable
from itertools import combinations
from uuid import UUID

from mloda.core.abstract_plugins.components.error_utils import internal_invariant_error
from mloda.core.abstract_plugins.components.link import JoinType
from mloda.core.abstract_plugins.components.utils import is_nominal_subclass
from mloda.core.core.step.join_step import JoinStep
from mloda.core.prepare.graph.graph import Graph
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.declared_sides import (
    star_keeping_two_groups_reason,
    star_missed_join_reason,
    star_mixed_joins_reason,
)
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


def _kept_and_dropped(record: ResolvedJoin) -> tuple[ResolvedJoinSide, ResolvedJoinSide] | None:
    """The declared (kept, dropped) sides of a left or right join, else None."""
    if record.jointype in (JoinType.LEFT, JoinType.ASOF):
        return record.left, record.right
    if record.jointype is JoinType.RIGHT:
        return record.right, record.left
    return None


def _class_names(uuids: Iterable[UUID], graph: Graph) -> str:
    nodes = graph.get_nodes()
    return " and ".join(sorted({nodes[uuid].feature_group_class.get_class_name() for uuid in uuids}))


def _role(record: ResolvedJoin, side: ResolvedJoinSide) -> str | None:
    """How a record treats one of its sides: inner, outer, kept or dropped; None for append and union."""
    if record.jointype is JoinType.INNER:
        return "inner"
    if record.jointype is JoinType.OUTER:
        return "outer"
    kept_and_dropped = _kept_and_dropped(record)
    if kept_and_dropped is None:
        return None
    return "kept" if side is kept_and_dropped[0] else "dropped"


_ALWAYS_REJECTED = (frozenset({"dropped", "inner"}), frozenset({"dropped", "outer"}), frozenset({"inner", "outer"}))
_REJECTED_ON_SAME_INDEX = (frozenset({"dropped", "kept"}), frozenset({"kept", "outer"}))


def _coalesces_keys(record: ResolvedJoin) -> bool:
    """Same-named key columns merge into one column; differently named ones keep both."""
    return record.left.index.index == record.right.index.index


def _order_dependent(
    roles: frozenset[str],
    first: tuple[ResolvedJoin, ResolvedJoinSide, str],
    second: tuple[ResolvedJoin, ResolvedJoinSide, str],
) -> bool:
    if roles in _ALWAYS_REJECTED:
        return True
    if roles not in _REJECTED_ON_SAME_INDEX or first[1].index != second[1].index:
        return False
    return any(_coalesces_keys(record) for record, _, role in (first, second) if role in ("dropped", "outer"))


def _raise_on_two_dropped(first: ResolvedJoin, second: ResolvedJoin, graph: Graph) -> None:
    first_sides = _kept_and_dropped(first)
    second_sides = _kept_and_dropped(second)
    if first_sides is None or second_sides is None:
        return
    kept_first, dropped_first = first_sides
    kept_second, dropped_second = second_sides
    shared_dropped = dropped_first.uuids & dropped_second.uuids
    if not shared_dropped or kept_first.uuids & kept_second.uuids:
        return
    nodes = graph.get_nodes()
    if {nodes[u].feature_group_class for u in kept_first.uuids} == {
        nodes[u].feature_group_class for u in kept_second.uuids
    }:
        return
    consumer = sorted(nodes[uuid].feature_group_class.get_class_name() for uuid in first.consumers & second.consumers)[
        0
    ]
    kept_names = sorted((_class_names(kept_first.uuids, graph), _class_names(kept_second.uuids, graph)))
    message = star_keeping_two_groups_reason(
        consumer,
        kept_names[0],
        kept_names[1],
        _class_names(shared_dropped, graph),
        (first.jointype, second.jointype),
    )
    raise ValueError(f"{message}.")


def _other_side(record: ResolvedJoin, side: ResolvedJoinSide) -> ResolvedJoinSide:
    return record.right if side is record.left else record.left


def _is_self_link(record: ResolvedJoin) -> bool:
    return bool(record.left.uuids & record.right.uuids)


def raise_on_join_star_keeping_two_groups(plan: ResolvedJoinPlan, graph: Graph) -> None:
    """Raise when one consumer reads two joins whose rows depend on which of them runs first."""
    for first, second in combinations(plan.records, 2):
        shared_consumers = first.consumers & second.consumers
        if not shared_consumers or _is_self_link(first) or _is_self_link(second):
            continue
        _raise_on_two_dropped(first, second, graph)
        for first_side in (first.left, first.right):
            for second_side in (second.left, second.right):
                shared = first_side.uuids & second_side.uuids
                first_role = _role(first, first_side)
                second_role = _role(second, second_side)
                if not shared or first_role is None or second_role is None:
                    continue
                roles = frozenset({first_role, second_role})
                if _order_dependent(roles, (first, first_side, first_role), (second, second_side, second_role)):
                    nodes = graph.get_nodes()
                    consumer = sorted(nodes[uuid].feature_group_class.get_class_name() for uuid in shared_consumers)[0]
                    others = sorted(
                        (
                            _class_names(_other_side(first, first_side).uuids, graph),
                            _class_names(_other_side(second, second_side).uuids, graph),
                        )
                    )
                    message = star_mixed_joins_reason(
                        consumer,
                        _class_names(shared, graph),
                        others[0],
                        others[1],
                        (first_role, second_role) if first_role <= second_role else (second_role, first_role),
                        (first.jointype, second.jointype),
                    )
                    raise ValueError(f"{message}.")


def _foreign_members(group: type[FeatureGroup], uuids: Iterable[UUID], graph: Graph) -> list[str]:
    nodes = graph.get_nodes()
    return sorted(
        nodes[uuid].feature_group_class.get_class_name()
        for uuid in uuids
        if not is_nominal_subclass(nodes[uuid].feature_group_class, group)
    )


def raise_on_dishonest_join_record(plan: ResolvedJoinPlan, join_steps: Iterable[JoinStep], graph: Graph) -> None:
    """Raise when a record's sides hold parents outside their declared group, or its ends overlap or miss their side."""
    step_of = {step.uuid: step for step in join_steps}
    for record in plan.records:
        step = step_of[record.token]
        for label, side in (("left", record.left), ("right", record.right)):
            foreign = _foreign_members(side.feature_group, side.uuids, graph)
            if foreign:
                raise ValueError(
                    internal_invariant_error(
                        f"Every {label} member of link {step.link} is a {side.feature_group.get_class_name()}.",
                        f"{label} side holds members of {foreign}.",
                    )
                )
        if record.jointype in (JoinType.APPEND, JoinType.UNION):
            continue
        if record.destination_framework is record.source_framework and not step.carriers:
            continue
        overlap = record.destination_uuids & record.source_uuids
        if overlap:
            raise ValueError(
                internal_invariant_error(
                    f"Join destination and source of link {step.link} are disjoint.",
                    f"Both name {sorted(str(uuid) for uuid in overlap)}.",
                )
            )
        for label, end, uuids in (
            ("destination", record.destination, record.destination_uuids),
            ("source", record.source, record.source_uuids),
        ):
            foreign = _foreign_members(end.feature_group, uuids, graph)
            if foreign:
                raise ValueError(
                    internal_invariant_error(
                        f"Every {label} member of link {step.link} is a {end.feature_group.get_class_name()}.",
                        f"{label} uuids hold members of {foreign}.",
                    )
                )
            if not uuids & end.uuids:
                raise ValueError(
                    internal_invariant_error(
                        f"The {label} end of link {step.link} reads a member of its {end.feature_group.get_class_name()} side.",
                        f"{label} uuids {sorted(str(uuid) for uuid in uuids)} share none with the side.",
                    )
                )


def _record_groups(records: tuple[ResolvedJoin, ...]) -> list[list[ResolvedJoin]]:
    """Records connected through shared consumers."""
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    first_with: dict[UUID, int] = {}
    for index, record in enumerate(records):
        for consumer in record.consumers:
            parent[find(index)] = find(first_with.setdefault(consumer, index))
    groups: dict[int, list[ResolvedJoin]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[find(index)].append(record)
    return list(groups.values())


def _sides_of(record: ResolvedJoin) -> frozenset[UUID]:
    return record.left.uuids | record.right.uuids


def _missed_join_reason(
    consumer: UUID, missed: ResolvedJoin, reached: set[UUID], own_records: list[ResolvedJoin], graph: Graph
) -> str:
    shared_side = missed.left if missed.left.uuids & reached else missed.right
    shared_uuids = shared_side.uuids & reached
    nodes = graph.get_nodes()
    own_consumers = frozenset().union(*(record.consumers for record in own_records))
    candidates = (missed.consumers & own_consumers) or missed.consumers
    other = sorted(nodes[uuid].feature_group_class.get_class_name() for uuid in candidates)[0]
    can_keep = True
    for record in own_records:
        for side in (record.left, record.right):
            if side.uuids & shared_uuids and _role(record, side) in ("outer", "dropped"):
                can_keep = False
    return star_missed_join_reason(
        nodes[consumer].feature_group_class.get_class_name(),
        other,
        _class_names(shared_uuids, graph),
        _class_names(_other_side(missed, shared_side).uuids, graph),
        can_keep,
    )


def raise_on_consumer_missing_a_star_join(plan: ResolvedJoinPlan, graph: Graph) -> None:
    """Raise when a consumer skips a join of a star that another consumer reads and the skipped join changes its rows."""
    ancestors = graph.parent_to_children_mapping
    for group in _record_groups(plan.records):
        if len(group) < 2:
            continue
        by_consumer: dict[UUID, set[int]] = defaultdict(set)
        for index, record in enumerate(group):
            for consumer in record.consumers:
                by_consumer[consumer].add(index)
        for consumer, own_indexes in by_consumer.items():
            missed = [
                record
                for index, record in enumerate(group)
                if index not in own_indexes
                and record.jointype not in (JoinType.APPEND, JoinType.UNION)
                and not _is_self_link(record)
                and not any(consumer in ancestors.get(uuid, ()) for uuid in _sides_of(record))
            ]
            if not missed:
                continue
            own_records = [group[index] for index in own_indexes]
            reached = set().union(*(_sides_of(record) for record in own_records))
            pending = missed
            changed = True
            while changed:
                changed = False
                still_pending = []
                for record in pending:
                    if not _sides_of(record) & reached:
                        still_pending.append(record)
                        continue
                    kept_and_dropped = _kept_and_dropped(record)
                    if kept_and_dropped is None or not kept_and_dropped[0].uuids & reached:
                        raise ValueError(f"{_missed_join_reason(consumer, record, reached, own_records, graph)}.")
                    reached.update(kept_and_dropped[1].uuids)
                    changed = True
                pending = still_pending
