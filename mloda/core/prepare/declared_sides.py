"""Splits a join's parents by the declared sides of its link, once, for run_link and the resolved join records."""

from collections import defaultdict
from collections.abc import Iterable
from typing import NamedTuple
from uuid import UUID

from mloda.core.abstract_plugins.components.link import JoinType, Link
from mloda.core.prepare.graph.graph import Graph
from mloda.core.prepare.resolve_links import inheritance_distance


class DeclaredSideSplit(NamedTuple):
    """The parents each declared side of a link owns."""

    left_uuids: frozenset[UUID]
    right_uuids: frozenset[UUID]
    left_uuids_any_distance: frozenset[UUID]
    right_uuids_any_distance: frozenset[UUID]


def _member_links(link: Link, first: type, second: type) -> str:
    """The per-member links that replace one base-class link."""
    left, right = link.left_feature_group.__name__, link.right_feature_group.__name__
    if issubclass(first, link.left_feature_group):
        return f"Link({first.__name__}, {right}) and Link({second.__name__}, {right})"
    return f"Link({left}, {first.__name__}) and Link({left}, {second.__name__})"


def one_framework_members_reason(consumer: str, link: Link, first: type, second: type) -> str:
    """Reason for one consumer reading two members of one link side on one framework, without final period."""
    return (
        f"{consumer} joins {first.__name__} and {second.__name__}, members of one link side, on one compute "
        "framework over one link: one join would keep only one member's columns. Read one member of the link side "
        f"per consumer, and declare one link per member: {_member_links(link, first, second)}"
    )


def several_consumers_members_reason(consumers: str, link: Link, first: type, second: type) -> str:
    """Reason for several consumers each reading one member of one link side over one link."""
    return (
        f"The consumers {consumers} each join one of {first.__name__} and {second.__name__}, members of one link "
        "side, on one compute framework over one link, which merges both into one join: declare one link per "
        f"member: {_member_links(link, first, second)}."
    )


def star_keeping_two_groups_reason(
    consumer: str, first: str, second: str, dropped: str, jointypes: Iterable[JoinType]
) -> str:
    """Reason for one consumer reading left or right joins that keep different groups and drop the same one."""
    kinds = {jointype.value for jointype in jointypes}
    kind = " or ".join(sorted(kinds, reverse=True))
    if kinds == {"right"}:
        keep_links = f"Link.right({first}, {dropped}) and Link.right({second}, {dropped})"
    else:
        keep_links = f"Link.left({dropped}, {first}) and Link.left({dropped}, {second})"
    return (
        f"{consumer} reads two {kind} joins that keep different groups, {first} and {second}, and drop the same "
        f"group {dropped}: the rows depend on which join runs first, so the result is not defined.\n"
        f'Resolution: keep {dropped} in both joins ({keep_links}, or Link.star({dropped}, {first}, {second}, jointype="left")), '
        f"use inner joins for both or outer joins for both, or read {first} and {second} in separate consumers"
    )


def _nearest(uuids_by_distance: dict[int, set[UUID]]) -> frozenset[UUID]:
    """The minimum-distance bucket, or an empty side."""
    if not uuids_by_distance:
        return frozenset()
    return frozenset(uuids_by_distance[min(uuids_by_distance)])


def _any_distance(uuids_by_distance: dict[int, set[UUID]]) -> frozenset[UUID]:
    """Every distance bucket unioned, or an empty side."""
    return frozenset().union(*uuids_by_distance.values()) if uuids_by_distance else frozenset()


def split_by_declared_side(link: Link, uuids: set[UUID], graph: Graph) -> DeclaredSideSplit:
    """A declared side is held by its nearest subclasses only."""
    left_by_distance: dict[int, set[UUID]] = defaultdict(set)
    right_by_distance: dict[int, set[UUID]] = defaultdict(set)
    nodes = graph.get_nodes()

    for uuid in uuids:
        feature_group_class = nodes[uuid].feature_group_class
        if issubclass(feature_group_class, link.left_feature_group):
            left_by_distance[inheritance_distance(feature_group_class, link.left_feature_group)].add(uuid)
        if issubclass(feature_group_class, link.right_feature_group):
            right_by_distance[inheritance_distance(feature_group_class, link.right_feature_group)].add(uuid)

    return DeclaredSideSplit(
        _nearest(left_by_distance),
        _nearest(right_by_distance),
        _any_distance(left_by_distance),
        _any_distance(right_by_distance),
    )
