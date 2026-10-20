"""Splits a join's parents by the declared sides of its link, once, for run_link and the resolved join records."""

from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any, NamedTuple, TypeVar
from uuid import UUID

from mloda.core.abstract_plugins.components.link import JoinType, Link
from mloda.core.prepare.graph.graph import Graph
from mloda.core.prepare.resolve_links import inheritance_distance

_T = TypeVar("_T")


class DeclaredSideSplit(NamedTuple):
    """The parents each declared side of a link owns."""

    left_uuids: frozenset[UUID]
    right_uuids: frozenset[UUID]
    left_uuids_any_distance: frozenset[UUID]
    right_uuids_any_distance: frozenset[UUID]


def narrow_by_right_index(link: Link, candidates: Iterable[_T], features: Callable[[_T], Iterable[Any]]) -> list[_T]:
    """Right-side candidates one link tells apart: by exact feature index, else by feature name in its key."""
    pool = list(candidates)
    exact = [c for c in pool if any(f.index is not None and f.index == link.right_index for f in features(c))]
    if exact:
        return exact
    return [c for c in pool if any(f.name in link.right_index.index for f in features(c))]


def _member_links(link: Link, first: type, second: type) -> str:
    """The per-member links that replace one base-class link, built with the link's own factory."""
    left, right = link.left_feature_group.__name__, link.right_feature_group.__name__
    factory = f"Link.{link.jointype.value}"

    def one(left_name: str, right_name: str) -> str:
        return f"{factory}(JoinSpec({left_name}, ...), JoinSpec({right_name}, ...))"

    if issubclass(first, link.left_feature_group):
        return f"{one(first.__name__, right)} and {one(second.__name__, right)}"
    if link.jointype in (JoinType.INNER, JoinType.OUTER):
        # symmetric joins: members lead, so the remedy reads the same for either link orientation
        return f"{one(first.__name__, left)} and {one(second.__name__, left)}"
    return f"{one(left, first.__name__)} and {one(left, second.__name__)}"


def one_framework_members_reason(consumer: str, link: Link, first: type, second: type) -> str:
    """Reason for one consumer reading two members of one link side on one framework, without final period."""
    return (
        f"{consumer} joins {first.__name__} and {second.__name__}, members of one link side, on one compute "
        "framework over one link: one join would keep only one member's columns. Read one member of the link side "
        f"per consumer, and declare one link per member, each read by its own consumer: "
        f"{_member_links(link, first, second)}"
    )


def several_consumers_members_reason(consumers: str, link: Link, first: type, second: type) -> str:
    """Reason for several consumers each reading one member of one link side over one link, without final period."""
    return (
        f"The consumers {consumers} each join one of {first.__name__} and {second.__name__}, members of one link "
        "side, on one compute framework over one link, which merges both into one "
        f"join: declare one link per member, each read by its own consumer: {_member_links(link, first, second)}"
    )


def star_keeping_two_groups_reason(
    consumer: str, first: str, second: str, dropped: str, jointypes: Iterable[JoinType]
) -> str:
    """Reason for one consumer reading left or right joins that keep different groups, without final period."""
    kinds = {jointype.value for jointype in jointypes}
    kind = " or ".join(sorted(kinds))
    if kinds == {"asof"}:
        keep_links = f"Link.asof({dropped}, {first}, ...) and Link.asof({dropped}, {second}, ...)"
        keep_text = f"keep {dropped} in both joins ({keep_links})"
    else:
        if kinds == {"right"}:
            keep_links = f"Link.right({first}, {dropped}) and Link.right({second}, {dropped})"
        else:
            keep_links = f"Link.left({dropped}, {first}) and Link.left({dropped}, {second})"
        star = f'Link.star({dropped}, {first}, {second}, index_column=..., jointype="left")'
        keep_text = f"keep {dropped} in both joins ({keep_links}, or {star})"
    return (
        f"{consumer} reads two {kind} joins that keep different groups, {first} and {second}, and drop the same "
        f"group {dropped}: the rows depend on which join runs first, so the result is not defined.\n"
        f"Resolution: {keep_text}, "
        f"use inner joins for both or outer joins for both, or read {first} and {second} in separate consumers"
    )


def star_mixed_joins_reason(
    consumer: str, shared: str, first: str, second: str, roles: tuple[str, str], jointypes: Iterable[JoinType]
) -> str:
    """Reason for one consumer reading two joins that treat a shared group differently, without final period."""
    kind = " and ".join(sorted({jointype.value for jointype in jointypes}))
    return (
        f"{consumer} reads two joins ({kind}) into {shared} with {first} and {second}, and {shared} is "
        f"{roles[0]} in one and {roles[1]} in the other: the rows depend on which join runs first, so the result "
        "is not defined.\n"
        f"Resolution: keep {shared} in every left or right join (an inner join beside them is fine), "
        f"use one join type for all links, or read {first} and {second} in separate consumers"
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
