"""Splits a join's parents by the declared sides of its link, once, for run_link and the resolved join records."""

from collections import defaultdict
from typing import NamedTuple
from uuid import UUID

from mloda.core.abstract_plugins.components.link import Link
from mloda.core.prepare.graph.graph import Graph
from mloda.core.prepare.resolve_links import inheritance_distance


class DeclaredSideSplit(NamedTuple):
    """The parents each declared side of a link owns."""

    left_uuids: frozenset[UUID]
    right_uuids: frozenset[UUID]
    left_uuids_any_distance: frozenset[UUID]
    right_uuids_any_distance: frozenset[UUID]


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
