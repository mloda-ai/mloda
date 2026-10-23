"""The orientation decided for one part of one link key, before any JoinStep is built."""

from dataclasses import dataclass
from typing import NamedTuple
from uuid import UUID

from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.prepare.declared_sides import DeclaredSideSplit
from mloda.core.prepare.resolve_links import LinkFrameworkTrekker
from mloda.core.prepare.resolved_join import JoinSide


class AppendOrUnionSides(NamedTuple):
    """The left/right feature uuids and frameworks an APPEND or UNION link resolved to."""

    destination_framework: type[ComputeFramework]
    source_framework: type[ComputeFramework]
    left_uuid: UUID
    right_uuid: UUID


@dataclass(frozen=True)
class DeclinedJoin:
    """A key orientation no consumer accepts."""

    attempted_key: LinkFrameworkTrekker


@dataclass(frozen=True)
class JoinDecision:
    """Every side and framework choice for one part; building only reads it."""

    attempted_key: LinkFrameworkTrekker
    link_fw: LinkFrameworkTrekker
    consumers: frozenset[UUID]
    parent_uuids: frozenset[UUID]
    order_uuids: frozenset[UUID]
    split: DeclaredSideSplit
    destination_framework: type[ComputeFramework]
    source_framework: type[ComputeFramework]
    buckets_inverted: bool
    destination_side: JoinSide
    carriers: frozenset[UUID]
    shared_source: bool
    shared_destination: bool
    buckets: tuple[frozenset[UUID], frozenset[UUID]] | None  # (destination, source) parents; None for APPEND/UNION
    append_sides: AppendOrUnionSides | None
    fanned_out: bool
    token: UUID
    order: int = 0  # index among the accepted decisions of one link
    destination_owner: frozenset[UUID] = frozenset()  # owning steps of the destination uuids
    runs_after: frozenset[UUID] = frozenset()  # tokens of earlier joins of this link that must finish first
