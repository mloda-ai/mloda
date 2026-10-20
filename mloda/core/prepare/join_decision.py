"""The orientation decided for one part of one link key, before any JoinStep is built."""

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.prepare.declared_sides import DeclaredSideSplit
from mloda.core.prepare.resolve_links import LinkFrameworkTrekker
from mloda.core.prepare.resolved_join import JoinSide

if TYPE_CHECKING:
    from mloda.core.prepare.execution_plan import AppendOrUnionSides


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
    carried_left: bool | None
    bound_buckets: tuple[frozenset[UUID], frozenset[UUID]] | None
    append_sides: "AppendOrUnionSides | None"
    fanned_out: bool
