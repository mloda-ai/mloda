"""The materialized join decision: one record per planned join orientation.

Shadow mode, nothing runs yet. Metadata only (classes, uuids, indices, scalars), so a record pickles to a worker.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple
from uuid import UUID

from mloda.core.abstract_plugins.components.index.index import Index
from mloda.core.abstract_plugins.components.link import JoinType, Link
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup


class JoinSide(Enum):
    LEFT = "left"
    RIGHT = "right"


def signed_uuids(uuids: Iterable[UUID]) -> tuple[str, ...]:
    """Order-free uuid rendering, so two representations of one join sign it the same way."""
    return tuple(sorted(str(uuid) for uuid in uuids))


@dataclass(frozen=True)
class ResolvedJoinSide:
    """One declared side of a link, with the parents that are that group."""

    feature_group: type[FeatureGroup]
    index: Index
    uuids: frozenset[UUID]
    declared_frameworks: frozenset[type[ComputeFramework]]


@dataclass(frozen=True)
class PlannedOrientation:
    """The join an orientation decided, not just the trekker key it was planned under."""

    link: Link
    consumers: frozenset[UUID]
    destination_side: JoinSide
    left_uuids: frozenset[UUID]
    right_uuids: frozenset[UUID]
    # True when destination/source uuids are already in declared left/right order (INNER/LEFT/RIGHT joins only).
    sides_in_declared_order: bool = False


@dataclass(frozen=True)
class DeclinedOrientation:
    """An orientation of a link that planned no join step."""

    link_uuid: UUID
    left_framework: type[ComputeFramework]
    right_framework: type[ComputeFramework]


class JoinSignature(NamedTuple):
    """The join a record or a join step names, comparable across both."""

    link_uuid: UUID
    jointype: str
    destination_uuids: tuple[str, ...]
    source_uuids: tuple[str, ...]
    destination_side: str
    destination_framework: str
    source_framework: str
    depends_on_links: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedJoin:
    """One join decision: the declared side it runs in, what it moves, what it waits for."""

    link_uuid: UUID
    jointype: JoinType
    left: ResolvedJoinSide
    right: ResolvedJoinSide
    destination_side: JoinSide
    destination_uuids: frozenset[UUID]
    source_uuids: frozenset[UUID]
    destination_framework: type[ComputeFramework]
    source_framework: type[ComputeFramework]
    consumers: frozenset[UUID]
    # Join dependency edges only, not the write serialization edges add_tfs adds later.
    depends_on: frozenset[UUID]
    token: UUID
    # Correlation only, never authoritative: the join step this record shadows.
    shadowed_step_uuid: UUID

    @property
    def inverted(self) -> bool:
        return self.destination_side is JoinSide.RIGHT

    # Known gap: on a case-override orientation whose destination_side is RIGHT, destination/source below can
    # name a different uuid set than destination_uuids/source_uuids; see
    # test_a_case_override_right_destination_crosses_the_legacy_destination_uuids.
    @property
    def destination(self) -> ResolvedJoinSide:
        return self.right if self.inverted else self.left

    @property
    def source(self) -> ResolvedJoinSide:
        return self.left if self.inverted else self.right

    @property
    def transform_from_feature_group(self) -> type[FeatureGroup]:
        return self.source.feature_group

    @property
    def transform_to_feature_group(self) -> type[FeatureGroup]:
        return self.destination.feature_group

    def signature(self, link_of_token: Mapping[UUID, UUID]) -> JoinSignature:
        return JoinSignature(
            link_uuid=self.link_uuid,
            jointype=self.jointype.value,
            destination_uuids=signed_uuids(self.destination_uuids),
            source_uuids=signed_uuids(self.source_uuids),
            destination_side=self.destination_side.value,
            destination_framework=self.destination_framework.get_class_name(),
            source_framework=self.source_framework.get_class_name(),
            depends_on_links=tuple(sorted({str(link_of_token[token]) for token in self.depends_on})),
        )


@dataclass(frozen=True)
class ResolvedJoinPlan:
    """Every record of a run, together with the orientations that planned nothing."""

    records: tuple[ResolvedJoin, ...]
    declined: tuple[DeclinedOrientation, ...]

    def link_of_token(self) -> dict[UUID, UUID]:
        return {record.token: record.link_uuid for record in self.records}

    def signatures(self) -> frozenset[JoinSignature]:
        link_of_token = self.link_of_token()
        return frozenset(record.signature(link_of_token) for record in self.records)

    def records_of_link(self, link_uuid: UUID) -> tuple[ResolvedJoin, ...]:
        return tuple(record for record in self.records if record.link_uuid == link_uuid)
