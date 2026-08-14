"""A parent read as the source of two joins that share a consumer, with no join writing back into
that parent, leaves two branches that can never reunite."""

from typing import Any, NamedTuple
from uuid import UUID

import pytest

from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.core.prepare.graph.graph import Graph
from mloda.core.prepare.graph.properties import NodeProperties
from mloda.core.prepare.resolve_links import LinkTrekker
from mloda.core.prepare.validate_resolved_join import raise_on_orphaned_join_source
from mloda.provider import FeatureGroup
from mloda.user import Index, JoinSpec, Link
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from tests.test_core.test_prepare.join_plan_helpers import feature, trek


SHARED_INDEX = Index(("orphan_source_idx",))

SOURCE_CFW = PyArrowTable
FIRST_DESTINATION_CFW = PythonDictFramework
SECOND_DESTINATION_CFW = PandasDataFrame


class OrphanSourceParent(FeatureGroup):
    """The parent whose original data two joins both read but neither writes back into."""


class OrphanFirstDestination(FeatureGroup):
    """Declared left of the first link; the first join's destination."""


class OrphanSecondDestination(FeatureGroup):
    """Declared right of the second link; the child's framework, so this join inverts onto it."""


class OrphanConsumer(FeatureGroup):
    """Needs all three parents, through both joins."""


class Built(NamedTuple):
    plan: ExecutionPlan
    source_uuid: UUID
    first_destination_uuid: UUID
    second_destination_uuid: UUID
    consumer_uuid: UUID
    first_link: Link
    second_link: Link


def _build_orphaned_source_plan() -> Built:
    """Both joins read the shared parent as their source and write to distinct destinations."""
    source = feature("orphan_source", SOURCE_CFW, SHARED_INDEX)
    first_destination = feature("orphan_first_destination", FIRST_DESTINATION_CFW, SHARED_INDEX)
    second_destination = feature("orphan_second_destination", SECOND_DESTINATION_CFW, SHARED_INDEX)
    consumer = feature("orphan_consumer", SECOND_DESTINATION_CFW)

    first_link = Link.inner(JoinSpec(OrphanFirstDestination, SHARED_INDEX), JoinSpec(OrphanSourceParent, SHARED_INDEX))
    second_link = Link.inner(
        JoinSpec(OrphanSourceParent, SHARED_INDEX), JoinSpec(OrphanSecondDestination, SHARED_INDEX)
    )

    graph = Graph()
    graph.add_node(source.uuid, NodeProperties(source, OrphanSourceParent))
    graph.add_node(first_destination.uuid, NodeProperties(first_destination, OrphanFirstDestination))
    graph.add_node(second_destination.uuid, NodeProperties(second_destination, OrphanSecondDestination))
    graph.add_node(consumer.uuid, NodeProperties(consumer, OrphanConsumer))
    graph.adjacency_list[source.uuid].append(consumer.uuid)
    graph.adjacency_list[first_destination.uuid].append(consumer.uuid)
    graph.adjacency_list[second_destination.uuid].append(consumer.uuid)
    graph.adjacency_list[consumer.uuid] = []
    graph.parent_to_children_mapping[consumer.uuid] = {source.uuid, first_destination.uuid, second_destination.uuid}

    link_trekker = LinkTrekker()
    # Natural key (declared left, declared right): first_link needs no inversion.
    trek(link_trekker, first_link, (FIRST_DESTINATION_CFW, SOURCE_CFW), consumer.uuid)
    # Flipped key: second_link's queue entry below uses the declared order, so run_link finds
    # nothing there and inverts onto (second_destination, source).
    trek(link_trekker, second_link, (SECOND_DESTINATION_CFW, SOURCE_CFW), consumer.uuid)

    queue: list[Any] = [
        (OrphanSourceParent, {source}),
        (OrphanFirstDestination, {first_destination}),
        (OrphanSecondDestination, {second_destination}),
        (first_link, FIRST_DESTINATION_CFW, SOURCE_CFW),
        (second_link, SOURCE_CFW, SECOND_DESTINATION_CFW),
        (OrphanConsumer, {consumer}),
    ]

    plan = ExecutionPlan()
    plan.create_execution_plan(queue, graph, link_trekker)
    return Built(
        plan, source.uuid, first_destination.uuid, second_destination.uuid, consumer.uuid, first_link, second_link
    )


def test_the_fixture_reads_the_shared_source_twice_and_writes_back_to_neither_join() -> None:
    """Pins the shape the check is meant to catch, independently of the check itself."""
    built = _build_orphaned_source_plan()
    resolved = built.plan.resolved_join_plan

    assert not resolved.declined
    assert len(resolved.records) == 2

    all_destination_uuids: set[UUID] = set()
    records_sourcing_the_shared_parent = []
    for record in resolved.records:
        all_destination_uuids |= record.destination_uuids
        if built.source_uuid in record.source_uuids:
            records_sourcing_the_shared_parent.append(record)

    assert len(records_sourcing_the_shared_parent) == 2, "both joins must read the shared parent as their source"
    assert built.source_uuid not in all_destination_uuids, "no join may write back into the shared parent"
    assert records_sourcing_the_shared_parent[0].consumers & records_sourcing_the_shared_parent[1].consumers == {
        built.consumer_uuid
    }


def test_raise_on_orphaned_join_source_raises_naming_the_dropped_feature_group() -> None:
    built = _build_orphaned_source_plan()

    with pytest.raises(ValueError) as excinfo:
        raise_on_orphaned_join_source(built.plan.resolved_join_plan)

    message = str(excinfo.value)
    assert OrphanSourceParent.get_class_name() in message, f"the dropped side must be named; got: {message}"
    assert OrphanFirstDestination.get_class_name() in message or str(built.first_link.uuid) in message, (
        f"one competing join must be identifiable; got: {message}"
    )
    assert OrphanSecondDestination.get_class_name() in message or str(built.second_link.uuid) in message, (
        f"the other competing join must be identifiable; got: {message}"
    )


def test_raise_on_orphaned_join_source_accepts_a_plan_with_no_records() -> None:
    """An empty plan has no source to lose."""
    raise_on_orphaned_join_source(ExecutionPlan().resolved_join_plan)
