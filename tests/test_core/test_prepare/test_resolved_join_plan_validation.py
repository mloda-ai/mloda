"""A parent read as the source of two joins that share a consumer, with no join writing back into
that parent, leaves two branches that can never reunite."""

from dataclasses import replace
from typing import Any, NamedTuple
from uuid import UUID

import pytest

from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.core.prepare.graph.graph import Graph
from mloda.core.prepare.graph.properties import NodeProperties
from mloda.core.prepare.resolve_links import LinkTrekker
from mloda.core.prepare.resolved_join import ResolvedJoin, ResolvedJoinPlan
from mloda.core.prepare.validate_resolved_join import (
    raise_on_consumer_missing_a_star_join,
    raise_on_join_star_keeping_two_groups,
    raise_on_orphaned_join_source,
)
from mloda.provider import FeatureGroup
from mloda.user import Index, JoinSpec, Link
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from tests.test_core.test_prepare.join_plan_helpers import feature, trek
from tests.test_plugins.compute_framework.test_tooling.shared_compute_frameworks import SecondCfw, ThirdCfw


SHARED_INDEX = Index(("orphan_source_idx",))
UNRELATED_INDEX = Index(("unrelated_writeback_idx",))

SOURCE_CFW = PyArrowTable
FIRST_DESTINATION_CFW = PythonDictFramework
SECOND_DESTINATION_CFW = PandasDataFrame
UNRELATED_SOURCE_CFW = SecondCfw
UNRELATED_CONSUMER_CFW = ThirdCfw


class OrphanSourceParent(FeatureGroup):
    """The parent whose original data two joins both read but neither writes back into."""


class OrphanFirstDestination(FeatureGroup):
    """Declared left of the first link; the first join's destination."""


class OrphanSecondDestination(FeatureGroup):
    """Declared right of the second link; the child's framework, so this join inverts onto it."""


class OrphanConsumer(FeatureGroup):
    """Needs all three parents, through both joins."""


class UnrelatedWritebackSource(FeatureGroup):
    """Joined with the shared parent for a consumer that has nothing to do with OrphanConsumer."""


class UnrelatedWritebackConsumer(FeatureGroup):
    """Needs only the shared parent and the unrelated source, never OrphanConsumer's destinations."""


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
    plan.create_execution_plan(queue, graph, link_trekker, validate=False)
    return Built(
        plan, source.uuid, first_destination.uuid, second_destination.uuid, consumer.uuid, first_link, second_link
    )


class BuiltWithUnrelatedWriteback(NamedTuple):
    plan: ExecutionPlan
    source_uuid: UUID
    consumer_uuid: UUID
    unrelated_consumer_uuid: UUID


def _build_orphaned_source_plan_with_unrelated_writeback() -> BuiltWithUnrelatedWriteback:
    """Same orphaned pair as _build_orphaned_source_plan, plus a third join for an unrelated
    consumer that writes its result back into the shared parent's uuid."""
    source = feature("orphan_source", SOURCE_CFW, SHARED_INDEX)
    first_destination = feature("orphan_first_destination", FIRST_DESTINATION_CFW, SHARED_INDEX)
    second_destination = feature("orphan_second_destination", SECOND_DESTINATION_CFW, SHARED_INDEX)
    consumer = feature("orphan_consumer", SECOND_DESTINATION_CFW)
    unrelated_source = feature("unrelated_writeback_source", UNRELATED_SOURCE_CFW, UNRELATED_INDEX)
    unrelated_consumer = feature("unrelated_writeback_consumer", UNRELATED_CONSUMER_CFW)

    first_link = Link.inner(JoinSpec(OrphanFirstDestination, SHARED_INDEX), JoinSpec(OrphanSourceParent, SHARED_INDEX))
    second_link = Link.inner(
        JoinSpec(OrphanSourceParent, SHARED_INDEX), JoinSpec(OrphanSecondDestination, SHARED_INDEX)
    )
    # Natural order (shared parent declared left): the shared parent is this join's destination.
    unrelated_link = Link.inner(
        JoinSpec(OrphanSourceParent, SHARED_INDEX), JoinSpec(UnrelatedWritebackSource, UNRELATED_INDEX)
    )

    graph = Graph()
    graph.add_node(source.uuid, NodeProperties(source, OrphanSourceParent))
    graph.add_node(first_destination.uuid, NodeProperties(first_destination, OrphanFirstDestination))
    graph.add_node(second_destination.uuid, NodeProperties(second_destination, OrphanSecondDestination))
    graph.add_node(consumer.uuid, NodeProperties(consumer, OrphanConsumer))
    graph.add_node(unrelated_source.uuid, NodeProperties(unrelated_source, UnrelatedWritebackSource))
    graph.add_node(unrelated_consumer.uuid, NodeProperties(unrelated_consumer, UnrelatedWritebackConsumer))
    graph.adjacency_list[source.uuid].append(consumer.uuid)
    graph.adjacency_list[source.uuid].append(unrelated_consumer.uuid)
    graph.adjacency_list[first_destination.uuid].append(consumer.uuid)
    graph.adjacency_list[second_destination.uuid].append(consumer.uuid)
    graph.adjacency_list[unrelated_source.uuid].append(unrelated_consumer.uuid)
    graph.adjacency_list[consumer.uuid] = []
    graph.adjacency_list[unrelated_consumer.uuid] = []
    graph.parent_to_children_mapping[consumer.uuid] = {source.uuid, first_destination.uuid, second_destination.uuid}
    graph.parent_to_children_mapping[unrelated_consumer.uuid] = {source.uuid, unrelated_source.uuid}

    link_trekker = LinkTrekker()
    trek(link_trekker, first_link, (FIRST_DESTINATION_CFW, SOURCE_CFW), consumer.uuid)
    trek(link_trekker, second_link, (SECOND_DESTINATION_CFW, SOURCE_CFW), consumer.uuid)
    trek(link_trekker, unrelated_link, (SOURCE_CFW, UNRELATED_SOURCE_CFW), unrelated_consumer.uuid)

    queue: list[Any] = [
        (OrphanSourceParent, {source}),
        (OrphanFirstDestination, {first_destination}),
        (OrphanSecondDestination, {second_destination}),
        (UnrelatedWritebackSource, {unrelated_source}),
        (first_link, FIRST_DESTINATION_CFW, SOURCE_CFW),
        (second_link, SOURCE_CFW, SECOND_DESTINATION_CFW),
        (unrelated_link, SOURCE_CFW, UNRELATED_SOURCE_CFW),
        (OrphanConsumer, {consumer}),
        (UnrelatedWritebackConsumer, {unrelated_consumer}),
    ]

    plan = ExecutionPlan()
    plan.create_execution_plan(queue, graph, link_trekker, validate=False)
    return BuiltWithUnrelatedWriteback(plan, source.uuid, consumer.uuid, unrelated_consumer.uuid)


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
    assert OrphanFirstDestination.get_class_name() in message, (
        f"one competing join must be identifiable; got: {message}"
    )
    assert OrphanSecondDestination.get_class_name() in message, (
        f"the other competing join must be identifiable; got: {message}"
    )


def test_the_unrelated_writeback_fixture_writes_into_the_shared_parent_for_a_different_consumer() -> None:
    """Pins the shape: the orphaned pair for OrphanConsumer is unchanged, and a third, unrelated
    join writes the shared parent's own uuid into its destination_uuids for UnrelatedConsumer."""
    built = _build_orphaned_source_plan_with_unrelated_writeback()
    resolved = built.plan.resolved_join_plan

    assert not resolved.declined
    assert len(resolved.records) == 3

    all_destination_uuids: set[UUID] = set()
    orphaned_pair = []
    for record in resolved.records:
        all_destination_uuids |= record.destination_uuids
        if built.source_uuid in record.source_uuids:
            orphaned_pair.append(record)

    assert built.source_uuid in all_destination_uuids, (
        "the unrelated join must write back into the shared parent for this to test the right scope"
    )
    assert len(orphaned_pair) == 2, "the two joins reading the shared parent as their source are unchanged"
    assert orphaned_pair[0].consumers & orphaned_pair[1].consumers == {built.consumer_uuid}
    assert built.unrelated_consumer_uuid not in (orphaned_pair[0].consumers | orphaned_pair[1].consumers), (
        "the write-back's consumer must share nothing with the orphaned pair's consumer"
    )


def test_raise_on_orphaned_join_source_still_raises_despite_an_unrelated_writeback() -> None:
    """An unrelated join writing back into the shared parent for a different consumer must not
    silence the check for the pair that actually shares a consumer and never reunites."""
    built = _build_orphaned_source_plan_with_unrelated_writeback()

    with pytest.raises(ValueError, match="is read as the join source"):
        raise_on_orphaned_join_source(built.plan.resolved_join_plan)


def test_raise_on_orphaned_join_source_accepts_a_plan_with_no_records() -> None:
    """An empty plan has no source to lose."""
    raise_on_orphaned_join_source(ExecutionPlan().resolved_join_plan)


class StarFirstSpoke(FeatureGroup):
    """One spoke of a join star."""


class StarSecondSpoke(FeatureGroup):
    """The other spoke of a join star."""


class StarHub(FeatureGroup):
    """The group both spokes join with."""


class StarConsumer(FeatureGroup):
    """Reads the spokes and the hub."""


class StarOtherConsumer(FeatureGroup):
    """A second consumer, reading the hub through the second link only."""


STAR_INDEX = Index(("star_idx",))
STAR_FIRST_CFW = PyArrowTable
STAR_SECOND_CFW = PandasDataFrame
STAR_HUB_CFW = PythonDictFramework


def _build_star_plan(
    make_links: Any, *, separate_consumers: bool = False, partial_consumer: bool = False
) -> tuple[ExecutionPlan, Graph]:
    """Two joins over StarFirstSpoke, StarSecondSpoke and StarHub, built by make_links(first, second, hub)."""
    first = feature("star_first", STAR_FIRST_CFW, STAR_INDEX)
    second = feature("star_second", STAR_SECOND_CFW, STAR_INDEX)
    hub = feature("star_hub", STAR_HUB_CFW, STAR_INDEX)
    consumer = feature("star_consumer", STAR_SECOND_CFW)
    other_consumer = feature("star_other_consumer", STAR_SECOND_CFW)

    graph = Graph()
    graph.add_node(first.uuid, NodeProperties(first, StarFirstSpoke))
    graph.add_node(second.uuid, NodeProperties(second, StarSecondSpoke))
    graph.add_node(hub.uuid, NodeProperties(hub, StarHub))
    graph.add_node(consumer.uuid, NodeProperties(consumer, StarConsumer))
    graph.add_node(other_consumer.uuid, NodeProperties(other_consumer, StarOtherConsumer))
    consumer_parents = (first, hub) if separate_consumers else (first, second, hub)
    for parent in consumer_parents:
        graph.adjacency_list[parent.uuid].append(consumer.uuid)
    other_parents = (second, hub) if separate_consumers else (first, hub) if partial_consumer else ()
    for parent in other_parents:
        graph.adjacency_list[parent.uuid].append(other_consumer.uuid)
    graph.adjacency_list[consumer.uuid] = []
    graph.adjacency_list[other_consumer.uuid] = []
    graph.parent_to_children_mapping[consumer.uuid] = {parent.uuid for parent in consumer_parents}
    if separate_consumers or partial_consumer:
        graph.parent_to_children_mapping[other_consumer.uuid] = {parent.uuid for parent in other_parents}

    cfws = {StarFirstSpoke: STAR_FIRST_CFW, StarSecondSpoke: STAR_SECOND_CFW, StarHub: STAR_HUB_CFW}
    links: list[Link] = make_links(StarFirstSpoke, StarSecondSpoke, StarHub)
    link_trekker = LinkTrekker()
    queue: list[Any] = [
        (StarFirstSpoke, {first}),
        (StarSecondSpoke, {second}),
        (StarHub, {hub}),
    ]
    for position, link in enumerate(links):
        owner = other_consumer if separate_consumers and position == 1 else consumer
        left_cfw, right_cfw = cfws[link.left_feature_group], cfws[link.right_feature_group]
        trek(link_trekker, link, (left_cfw, right_cfw), owner.uuid)
        if partial_consumer and position == 0:
            trek(link_trekker, link, (left_cfw, right_cfw), other_consumer.uuid)
        queue.append((link, left_cfw, right_cfw))
    queue.append((StarConsumer, {consumer}))
    if separate_consumers or partial_consumer:
        queue.append((StarOtherConsumer, {other_consumer}))

    plan = ExecutionPlan()
    plan.create_execution_plan(queue, graph, link_trekker, validate=False)
    return plan, graph


def _two_spokes_kept(first: Any, second: Any, hub: Any) -> list[Link]:
    return [
        Link.left(JoinSpec(first, STAR_INDEX), JoinSpec(hub, STAR_INDEX)),
        Link.left(JoinSpec(second, STAR_INDEX), JoinSpec(hub, STAR_INDEX)),
    ]


def _hub_kept(first: Any, second: Any, hub: Any) -> list[Link]:
    return [
        Link.left(JoinSpec(hub, STAR_INDEX), JoinSpec(first, STAR_INDEX)),
        Link.left(JoinSpec(hub, STAR_INDEX), JoinSpec(second, STAR_INDEX)),
    ]


def _chain(first: Any, second: Any, hub: Any) -> list[Link]:
    return [
        Link.left(JoinSpec(first, STAR_INDEX), JoinSpec(hub, STAR_INDEX)),
        Link.left(JoinSpec(hub, STAR_INDEX), JoinSpec(second, STAR_INDEX)),
    ]


def _snowflake_chain(first: Any, second: Any, hub: Any) -> list[Link]:
    """The hub joins first on STAR_INDEX and second on another index, so the joins are order independent."""
    other_index = Index(("star_other_idx",))
    return [
        Link.left(JoinSpec(first, STAR_INDEX), JoinSpec(hub, STAR_INDEX)),
        Link.left(JoinSpec(hub, other_index), JoinSpec(second, other_index)),
    ]


def _foreign_key_chain(kind_second: str) -> Any:
    """The first spoke joins the hub from a differently named key column; the second joins on the hub's key."""

    def make_links(first: Any, second: Any, hub: Any) -> list[Link]:
        key_name = "fk"
        hub_spec, second_spec = JoinSpec(hub, STAR_INDEX), JoinSpec(second, STAR_INDEX)
        if kind_second == "outer_other_key":
            return [
                Link.left(hub_spec, JoinSpec(first, STAR_INDEX)),
                Link.outer(hub_spec, JoinSpec(second, key_name)),
            ]
        return [Link.left(JoinSpec(first, key_name), hub_spec), Link.left(hub_spec, second_spec)]

    return make_links


def _kept_and_outer_on_different_indexes(first: Any, second: Any, hub: Any) -> list[Link]:
    other_index = Index(("star_other_idx",))
    return [
        Link.left(JoinSpec(hub, STAR_INDEX), JoinSpec(first, STAR_INDEX)),
        Link.outer(JoinSpec(hub, other_index), JoinSpec(second, other_index)),
    ]


def _star_of(kind_first: str, first_hub_first: bool, kind_second: str, second_hub_first: bool) -> Any:
    """Links of the given kinds into the hub, each with the hub declared first or second."""

    def make_links(first: Any, second: Any, hub: Any) -> list[Link]:
        links = []
        for kind, spoke, hub_first in ((kind_first, first, first_hub_first), (kind_second, second, second_hub_first)):
            sides = (hub, spoke) if hub_first else (spoke, hub)
            left_spec, right_spec = JoinSpec(sides[0], STAR_INDEX), JoinSpec(sides[1], STAR_INDEX)
            if kind == "asof":
                links.append(Link.asof(left_spec, right_spec, left_time_column="t", right_time_column="t"))
            else:
                links.append(getattr(Link, kind)(left_spec, right_spec))
        return links

    return make_links


def test_raise_on_join_star_keeping_two_groups_names_the_consumer_the_kept_groups_and_the_dropped_group() -> None:
    plan, graph = _build_star_plan(_two_spokes_kept)

    with pytest.raises(ValueError) as excinfo:
        raise_on_join_star_keeping_two_groups(plan.resolved_join_plan, graph)

    message = str(excinfo.value)
    for name in ("StarConsumer", "StarFirstSpoke", "StarSecondSpoke", "StarHub"):
        assert name in message, f"{name} must be named; got: {message}"
    assert "not defined" in message
    assert "Link.star(" in message
    assert 'index_column=..., jointype="left")' in message
    assert "two left joins" in message
    assert "right or left" not in message
    assert "inner joins for both or outer joins for both" in message
    assert "separate consumers" in message


_ORDER_DEPENDENT_STARS = [
    pytest.param(_star_of("left", False, "inner", False), id="left_keeping_a_spoke_plus_inner"),
    pytest.param(_star_of("left", False, "outer", False), id="left_keeping_a_spoke_plus_outer"),
    pytest.param(_star_of("left", True, "outer", False), id="left_keeping_the_hub_plus_outer"),
    pytest.param(_star_of("inner", False, "outer", False), id="inner_plus_outer"),
    pytest.param(_chain, id="chain_through_the_middle_group"),
    pytest.param(_star_of("right", False, "left", False), id="right_keeping_the_hub_plus_left_dropping_it"),
]


@pytest.mark.parametrize("make_links", _ORDER_DEPENDENT_STARS)
def test_raise_on_join_star_keeping_two_groups_rejects_joins_whose_rows_depend_on_which_runs_first(
    make_links: Any,
) -> None:
    plan, graph = _build_star_plan(make_links)

    with pytest.raises(ValueError) as excinfo:
        raise_on_join_star_keeping_two_groups(plan.resolved_join_plan, graph)

    message = str(excinfo.value)
    for name in ("StarConsumer", "StarFirstSpoke", "StarSecondSpoke", "StarHub"):
        assert name in message, f"{name} must be named; got: {message}"
    assert "runs first" in message
    assert "separate consumers" in message


def test_raise_on_join_star_keeping_two_groups_rejects_asofs_keeping_different_groups() -> None:
    plan, graph = _build_star_plan(_star_of("asof", False, "asof", False))

    with pytest.raises(ValueError, match="keep different groups") as excinfo:
        raise_on_join_star_keeping_two_groups(plan.resolved_join_plan, graph)

    message = str(excinfo.value)
    for name in ("StarConsumer", "StarFirstSpoke", "StarSecondSpoke", "StarHub"):
        assert name in message, f"{name} must be named; got: {message}"
    assert "Link.left(" not in message
    assert 'jointype="left"' not in message
    assert "Link.asof(" in message


@pytest.mark.parametrize(
    "make_links, separate_consumers",
    [
        pytest.param(_hub_kept, False, id="hub_kept_star"),
        pytest.param(_two_spokes_kept, True, id="different_consumers"),
        pytest.param(_star_of("inner", False, "inner", True), False, id="inner_with_inner"),
        pytest.param(_star_of("outer", False, "outer", True), False, id="outer_with_outer"),
        pytest.param(_star_of("left", True, "inner", False), False, id="kept_hub_with_inner"),
        pytest.param(_star_of("right", False, "left", True), False, id="kept_with_kept"),
        pytest.param(_snowflake_chain, False, id="chain_on_different_index_columns"),
        pytest.param(_foreign_key_chain("left"), False, id="dropped_then_kept_on_differently_named_keys"),
        pytest.param(_foreign_key_chain("outer_other_key"), False, id="kept_with_outer_on_differently_named_keys"),
        pytest.param(_kept_and_outer_on_different_indexes, False, id="kept_with_outer_on_different_indexes"),
        pytest.param(_star_of("asof", True, "asof", True), False, id="asof_keeping_the_hub_twice"),
    ],
)
def test_raise_on_join_star_keeping_two_groups_accepts_plans_with_a_defined_result(
    make_links: Any, separate_consumers: bool
) -> None:
    plan, graph = _build_star_plan(make_links, separate_consumers=separate_consumers)

    assert len(plan.resolved_join_plan.records) == 2
    raise_on_join_star_keeping_two_groups(plan.resolved_join_plan, graph)


def test_raise_on_join_star_keeping_two_groups_accepts_two_steps_of_one_kept_group_class() -> None:
    other_index = Index(("star_other_idx",))
    first_arrow = feature("star_first", STAR_FIRST_CFW, STAR_INDEX)
    first_pandas = feature("star_first", STAR_SECOND_CFW, other_index)
    hub = feature("star_hub", STAR_HUB_CFW, STAR_INDEX)
    consumer = feature("star_consumer", STAR_SECOND_CFW)

    graph = Graph()
    graph.add_node(first_arrow.uuid, NodeProperties(first_arrow, StarFirstSpoke))
    graph.add_node(first_pandas.uuid, NodeProperties(first_pandas, StarFirstSpoke))
    graph.add_node(hub.uuid, NodeProperties(hub, StarHub))
    graph.add_node(consumer.uuid, NodeProperties(consumer, StarConsumer))
    parents = (first_arrow, first_pandas, hub)
    for parent in parents:
        graph.adjacency_list[parent.uuid].append(consumer.uuid)
    graph.adjacency_list[consumer.uuid] = []
    graph.parent_to_children_mapping[consumer.uuid] = {parent.uuid for parent in parents}

    links = [
        Link.left(JoinSpec(StarFirstSpoke, STAR_INDEX), JoinSpec(StarHub, STAR_INDEX)),
        Link.left(JoinSpec(StarFirstSpoke, other_index), JoinSpec(StarHub, other_index)),
    ]
    orientations = [(STAR_FIRST_CFW, STAR_HUB_CFW), (STAR_SECOND_CFW, STAR_HUB_CFW)]
    link_trekker = LinkTrekker()
    queue: list[Any] = [(StarFirstSpoke, {first_arrow, first_pandas}), (StarHub, {hub})]
    for link, orientation in zip(links, orientations):
        trek(link_trekker, link, orientation, consumer.uuid)
        queue.append((link, *orientation))
    queue.append((StarConsumer, {consumer}))

    plan = ExecutionPlan()
    plan.create_execution_plan(queue, graph, link_trekker, validate=False)

    assert len(plan.resolved_join_plan.records) == 2
    raise_on_join_star_keeping_two_groups(plan.resolved_join_plan, graph)


def test_raise_on_join_star_keeping_two_groups_accepts_a_plan_with_no_records() -> None:
    raise_on_join_star_keeping_two_groups(ExecutionPlan().resolved_join_plan, Graph())


def _star_missed_join_kept(missed_kind: str) -> Any:
    def make_links(first: Any, second: Any, hub: Any) -> list[Link]:
        return [
            Link.inner(JoinSpec(first, STAR_INDEX), JoinSpec(hub, STAR_INDEX)),
            getattr(Link, missed_kind)(JoinSpec(second, STAR_INDEX), JoinSpec(hub, STAR_INDEX)),
        ]

    return make_links


class StarThirdConsumer(FeatureGroup):
    """A third consumer, sorting after the others."""


def _shared_record_plan(make_links: Any) -> tuple[ResolvedJoinPlan, Graph]:
    """The star plan with one record per link, the consumers of a link's records merged."""
    plan, graph = _build_star_plan(make_links, partial_consumer=True)
    merged: dict[Any, ResolvedJoin] = {}
    for record in plan.resolved_join_plan.records:
        key = record.link_uuid
        merged[key] = (
            replace(merged[key], consumers=merged[key].consumers | record.consumers) if key in merged else record
        )
    return ResolvedJoinPlan(records=tuple(merged.values()), declined=plan.resolved_join_plan.declined), graph


def _consumer_uuids(graph: Graph) -> dict[str, Any]:
    return {node.feature_group_class.get_class_name(): uuid for uuid, node in graph.get_nodes().items()}


def test_raise_on_consumer_missing_a_star_join_names_both_consumers_and_the_hub() -> None:
    plan, graph = _shared_record_plan(_star_missed_join_kept("inner"))

    with pytest.raises(ValueError) as excinfo:
        raise_on_consumer_missing_a_star_join(plan, graph)

    message = str(excinfo.value)
    for name in ("StarOtherConsumer", "StarConsumer", "StarHub"):
        assert name in message, f"{name} must be named; got: {message}"
    assert "Resolution:" in message
    assert "left or right" in message


def test_raise_on_consumer_missing_a_star_join_does_not_offer_a_left_or_right_link_for_outer_joins() -> None:
    plan, graph = _shared_record_plan(_star_of("outer", False, "outer", False))

    with pytest.raises(ValueError) as excinfo:
        raise_on_consumer_missing_a_star_join(plan, graph)

    message = str(excinfo.value)
    assert "separate runs" in message
    assert "left or right" not in message


def test_raise_on_consumer_missing_a_star_join_pairs_the_consumer_with_one_that_reads_a_join_it_reads() -> None:
    plan, graph = _shared_record_plan(_star_missed_join_kept("inner"))
    third = feature("star_third_consumer", STAR_SECOND_CFW)
    graph.add_node(third.uuid, NodeProperties(third, StarThirdConsumer))
    uuids = _consumer_uuids(graph)
    shared_first, shared_second = uuids["StarConsumer"], uuids["StarOtherConsumer"]
    # StarOtherConsumer reads the first join only, StarConsumer the second only, the third consumer both.
    plan = ResolvedJoinPlan(
        records=(
            replace(plan.records[0], consumers=frozenset({shared_second, third.uuid})),
            replace(plan.records[1], consumers=frozenset({shared_first, third.uuid})),
        ),
        declined=plan.declined,
    )

    with pytest.raises(ValueError) as excinfo:
        raise_on_consumer_missing_a_star_join(plan, graph)

    assert "and StarThirdConsumer in separate runs" in str(excinfo.value)


def test_raise_on_consumer_missing_a_star_join_accepts_a_missed_join_keeping_the_shared_group() -> None:
    plan, graph = _shared_record_plan(_star_missed_join_kept("right"))

    assert len(plan.records) == 2
    raise_on_consumer_missing_a_star_join(plan, graph)


def test_raise_on_consumer_missing_a_star_join_accepts_a_plan_with_no_records() -> None:
    raise_on_consumer_missing_a_star_join(ExecutionPlan().resolved_join_plan, Graph())
