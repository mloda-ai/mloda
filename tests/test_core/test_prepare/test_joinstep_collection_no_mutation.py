"""JoinStepCollection: get_required_join_uuids must not insert a key on a cache miss, and
earlier_joins_sharing_destination returns earlier joins (of any link) with an overlapping destination."""

from __future__ import annotations

from typing import Callable
from uuid import UUID, uuid4

import pytest

from mloda.core.core.step.join_step import JoinStep
from mloda.core.prepare.joinstep_collection import JoinStepCollection
from mloda.provider import ComputeFramework, FeatureGroup
from mloda.user import Index, JoinSpec, Link
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework

_LEFT_INDEX = Index(("left_key",))
_RIGHT_INDEX = Index(("right_key",))


class _CollLeft(FeatureGroup):
    pass


class _CollRight(FeatureGroup):
    pass


def _link() -> Link:
    return Link.inner(JoinSpec(_CollLeft, _LEFT_INDEX), JoinSpec(_CollRight, _RIGHT_INDEX))


def test_get_required_join_uuids_returns_empty_set_for_unregistered_step() -> None:
    coll = JoinStepCollection()
    link = _link()
    step = JoinStep(link, PyArrowTable, PandasDataFrame, set(), set(), set())

    result = coll.get_required_join_uuids(step)

    assert result == set()


def test_get_required_join_uuids_does_not_mutate_collection_on_miss() -> None:
    coll = JoinStepCollection()
    link = _link()
    registered = JoinStep(link, PyArrowTable, PandasDataFrame, set(), set(), set())
    unregistered = JoinStep(link, PandasDataFrame, PyArrowTable, set(), set(), set())

    coll.add(registered)
    assert len(coll.collection) == 1

    coll.get_required_join_uuids(unregistered)

    assert len(coll.collection) == 1, (
        "looking up an unregistered JoinStep must not insert a new entry into the collection"
    )


def _step(link: Link, destination_uuids: set[UUID]) -> JoinStep:
    return JoinStep(link, PyArrowTable, PandasDataFrame, set(), destination_uuids, set())


def test_earlier_joins_sharing_destination_orders_same_link_joins_by_insertion() -> None:
    coll = JoinStepCollection()
    link = _link()
    shared = uuid4()
    earlier = _step(link, {shared})
    later = _step(link, {shared, uuid4()})
    coll.add(earlier)
    coll.add(later)

    assert coll.earlier_joins_sharing_destination(later) == {earlier.uuid}
    assert coll.earlier_joins_sharing_destination(earlier) == set()


def test_earlier_joins_sharing_destination_returns_another_links_join() -> None:
    coll = JoinStepCollection()
    shared = uuid4()
    earlier = _step(_link(), {shared})
    coll.add(earlier)
    later = _step(_link(), {shared})
    coll.add(later)

    assert coll.earlier_joins_sharing_destination(later) == {earlier.uuid}


def test_earlier_joins_sharing_destination_ignores_disjoint_destinations() -> None:
    coll = JoinStepCollection()
    link = _link()
    coll.add(_step(link, {uuid4()}))
    later = _step(link, {uuid4()})
    coll.add(later)

    assert coll.earlier_joins_sharing_destination(later) == set()


def _fw_step(
    destination: type[ComputeFramework], source: type[ComputeFramework], destination_uuids: set[UUID] | None = None
) -> JoinStep:
    return JoinStep(_link(), destination, source, set(), destination_uuids or set(), set())


def _add_three(
    first: JoinStep, second: JoinStep, third: JoinStep
) -> tuple[JoinStepCollection, set[UUID], set[UUID], set[UUID]]:
    coll = JoinStepCollection()
    coll.add(first)
    first_set = coll.get_required_join_uuids(first)
    coll.add(second)
    second_set = coll.get_required_join_uuids(second)
    coll.add(third)
    return coll, first_set, second_set, coll.get_required_join_uuids(third)


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: _fw_step(PyArrowTable, PandasDataFrame), id="overlapping_frameworks"),
        pytest.param(lambda: _fw_step(PyArrowTable, PyArrowTable), id="same_destination_and_source"),
    ],
)
def test_each_join_waits_on_exactly_the_earlier_joins_with_distinct_sets(build: Callable[[], JoinStep]) -> None:
    first, second, third = build(), build(), build()

    coll, first_set, second_set, third_set = _add_three(first, second, third)

    assert first_set == set()
    assert second_set == {first.uuid}
    assert third_set == {first.uuid, second.uuid}
    for step in (first, second, third):
        assert step.uuid not in coll.get_required_join_uuids(step)
    assert first_set is not second_set
    assert second_set is not third_set
    assert first_set is not third_set
    assert coll.get_required_join_uuids(first) == set()
    assert coll.get_required_join_uuids(second) == {first.uuid}


def test_a_join_on_untouched_frameworks_waits_on_nothing() -> None:
    coll = JoinStepCollection()
    coll.add(_fw_step(PyArrowTable, PandasDataFrame))
    coll.add(_fw_step(PandasDataFrame, PyArrowTable))
    unrelated = _fw_step(PythonDictFramework, PythonDictFramework)

    coll.add(unrelated)

    assert coll.get_required_join_uuids(unrelated) == set()


def test_earlier_joins_sharing_destination_for_a_join_never_added_returns_every_match() -> None:
    coll = JoinStepCollection()
    link = _link()
    shared = uuid4()
    first = _step(link, {shared})
    second = _step(link, {shared, uuid4()})
    disjoint = _step(link, {uuid4()})
    other_link = _step(_link(), {shared})
    for step in (first, second, disjoint, other_link):
        coll.add(step)

    never_added = _step(link, {shared})

    assert coll.earlier_joins_sharing_destination(never_added) == {first.uuid, second.uuid, other_link.uuid}


def test_collection_keeps_joins_in_insertion_order() -> None:
    coll = JoinStepCollection()
    link = _link()
    steps = [_step(link, {uuid4()}) for _ in range(4)]
    for step in steps:
        coll.add(step)

    assert list(coll.collection) == steps
    assert isinstance(coll.collection, dict)
