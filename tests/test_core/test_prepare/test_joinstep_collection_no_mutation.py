"""JoinStepCollection: get_required_join_uuids must not insert a key on a cache miss, and
earlier_joins_sharing_destination returns only earlier same-link joins with an overlapping destination."""

from __future__ import annotations

from uuid import UUID, uuid4

from mloda.core.core.step.join_step import JoinStep
from mloda.core.prepare.joinstep_collection import JoinStepCollection
from mloda.provider import FeatureGroup
from mloda.user import Index, JoinSpec, Link
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

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


def test_earlier_joins_sharing_destination_ignores_another_links_join() -> None:
    coll = JoinStepCollection()
    shared = uuid4()
    other_link = _link()
    coll.add(_step(other_link, {shared}))
    later = _step(_link(), {shared})
    coll.add(later)

    assert coll.earlier_joins_sharing_destination(later) == set()


def test_earlier_joins_sharing_destination_ignores_disjoint_destinations() -> None:
    coll = JoinStepCollection()
    link = _link()
    coll.add(_step(link, {uuid4()}))
    later = _step(link, {uuid4()})
    coll.add(later)

    assert coll.earlier_joins_sharing_destination(later) == set()
