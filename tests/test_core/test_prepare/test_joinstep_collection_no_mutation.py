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
    "build, expected",
    [
        pytest.param(
            lambda: _fw_step(PyArrowTable, PandasDataFrame),
            lambda a, b: (set(), {a}, {b}),
            id="overlapping_frameworks",
        ),
        pytest.param(
            lambda: _fw_step(PyArrowTable, PyArrowTable),
            lambda a, b: (set(), set(), set()),
            id="same_destination_and_source",
        ),
    ],
)
def test_each_join_waits_on_its_open_joins_with_distinct_sets(
    build: Callable[[], JoinStep],
    expected: Callable[[UUID, UUID], tuple[set[UUID], set[UUID], set[UUID]]],
) -> None:
    first, second, third = build(), build(), build()

    coll, first_set, second_set, third_set = _add_three(first, second, third)

    assert (first_set, second_set, third_set) == expected(first.uuid, second.uuid)
    for step in (first, second, third):
        assert step.uuid not in coll.get_required_join_uuids(step)
    assert first_set is not second_set
    assert second_set is not third_set
    assert first_set is not third_set
    assert coll.get_required_join_uuids(first) == set()
    assert coll.get_required_join_uuids(second) == expected(first.uuid, second.uuid)[1]


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


_PA, _PD, _DICT = PyArrowTable, PandasDataFrame, PythonDictFramework
_FwPair = tuple[type[ComputeFramework], type[ComputeFramework]]

_SEQUENCES = [
    pytest.param([(_PA, _PD), (_PA, _PA), (_PA, _DICT), (_PA, _PD)], id="mixed"),
    pytest.param([(_PA, _PA), (_PA, _PD), (_PA, _PA), (_PD, _PA)], id="same_framework_first_and_after_cross"),
    pytest.param([(_PA, _PD), (_PD, _PD), (_DICT, _PD), (_PA, _PD)], id="same_framework_on_source"),
    pytest.param([(_PA, _PD), (_PD, _DICT), (_DICT, _PA), (_PA, _PD)], id="ring"),
]


@pytest.mark.parametrize("pairs", _SEQUENCES)
def test_tokens_stay_within_sharing_joins_and_reach_all_of_them_through_cross_joins(pairs: list[_FwPair]) -> None:
    coll = JoinStepCollection()
    steps = [_fw_step(destination, source) for destination, source in pairs]
    for step in steps:
        coll.add(step)
    by_uuid = {step.uuid: step for step in steps}

    for index, step in enumerate(steps):
        frameworks = {step.destination_framework, step.source_framework}
        sharing = {
            earlier.uuid
            for earlier in steps[:index]
            if frameworks & {earlier.destination_framework, earlier.source_framework}
        }
        tokens = coll.get_required_join_uuids(step)
        assert tokens <= sharing
        if step.destination_framework == step.source_framework:
            continue
        reached: set[UUID] = set()
        stack = list(tokens)
        while stack:
            token = stack.pop()
            if token in reached:
                continue
            reached.add(token)
            joined = by_uuid[token]
            if joined.destination_framework != joined.source_framework:
                stack.extend(coll.get_required_join_uuids(joined))
        assert reached >= sharing


def test_a_cross_join_waits_on_what_was_open_and_later_joins_reach_the_rest_through_it() -> None:
    coll = JoinStepCollection()
    a, s, c, d = _fw_step(_PA, _PD), _fw_step(_PA, _PA), _fw_step(_PA, _DICT), _fw_step(_PA, _PD)
    for step in (a, s, c, d):
        coll.add(step)

    assert coll.get_required_join_uuids(c) == {a.uuid, s.uuid}
    assert coll.get_required_join_uuids(d) == {c.uuid, a.uuid}


def test_a_same_framework_join_after_a_cross_join_stays_open_for_the_next_cross_join() -> None:
    coll = JoinStepCollection()
    s1, c, s2, e = _fw_step(_PA, _PA), _fw_step(_PA, _PD), _fw_step(_PA, _PA), _fw_step(_PD, _PA)
    for step in (s1, c, s2, e):
        coll.add(step)

    assert coll.get_required_join_uuids(e) == {c.uuid, s2.uuid}


def test_fifty_cross_framework_joins_each_carry_one_token() -> None:
    coll = JoinStepCollection()
    steps = [_fw_step(_PA, _PD) for _ in range(50)]
    for step in steps:
        coll.add(step)

    assert coll.get_required_join_uuids(steps[0]) == set()
    for previous, step in zip(steps, steps[1:]):
        assert coll.get_required_join_uuids(step) == {previous.uuid}


def test_destination_index_gives_the_same_answer_as_a_brute_force_comparison() -> None:
    coll = JoinStepCollection()
    link = _link()
    feature_one, feature_two, owner_step, raw = uuid4(), uuid4(), uuid4(), uuid4()
    owner = {feature_one: owner_step, feature_two: owner_step}
    first = _step(link, {feature_one})
    second = _step(link, {feature_two})
    third = _step(link, {raw})
    fourth = _step(link, {raw, uuid4()})
    fifth = _step(link, {uuid4()})
    steps = [first, second, third, fourth, fifth]
    for step in steps:
        coll.add(step)

    index = coll.joins_by_destination_owner(owner)

    def owners(step: JoinStep) -> set[UUID]:
        return {owner.get(u, u) for u in step.destination_framework_uuids}

    never_added = _step(link, {feature_two})
    for position, step in enumerate([*steps, never_added]):
        expected = {earlier.uuid for earlier in steps[:position] if owners(earlier) & owners(step)}
        assert coll.earlier_joins_sharing_destination(step, owner, index) == expected
    assert type(index) is dict
    assert coll.earlier_joins_sharing_destination(second, owner, index) == {first.uuid}
    assert coll.earlier_joins_sharing_destination(fourth, owner, index) == {third.uuid}
    assert index[owner_step] == [first, second]
