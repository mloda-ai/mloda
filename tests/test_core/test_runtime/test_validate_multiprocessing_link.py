"""Unit tests for raise_on_unpicklable_join_link, hand-building Link/JoinStep objects.
A feature group class created inside a local function has "<locals>" in its
__qualname__ and is genuinely unpicklable, which is what a JoinStep queued to a
multiprocessing worker would otherwise fail deep inside pickle for."""

from __future__ import annotations

import pytest

from mloda.core.core.step.join_step import JoinStep
from mloda.provider import FeatureGroup
from mloda.user import Index, JoinSpec, Link
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.core.runtime.validate_multiprocessing_link import raise_on_unpicklable_join_link


class ValidateLinkLeft(FeatureGroup):
    """Ordinary module-level feature group: picklable."""


class ValidateLinkRight(FeatureGroup):
    """Ordinary module-level feature group: picklable."""


def _make_local_feature_group() -> type[FeatureGroup]:
    """A FeatureGroup subclass whose __qualname__ contains '<locals>': unpicklable."""

    class LocallyDefinedFeatureGroup(FeatureGroup):
        pass

    return LocallyDefinedFeatureGroup


def _joinstep_for(link: Link) -> JoinStep:
    return JoinStep(link, PyArrowTable, PandasDataFrame, set(), set(), set())


def test_a_link_with_an_unpicklable_left_feature_group_is_rejected() -> None:
    unpicklable_left = _make_local_feature_group()
    link = Link.inner(
        JoinSpec(unpicklable_left, Index(("left_key",))),
        JoinSpec(ValidateLinkRight, Index(("right_key",))),
    )
    step = _joinstep_for(link)

    with pytest.raises(ValueError) as excinfo:
        raise_on_unpicklable_join_link([step])

    message = str(excinfo.value)
    assert unpicklable_left.__name__ in message, f"the offending class must be named; got: {message}"
    assert str(link.uuid) in message, f"the link must be named; got: {message}"


def test_a_link_with_an_unpicklable_right_feature_group_is_rejected() -> None:
    unpicklable_right = _make_local_feature_group()
    link = Link.inner(
        JoinSpec(ValidateLinkLeft, Index(("left_key",))),
        JoinSpec(unpicklable_right, Index(("right_key",))),
    )
    step = _joinstep_for(link)

    with pytest.raises(ValueError) as excinfo:
        raise_on_unpicklable_join_link([step])

    message = str(excinfo.value)
    assert unpicklable_right.__name__ in message, f"the offending class must be named; got: {message}"
    assert str(link.uuid) in message, f"the link must be named; got: {message}"


def test_a_link_whose_both_sides_are_picklable_does_not_raise() -> None:
    link = Link.inner(
        JoinSpec(ValidateLinkLeft, Index(("left_key",))),
        JoinSpec(ValidateLinkRight, Index(("right_key",))),
    )
    step = _joinstep_for(link)

    raise_on_unpicklable_join_link([step])


def test_non_joinstep_entries_are_ignored_not_crashed_on() -> None:
    link = Link.inner(
        JoinSpec(ValidateLinkLeft, Index(("left_key",))),
        JoinSpec(ValidateLinkRight, Index(("right_key",))),
    )
    step = _joinstep_for(link)

    raise_on_unpicklable_join_link([object(), None, step])
