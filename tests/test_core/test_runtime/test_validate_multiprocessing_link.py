"""Unit tests for raise_on_unpicklable_join_link, hand-building Link/JoinStep objects.
A feature group class built via type(name, (FeatureGroup,), {}) is genuinely
unpicklable (pickle cannot resolve it back by module/qualname), which is what a
JoinStep queued to a multiprocessing worker would otherwise fail deep inside
pickle for."""

from __future__ import annotations

import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.core.step.join_step import JoinStep
from mloda.core.core.step.transform_frame_work_step import TransformFrameworkStep
from mloda.provider import FeatureGroup
from mloda.user import Index, JoinSpec, Link
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.core.runtime.validate_multiprocessing_link import (
    raise_on_unpicklable_join_link,
    raise_on_unpicklable_multiprocessing_steps,
)
from mloda_plugins.feature_group.experimental.dynamic_feature_group_factory.dynamic_feature_group_factory import (
    DynamicFeatureGroupCreator,
)


class ValidateLinkLeft(FeatureGroup):
    """Ordinary module-level feature group: picklable."""


class ValidateLinkRight(FeatureGroup):
    """Ordinary module-level feature group: picklable."""


def _make_local_feature_group(name: str = "LocallyDefinedFeatureGroup") -> type[FeatureGroup]:
    """A dynamically built FeatureGroup subclass pickle cannot resolve back by name: unpicklable."""

    return type(name, (FeatureGroup,), {})


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
    assert f"references {unpicklable_left.__name__} (" in message, f"the offending class must be named; got: {message}"
    assert f"references {ValidateLinkRight.__name__} (" not in message, (
        f"the picklable side must not be blamed; got: {message}"
    )
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
    assert f"references {unpicklable_right.__name__} (" in message, f"the offending class must be named; got: {message}"
    assert f"references {ValidateLinkLeft.__name__} (" not in message, (
        f"the picklable side must not be blamed; got: {message}"
    )
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


def test_a_link_with_both_feature_groups_unpicklable_blames_the_left_one() -> None:
    unpicklable_left = _make_local_feature_group("LeftUnpicklableFeatureGroup")
    unpicklable_right = _make_local_feature_group("RightUnpicklableFeatureGroup")
    link = Link.inner(
        JoinSpec(unpicklable_left, Index(("left_key",))),
        JoinSpec(unpicklable_right, Index(("right_key",))),
    )
    step = _joinstep_for(link)

    with pytest.raises(ValueError) as excinfo:
        raise_on_unpicklable_join_link([step])

    message = str(excinfo.value)
    assert f"references {unpicklable_left.__name__} (" in message, (
        f"the left side must be blamed first, per the left-then-right check order; got: {message}"
    )


def test_a_link_with_unpicklable_discriminator_and_picklable_feature_groups_is_rejected() -> None:
    link = Link.inner(
        JoinSpec(ValidateLinkLeft, Index(("left_key",))),
        JoinSpec(ValidateLinkRight, Index(("right_key",))),
        left_discriminator={"marker": _make_local_feature_group("DiscriminatorMarkerFeatureGroup")},
    )
    step = _joinstep_for(link)

    with pytest.raises(ValueError) as excinfo:
        raise_on_unpicklable_join_link([step])

    message = str(excinfo.value)
    assert f"references {ValidateLinkLeft.__name__} (" not in message, (
        f"neither feature group is individually unpicklable; got: {message}"
    )
    assert f"references {ValidateLinkRight.__name__} (" not in message, (
        f"neither feature group is individually unpicklable; got: {message}"
    )


def test_a_link_with_a_dynamic_feature_group_creator_class_is_rejected() -> None:
    dynamic_fg = DynamicFeatureGroupCreator.create(
        properties={}, class_name="ReviewProbeDynamicFeatureGroup_ValidateLinkTest"
    )
    link = Link.inner(
        JoinSpec(dynamic_fg, Index(("left_key",))),
        JoinSpec(ValidateLinkRight, Index(("right_key",))),
    )
    step = _joinstep_for(link)

    with pytest.raises(ValueError) as excinfo:
        raise_on_unpicklable_join_link([step])

    message = str(excinfo.value)
    assert dynamic_fg.__name__ in message, f"the dynamically created class must be named; got: {message}"


def test_feature_group_step_with_unpicklable_class_is_rejected() -> None:
    unpicklable_fg = _make_local_feature_group("DynamicFeatureGroupStepTarget")
    step = FeatureGroupStep(unpicklable_fg, FeatureSet(), set(), PandasDataFrame)

    with pytest.raises(ValueError) as excinfo:
        raise_on_unpicklable_multiprocessing_steps([step])

    message = str(excinfo.value)
    assert f"FeatureGroupStep for {unpicklable_fg.__name__} (" in message, (
        f"the offending class must be named; got: {message}"
    )


def test_feature_group_step_with_picklable_class_does_not_raise() -> None:
    step = FeatureGroupStep(ValidateLinkLeft, FeatureSet(), set(), PandasDataFrame)
    raise_on_unpicklable_multiprocessing_steps([step])


def test_transform_framework_step_with_unpicklable_from_class_is_rejected() -> None:
    unpicklable_from = _make_local_feature_group("DynamicFromFeatureGroup")
    step = TransformFrameworkStep(
        PandasDataFrame, PyArrowTable, set(), unpicklable_from, ValidateLinkRight
    )

    with pytest.raises(ValueError) as excinfo:
        raise_on_unpicklable_multiprocessing_steps([step])

    message = str(excinfo.value)
    assert f"TransformFrameworkStep references {unpicklable_from.__name__} (" in message, (
        f"the offending from-class must be named; got: {message}"
    )


def test_transform_framework_step_with_unpicklable_to_class_is_rejected() -> None:
    unpicklable_to = _make_local_feature_group("DynamicToFeatureGroup")
    step = TransformFrameworkStep(
        PandasDataFrame, PyArrowTable, set(), ValidateLinkLeft, unpicklable_to
    )

    with pytest.raises(ValueError) as excinfo:
        raise_on_unpicklable_multiprocessing_steps([step])

    message = str(excinfo.value)
    assert f"TransformFrameworkStep references {unpicklable_to.__name__} (" in message, (
        f"the offending to-class must be named; got: {message}"
    )


def test_transform_framework_step_with_picklable_classes_does_not_raise() -> None:
    step = TransformFrameworkStep(
        PandasDataFrame, PyArrowTable, set(), ValidateLinkLeft, ValidateLinkRight
    )
    raise_on_unpicklable_multiprocessing_steps([step])

