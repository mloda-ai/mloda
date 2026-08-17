"""Unit tests for raise_on_unpicklable_step_feature_group, hand-building FeatureGroupStep and
TransformFrameworkStep objects. A feature group class built via type(name, (FeatureGroup,), {}) is
genuinely unpicklable (pickle cannot resolve it back by module/qualname), which is what such a step
queued to a multiprocessing worker would otherwise fail deep inside pickle for."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.core.step.join_step import JoinStep
from mloda.core.core.step.transform_frame_work_step import TransformFrameworkStep
from mloda.core.runtime.validate_multiprocessing_link import raise_on_unpicklable_step_feature_group
from mloda.provider import FeatureGroup
from mloda.user import Index, JoinSpec, Link
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.feature_group.experimental.dynamic_feature_group_factory.dynamic_feature_group_factory import (
    DynamicFeatureGroupCreator,
)

_DYNAMIC_CLASS_NAME = "ReviewProbeDynamicFeatureGroup_ValidateStepTest"


@pytest.fixture(autouse=True)
def _cleanup_dynamic_feature_groups() -> Iterator[None]:
    yield
    DynamicFeatureGroupCreator._created_classes.pop(_DYNAMIC_CLASS_NAME, None)


class ValidateStepFeatureGroup(FeatureGroup):
    """Ordinary module-level feature group: picklable."""


class SyncOnlyComputeFramework(ComputeFramework):
    """A compute framework restricted to SYNC, like SqliteFramework or DuckDBFramework."""

    @classmethod
    def supported_parallelization_modes(cls) -> set[ParallelizationMode]:
        return {ParallelizationMode.SYNC}


def _make_local_feature_group(name: str = "LocallyDefinedFeatureGroup") -> type[FeatureGroup]:
    """A dynamically built FeatureGroup subclass pickle cannot resolve back by name: unpicklable."""

    return type(name, (FeatureGroup,), {})


def _feature_group_step(
    feature_group: type[FeatureGroup], compute_framework: type[ComputeFramework] = PandasDataFrame
) -> FeatureGroupStep:
    return FeatureGroupStep(feature_group, FeatureSet([Feature("f")]), set(), compute_framework)


def _transform_step(
    from_feature_group: type[FeatureGroup], to_feature_group: type[FeatureGroup]
) -> TransformFrameworkStep:
    return TransformFrameworkStep(PandasDataFrame, PyArrowTable, set(), from_feature_group, to_feature_group)


def test_a_feature_group_step_with_an_unpicklable_feature_group_is_rejected() -> None:
    unpicklable = _make_local_feature_group()
    step = _feature_group_step(unpicklable)

    with pytest.raises(ValueError) as excinfo:
        raise_on_unpicklable_step_feature_group([step])

    message = str(excinfo.value)
    assert f"references {unpicklable.__name__} (" in message, f"the offending class must be named; got: {message}"
    assert "FeatureGroupStep" in message, f"the step type must be named; got: {message}"


def test_a_feature_group_step_with_a_picklable_feature_group_does_not_raise() -> None:
    step = _feature_group_step(ValidateStepFeatureGroup)

    raise_on_unpicklable_step_feature_group([step])


def test_a_feature_group_step_whose_compute_framework_does_not_support_multiprocessing_is_not_rejected() -> None:
    """A step that will never actually route to a multiprocessing worker must not be blocked."""
    unpicklable = _make_local_feature_group()
    step = _feature_group_step(unpicklable, compute_framework=SyncOnlyComputeFramework)

    raise_on_unpicklable_step_feature_group([step])


def test_a_feature_group_step_unpicklable_for_a_reason_other_than_its_feature_group_is_rejected() -> None:
    """The whole step crosses the multiprocessing queue, not just its feature_group attribute."""
    feature = Feature("f", options=Options({"unpicklable": lambda: None}))
    step = FeatureGroupStep(ValidateStepFeatureGroup, FeatureSet([feature]), set(), PandasDataFrame)

    with pytest.raises(ValueError) as excinfo:
        raise_on_unpicklable_step_feature_group([step])

    message = str(excinfo.value)
    assert f"references {ValidateStepFeatureGroup.__name__} (" not in message, (
        f"the feature group is individually picklable; got: {message}"
    )
    assert "FeatureGroupStep" in message, f"the step type must be named; got: {message}"


def test_a_transform_framework_step_with_an_unpicklable_from_feature_group_is_rejected() -> None:
    unpicklable = _make_local_feature_group("UnpicklableFromFeatureGroup")
    step = _transform_step(unpicklable, ValidateStepFeatureGroup)

    with pytest.raises(ValueError) as excinfo:
        raise_on_unpicklable_step_feature_group([step])

    message = str(excinfo.value)
    assert f"references {unpicklable.__name__} (" in message, f"the offending class must be named; got: {message}"
    assert f"references {ValidateStepFeatureGroup.__name__} (" not in message, (
        f"the picklable side must not be blamed; got: {message}"
    )
    assert "TransformFrameworkStep" in message, f"the step type must be named; got: {message}"


def test_a_transform_framework_step_with_an_unpicklable_to_feature_group_is_rejected() -> None:
    unpicklable = _make_local_feature_group("UnpicklableToFeatureGroup")
    step = _transform_step(ValidateStepFeatureGroup, unpicklable)

    with pytest.raises(ValueError) as excinfo:
        raise_on_unpicklable_step_feature_group([step])

    message = str(excinfo.value)
    assert f"references {unpicklable.__name__} (" in message, f"the offending class must be named; got: {message}"
    assert f"references {ValidateStepFeatureGroup.__name__} (" not in message, (
        f"the picklable side must not be blamed; got: {message}"
    )


def test_a_transform_framework_step_with_both_feature_groups_unpicklable_blames_the_from_one() -> None:
    unpicklable_from = _make_local_feature_group("FromUnpicklableFeatureGroup")
    unpicklable_to = _make_local_feature_group("ToUnpicklableFeatureGroup")
    step = _transform_step(unpicklable_from, unpicklable_to)

    with pytest.raises(ValueError) as excinfo:
        raise_on_unpicklable_step_feature_group([step])

    message = str(excinfo.value)
    assert f"references {unpicklable_from.__name__} (" in message, (
        f"the from side must be blamed first, per the from-then-to check order; got: {message}"
    )


def test_a_transform_framework_step_with_both_feature_groups_picklable_does_not_raise() -> None:
    step = _transform_step(ValidateStepFeatureGroup, ValidateStepFeatureGroup)

    raise_on_unpicklable_step_feature_group([step])


def test_non_matching_step_entries_are_ignored_not_crashed_on() -> None:
    raise_on_unpicklable_step_feature_group([object(), None])


def test_a_join_step_is_ignored_by_this_function() -> None:
    unpicklable = _make_local_feature_group("UnpicklableJoinStepFeatureGroup")
    link = Link.inner(
        JoinSpec(unpicklable, Index(("left_key",))),
        JoinSpec(ValidateStepFeatureGroup, Index(("right_key",))),
    )
    step = JoinStep(link, PyArrowTable, PandasDataFrame, set(), set(), set())

    raise_on_unpicklable_step_feature_group([step])


def test_a_feature_group_step_with_a_dynamic_feature_group_creator_class_is_rejected() -> None:
    """Mirrors ConcatenatedFileContent._create_join_class in read_context_files.py."""
    dynamic_fg = DynamicFeatureGroupCreator.create(properties={}, class_name=_DYNAMIC_CLASS_NAME)
    step = _feature_group_step(dynamic_fg)

    with pytest.raises(ValueError) as excinfo:
        raise_on_unpicklable_step_feature_group([step])

    message = str(excinfo.value)
    assert dynamic_fg.__name__ in message, f"the dynamically created class must be named; got: {message}"
