"""Unit tests for raise_on_unpicklable_step_feature_group, hand-building FeatureGroupStep and
TransformFrameworkStep objects. A feature group class built via type(name, (FeatureGroup,), {}) is
genuinely unpicklable (pickle cannot resolve it back by module/qualname), which is what such a step
queued to a multiprocessing worker would otherwise fail deep inside pickle for."""

from __future__ import annotations

import pytest

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.core.step.transform_frame_work_step import TransformFrameworkStep
from mloda.core.runtime.validate_multiprocessing_link import raise_on_unpicklable_step_feature_group
from mloda.provider import FeatureGroup
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.feature_group.experimental.dynamic_feature_group_factory.dynamic_feature_group_factory import (
    DynamicFeatureGroupCreator,
)


class ValidateStepFeatureGroup(FeatureGroup):
    """Ordinary module-level feature group: picklable."""


def _make_local_feature_group(name: str = "LocallyDefinedFeatureGroup") -> type[FeatureGroup]:
    """A dynamically built FeatureGroup subclass pickle cannot resolve back by name: unpicklable."""

    return type(name, (FeatureGroup,), {})


def _feature_group_step(feature_group: type[FeatureGroup]) -> FeatureGroupStep:
    return FeatureGroupStep(feature_group, FeatureSet([Feature("f")]), set(), PandasDataFrame)


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


def test_a_transform_framework_step_with_an_unpicklable_from_feature_group_is_rejected() -> None:
    unpicklable = _make_local_feature_group("UnpicklableFromFeatureGroup")
    step = _transform_step(unpicklable, ValidateStepFeatureGroup)

    with pytest.raises(ValueError) as excinfo:
        raise_on_unpicklable_step_feature_group([step])

    message = str(excinfo.value)
    assert f"references {unpicklable.__name__} (" in message, f"the offending class must be named; got: {message}"
    assert "TransformFrameworkStep" in message, f"the step type must be named; got: {message}"


def test_a_transform_framework_step_with_an_unpicklable_to_feature_group_is_rejected() -> None:
    unpicklable = _make_local_feature_group("UnpicklableToFeatureGroup")
    step = _transform_step(ValidateStepFeatureGroup, unpicklable)

    with pytest.raises(ValueError) as excinfo:
        raise_on_unpicklable_step_feature_group([step])

    message = str(excinfo.value)
    assert f"references {unpicklable.__name__} (" in message, f"the offending class must be named; got: {message}"


def test_a_transform_framework_step_with_both_feature_groups_picklable_does_not_raise() -> None:
    step = _transform_step(ValidateStepFeatureGroup, ValidateStepFeatureGroup)

    raise_on_unpicklable_step_feature_group([step])


def test_non_matching_step_entries_are_ignored_not_crashed_on() -> None:
    raise_on_unpicklable_step_feature_group([object(), None])


def test_a_feature_group_step_with_a_dynamic_feature_group_creator_class_is_rejected() -> None:
    """Mirrors ConcatenatedFileContent._create_join_class in read_context_files.py."""
    dynamic_fg = DynamicFeatureGroupCreator.create(
        properties={}, class_name="ReviewProbeDynamicFeatureGroup_ValidateStepTest"
    )
    step = _feature_group_step(dynamic_fg)

    with pytest.raises(ValueError) as excinfo:
        raise_on_unpicklable_step_feature_group([step])

    message = str(excinfo.value)
    assert dynamic_fg.__name__ in message, f"the dynamically created class must be named; got: {message}"
