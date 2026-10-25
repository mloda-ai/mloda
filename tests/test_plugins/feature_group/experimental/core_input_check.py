"""Helpers that drive an operation family through the core declared-input check."""

from __future__ import annotations

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.abstract_plugins.components.validators.missing_input_validator import MissingInputFeatureError
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.provider import FeatureSet
from mloda.user import Feature


def _prepare(impl: type[Any], data: Any, feature: Feature | str) -> tuple[ComputeFramework, FeatureSet]:
    framework = impl.compute_framework(
        mode=ParallelizationMode.SYNC, children_if_root=frozenset(), function_extender=set()
    )
    framework.data = data
    framework.set_column_names()
    feature_set = FeatureSet()
    feature_set.add(feature if isinstance(feature, Feature) else Feature(feature))
    feature_set.materialize_option_defaults(impl.feature_group)
    return framework, feature_set


def assert_core_rejects_missing_input(
    impl: type[Any],
    data: Any,
    feature: Feature | str,
    missing: tuple[str, ...],
    available: tuple[str, ...] | None = None,
) -> MissingInputFeatureError:
    """Core must raise MissingInputFeatureError for `missing` before calculate_feature runs."""
    framework, feature_set = _prepare(impl, data, feature)
    with pytest.raises(MissingInputFeatureError) as exc_info:
        framework.run_validate_input_features(impl.feature_group, feature_set)
    assert feature_set.input_check == "checked"
    assert exc_info.value.missing_inputs == missing
    if available is not None:
        assert exc_info.value.available_columns == available
    return exc_info.value


def run_through_core(impl: type[Any], data: Any, feature: Feature | str) -> Any:
    """Validate declared inputs through core, then run the implementation's calculate."""
    framework, feature_set = _prepare(impl, data, feature)
    framework.run_validate_input_features(impl.feature_group, feature_set)
    assert feature_set.input_check == "checked"
    return impl.calculate(impl.feature_group, framework.data, feature_set, None)
