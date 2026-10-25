"""Tests for improved error messages in ComputeFramework."""

from typing import Any

import pytest

from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode


class DictComputeFramework(ComputeFramework):
    """Test compute framework that expects dict data."""

    @classmethod
    def expected_data_framework(cls) -> Any:
        return dict


def _make_cfw() -> DictComputeFramework:
    cfw = DictComputeFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
    return cfw


class _UnrelatedError(Exception):
    pass


_UNRAISED = _UnrelatedError("not a missing column")


class _RaisingFeatureGroup(FeatureGroup):
    @classmethod
    def calculate_feature(cls, data: Any, features: Any) -> Any:
        raise _UNRAISED


class TestIsMissingColumnError:
    def test_default_is_false(self) -> None:
        assert ComputeFramework.is_missing_column_error(KeyError("x")) is False
        assert DictComputeFramework.is_missing_column_error(_UnrelatedError("x")) is False

    def test_non_matching_error_from_calculate_feature_passes_through_unchanged(self) -> None:
        cfw = _make_cfw()
        with pytest.raises(_UnrelatedError) as excinfo:
            cfw.run_calculation(_RaisingFeatureGroup, FeatureSet([Feature("a")]), location=None)
        assert excinfo.value is _UNRAISED


class TestValidateExpectedFrameworkErrorMessage:
    """Tests for validate_expected_framework error message clarity."""

    def test_error_contains_actual_type(self) -> None:
        cfw = _make_cfw()
        cfw.data = [1, 2, 3]

        with pytest.raises(ValueError, match="list"):
            cfw.validate_expected_framework()

    def test_error_contains_expected_type(self) -> None:
        cfw = _make_cfw()
        cfw.data = [1, 2, 3]

        with pytest.raises(ValueError, match="Expected type: dict"):
            cfw.validate_expected_framework()

    def test_error_contains_framework_name(self) -> None:
        cfw = _make_cfw()
        cfw.data = [1, 2, 3]

        with pytest.raises(ValueError, match="DictComputeFramework"):
            cfw.validate_expected_framework()

    def test_error_contains_guidance(self) -> None:
        cfw = _make_cfw()
        cfw.data = [1, 2, 3]

        with pytest.raises(ValueError, match="calculate_feature"):
            cfw.validate_expected_framework()
