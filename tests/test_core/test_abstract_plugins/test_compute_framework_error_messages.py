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


class _MissingColumn(Exception):
    pass


class MissingColumnComputeFramework(DictComputeFramework):
    """Treats _MissingColumn as a missing-column error and raises it from transform."""

    @classmethod
    def is_missing_column_error(cls, error: Exception) -> bool:
        return isinstance(error, _MissingColumn)

    def transform(self, data: Any, feature_names: Any) -> Any:
        raise _MissingColumn("no such column")


class _ListFeatureGroup(FeatureGroup):
    @classmethod
    def calculate_feature(cls, data: Any, features: Any) -> Any:
        return [1]


class _MissingColumnInSaveFeatureGroup(FeatureGroup):
    @classmethod
    def calculate_feature(cls, data: Any, features: Any) -> Any:
        return {"a": [1]}


class TestMissingColumnAfterSave:
    def test_a_missing_column_after_a_successful_save_is_still_made_helpful(self) -> None:
        cfw = MissingColumnComputeFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        calls: list[int] = []

        with pytest.raises(ValueError, match="input_features"):
            cfw.run_calculation(
                _ListFeatureGroup, FeatureSet([Feature("a")]), location=None, after_calculate=lambda: calls.append(1)
            )
        assert calls == [1]

    def test_a_missing_column_error_raised_by_the_save_keeps_its_type(self) -> None:
        cfw = MissingColumnComputeFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        def failing_save() -> None:
            raise _MissingColumn("from saver")

        with pytest.raises(_MissingColumn, match="from saver"):
            cfw.run_calculation(
                _MissingColumnInSaveFeatureGroup,
                FeatureSet([Feature("a")]),
                location=None,
                after_calculate=failing_save,
            )


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
