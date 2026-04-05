"""Tests for improved error messages in ComputeFramework."""

from typing import Any

import pytest

from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode


class DictComputeFramework(ComputeFramework):
    """Test compute framework that expects dict data."""

    @classmethod
    def expected_data_framework(cls) -> Any:
        return dict


def _make_cfw() -> DictComputeFramework:
    cfw = DictComputeFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
    return cfw


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
