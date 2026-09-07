"""Tests for improved error messages in FeatureGroupStep.

These tests verify that error messages in get_api_input_data method
use the format_feature_group_class format: "ClassName (module.path)"
instead of the default repr which shows "<class 'module.ClassName'>".
"""

from unittest.mock import MagicMock

import pytest

from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.user import Feature, Options


class MockFeatureGroup(FeatureGroup):
    """A mock feature group for testing error messages."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None


class TestFeatureGroupStepErrorMessageDataNotNone:
    """Tests for error message when data is not None (line 107)."""

    def test_error_message_contains_class_name(self) -> None:
        """Error message should contain the class name without angle brackets."""
        mock_features = MagicMock(spec=FeatureSet)
        mock_features.features = set()
        mock_compute_framework = MagicMock()

        step = FeatureGroupStep(
            feature_group=MockFeatureGroup,
            features=mock_features,
            required_uuids=set(),
            compute_framework=mock_compute_framework,
            api_input_data=True,
        )

        mock_cfw_register = MagicMock()

        with pytest.raises(ValueError) as exc_info:
            step.get_api_input_data(data="some_data", cfw_register=mock_cfw_register)

        error_message = str(exc_info.value)
        assert "MockFeatureGroup" in error_message
        assert "<class" not in error_message

    def test_error_message_contains_module_path_in_parentheses(self) -> None:
        """Error message should contain the module path in parentheses.

        Expected format: "MockFeatureGroup (tests.test_core...)"
        Not the default repr: "<class 'tests.test_core...MockFeatureGroup'>"
        """
        mock_features = MagicMock(spec=FeatureSet)
        mock_features.features = set()
        mock_compute_framework = MagicMock()

        step = FeatureGroupStep(
            feature_group=MockFeatureGroup,
            features=mock_features,
            required_uuids=set(),
            compute_framework=mock_compute_framework,
            api_input_data=True,
        )

        mock_cfw_register = MagicMock()

        with pytest.raises(ValueError) as exc_info:
            step.get_api_input_data(data="some_data", cfw_register=mock_cfw_register)

        error_message = str(exc_info.value)
        assert "MockFeatureGroup (" in error_message
        assert "test_feature_group_step_error_messages)" in error_message


class TestFeatureGroupStepErrorMessageApiInputDataNotBaseApiData:
    """Tests for error message when api_input_data is not BaseApiData (line 110)."""

    def test_error_message_contains_class_name(self) -> None:
        """Error message should contain the class name without angle brackets."""
        mock_features = MagicMock(spec=FeatureSet)
        mock_features.features = set()
        mock_compute_framework = MagicMock()

        step = FeatureGroupStep(
            feature_group=MockFeatureGroup,
            features=mock_features,
            required_uuids=set(),
            compute_framework=mock_compute_framework,
            api_input_data=True,
        )

        mock_cfw_register = MagicMock()

        with pytest.raises(ValueError) as exc_info:
            step.get_api_input_data(data=None, cfw_register=mock_cfw_register)

        error_message = str(exc_info.value)
        assert "MockFeatureGroup" in error_message
        assert "<class" not in error_message

    def test_error_message_contains_module_path_in_parentheses(self) -> None:
        """Error message should contain the module path in parentheses.

        Expected format: "MockFeatureGroup (tests.test_core...)"
        Not the default repr: "<class 'tests.test_core...MockFeatureGroup'>"
        """
        mock_features = MagicMock(spec=FeatureSet)
        mock_features.features = set()
        mock_compute_framework = MagicMock()

        step = FeatureGroupStep(
            feature_group=MockFeatureGroup,
            features=mock_features,
            required_uuids=set(),
            compute_framework=mock_compute_framework,
            api_input_data=True,
        )

        mock_cfw_register = MagicMock()

        with pytest.raises(ValueError) as exc_info:
            step.get_api_input_data(data=None, cfw_register=mock_cfw_register)

        error_message = str(exc_info.value)
        assert "MockFeatureGroup (" in error_message
        assert "test_feature_group_step_error_messages)" in error_message


def test_execute_does_not_upload_finished_data_twice() -> None:
    features = MagicMock(spec=FeatureSet)
    features.features = set()
    features.artifact_to_save = None
    step = FeatureGroupStep(MockFeatureGroup, features, set(), ComputeFramework)
    step.need_to_upload = True

    cfw_register = MagicMock()
    cfw_register.get_location.return_value = "flight-location"
    cfw_register.get_runtime_artifacts.return_value = None
    cfw = MagicMock(spec=ComputeFramework)
    cfw.uuid = "framework-id"

    def run_calculation(*_: object) -> str:
        cfw.data = "object-id"
        cfw.upload_finished_data("flight-location")
        return "object-id"

    cfw.run_calculation.side_effect = run_calculation

    step.execute(cfw_register, cfw)

    cfw.upload_finished_data.assert_called_once_with("flight-location")
