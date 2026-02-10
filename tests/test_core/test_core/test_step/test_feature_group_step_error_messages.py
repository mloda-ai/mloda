"""Tests for improved error messages in FeatureGroupStep.

These tests verify that error messages in get_api_input_data method
use the format_feature_group_class format: "ClassName (module.path)"
instead of the default repr which shows "<class 'module.ClassName'>".
"""

from typing import Optional, Set
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.user import Feature, Options


class MockFeatureGroup(FeatureGroup):
    """A mock feature group for testing error messages."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
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
