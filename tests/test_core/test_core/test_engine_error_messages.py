"""Tests for improved error messages in Engine class.

These tests verify that error messages use human-readable formatting
with class names and module paths instead of raw class representations.
"""

from typing import List, Optional, Set, Type, Union, cast
from unittest.mock import patch

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.index.index import Index
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.core.engine import Engine
from mloda.user import Feature, Features, Options

from tests.test_core.test_abstract_plugins.test_abstract_compute_framework import (
    BaseTestComputeFramework1,
)


class NoIndexFeatureGroup(FeatureGroup):
    """A FeatureGroup that returns None from index_columns() to trigger the error."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        if isinstance(feature_name, FeatureName):
            return "NoIndexTestFeature" in feature_name.name
        return "NoIndexTestFeature" in feature_name

    @classmethod
    def index_columns(cls) -> Optional[List[Index]]:
        """Returns None to simulate the 'no indexes defined' error condition."""
        return None

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return None


class TestEngineErrorMessages:
    """Tests for improved error messages in the Engine class."""

    def test_no_indexes_error_contains_class_name_without_angle_brackets(self) -> None:
        """Test that 'no indexes defined' error shows ClassName (module.path) format.

        The error message should contain:
        - The class name without <class '...'> wrapper
        - The module path in parentheses
        """
        with patch(
            "mloda.core.prepare.accessible_plugins.PreFilterPlugins.resolve_feature_group_compute_framework_limitations"
        ) as mocked_plugins:
            mocked_plugins.return_value = {
                NoIndexFeatureGroup: {BaseTestComputeFramework1},
            }

            # We need to patch _add_index_feature to force calling it directly
            # since the condition on line 127 checks feature_group.index_columns()
            # which would return None and skip calling _add_index_feature

            features = Features(["NoIndexTestFeature"])
            compute_frameworks: Set[Type[ComputeFramework]] = cast(
                Set[Type[ComputeFramework]], {BaseTestComputeFramework1}
            )

            # Create engine with mocked setup
            with patch("mloda.core.core.engine.Engine.create_setup_execution_plan"):
                engine = Engine(features, compute_frameworks, None)

            # Manually test the _add_index_feature method to trigger the error
            feature_group_class = NoIndexFeatureGroup
            feature_group = NoIndexFeatureGroup()
            test_feature = Feature("NoIndexTestFeature")
            test_features = Features(["NoIndexTestFeature"])

            with pytest.raises(ValueError) as exc_info:
                engine._add_index_feature(feature_group_class, feature_group, test_feature, test_features)

            error_message = str(exc_info.value)

            # The error should NOT contain the raw class representation
            assert "<class '" not in error_message, (
                f"Error message should not contain '<class ' wrapper. Got: {error_message}"
            )

            # The error should contain the class name
            assert "NoIndexFeatureGroup" in error_message, (
                f"Error message should contain class name 'NoIndexFeatureGroup'. Got: {error_message}"
            )

            # The error should contain the module path in parentheses
            assert "test_engine_error_messages" in error_message, (
                f"Error message should contain module path. Got: {error_message}"
            )

            # The module path should be in parentheses
            assert "(" in error_message and ")" in error_message, (
                f"Error message should contain module path in parentheses. Got: {error_message}"
            )
