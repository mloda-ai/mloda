"""Tests for improved error messages in Engine class.

These tests verify that error messages use human-readable formatting
with class names and module paths instead of raw class representations.
"""

import sqlite3
from typing import Any, cast
from unittest.mock import patch

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.index.index import Index
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.core.engine import Engine
from mloda.provider import FeatureSet, DataCreatorFeatureGroup
from mloda.user import Feature, Features, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from tests.test_core.test_abstract_plugins.test_abstract_compute_framework import (
    BaseTestComputeFramework1,
)


class NoIndexFeatureGroup(FeatureGroup):
    """A FeatureGroup that returns None from index_columns() to trigger the error."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        if isinstance(feature_name, FeatureName):
            return "NoIndexTestFeature" in str(feature_name)
        return "NoIndexTestFeature" in feature_name

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        """Returns None to simulate the 'no indexes defined' error condition."""
        return None

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
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
            compute_frameworks: set[type[ComputeFramework]] = cast(
                set[type[ComputeFramework]], {BaseTestComputeFramework1}
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


class ConnectionInInputFeaturesRoot(DataCreatorFeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"conn_in_input_root"}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"conn_in_input_root": [1]}


class ConnectionInInputFeaturesConsumer(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("conn_in_input_root", connection=sqlite3.connect(":memory:"))}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"conn_in_input_consumer": [1]}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"conn_in_input_consumer"}


def test_a_plugin_returning_a_feature_with_a_connection_raises_naming_the_plugin() -> None:
    with pytest.raises(ValueError, match="ConnectionInInputFeaturesConsumer") as raised:
        mloda.run_all(
            [Feature("conn_in_input_consumer")],
            compute_frameworks=[PyArrowTable],
            plugin_collector=PluginCollector.enabled_feature_groups(
                {ConnectionInInputFeaturesConsumer, ConnectionInInputFeaturesRoot}
            ),
        )
    assert "connection_handle" in str(raised.value)
