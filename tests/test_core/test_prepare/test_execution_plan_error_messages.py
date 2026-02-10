"""Tests for error message formatting in ExecutionPlan.

These tests verify that error messages in execution_plan.py use the
format_feature_group_class function to produce readable class names
like "ClassName (module.path)" instead of raw "<class '...'>" format.

Target errors:
1. Line 230: "Feature group {ep.feature_group} has no uuid."
2. Line 867: "Feature group {feature_group} has no feature set name."
3. Line 874: "Feature group {feature_group} has no matching api data class for feature."
"""

from typing import Optional, Set, Type, Union
from unittest.mock import MagicMock, patch

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.input_data.api.api_input_data import ApiInputData
from mloda.core.abstract_plugins.components.input_data.api.api_input_data_collection import (
    ApiInputDataCollection,
)
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.execution_plan import ExecutionPlan


class MockComputeFramework(ComputeFramework):
    """Mock compute framework for testing."""

    pass


class ApiInputDataFeatureGroupFixture(FeatureGroup):
    """A feature group fixture that uses ApiInputData for testing error messages."""

    @classmethod
    def input_data(cls) -> Optional[ApiInputData]:
        return ApiInputData()

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        if isinstance(feature_name, FeatureName):
            feature_name = feature_name.name
        return feature_name == "test_api_feature"

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return None


class NoUuidFeatureGroup(FeatureGroup):
    """A test feature group for testing 'no uuid' error message."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        if isinstance(feature_name, FeatureName):
            feature_name = feature_name.name
        return feature_name == "no_uuid_test_feature"

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return None


class TestNoFeatureSetNameErrorMessage:
    """Tests for the 'no feature set name' error message format (Line 867).

    This error is triggered in prepare_api_input_data when
    feature_set.get_name_of_one_feature().name is None.

    We mock get_name_of_one_feature to return a FeatureName with None
    to bypass the validator and trigger the actual error.
    """

    def test_error_message_contains_class_name(self) -> None:
        """Test that the 'no feature set name' error includes the class name.

        The error message should contain the class name 'ApiInputDataFeatureGroupFixture'.
        """
        execution_plan = ExecutionPlan(
            api_input_data_collection=ApiInputDataCollection(),
        )

        feature_set = FeatureSet()
        feature = Feature("test_api_feature")
        feature_set.add(feature)

        mock_feature_name = MagicMock(spec=FeatureName)
        mock_feature_name.name = None

        with patch.object(feature_set, "get_name_of_one_feature", return_value=mock_feature_name):
            with pytest.raises(ValueError) as exc_info:
                execution_plan.prepare_api_input_data(ApiInputDataFeatureGroupFixture, feature_set)

        error_message = str(exc_info.value)

        assert "no feature set name" in error_message.lower(), (
            f"Error should be about 'no feature set name', but got: {error_message}"
        )
        assert "ApiInputDataFeatureGroupFixture" in error_message, (
            f"Error message should contain class name 'ApiInputDataFeatureGroupFixture', but got: {error_message}"
        )

    def test_error_message_contains_module_path(self) -> None:
        """Test that the 'no feature set name' error includes module path.

        The error message should contain the module path in parentheses like:
          ApiInputDataFeatureGroupFixture (tests.test_core.test_prepare.test_execution_plan_error_messages)

        """
        execution_plan = ExecutionPlan(
            api_input_data_collection=ApiInputDataCollection(),
        )

        feature_set = FeatureSet()
        feature = Feature("test_api_feature")
        feature_set.add(feature)

        mock_feature_name = MagicMock(spec=FeatureName)
        mock_feature_name.name = None

        with patch.object(feature_set, "get_name_of_one_feature", return_value=mock_feature_name):
            with pytest.raises(ValueError) as exc_info:
                execution_plan.prepare_api_input_data(ApiInputDataFeatureGroupFixture, feature_set)

        error_message = str(exc_info.value)

        assert "no feature set name" in error_message.lower(), (
            f"Error should be about 'no feature set name', but got: {error_message}"
        )
        assert "(tests.test_core.test_prepare.test_execution_plan_error_messages)" in error_message, (
            f"Error message should contain module path in parentheses, but got: {error_message}"
        )

    def test_error_message_does_not_contain_raw_class_representation(self) -> None:
        """Test that the error message does NOT contain raw class representation.

        The raw representation looks like:
            Feature group <class '...'> has no feature set name.

        This raw '<class' representation should NOT appear in the formatted message.
        """
        execution_plan = ExecutionPlan(
            api_input_data_collection=ApiInputDataCollection(),
        )

        feature_set = FeatureSet()
        feature = Feature("test_api_feature")
        feature_set.add(feature)

        mock_feature_name = MagicMock(spec=FeatureName)
        mock_feature_name.name = None

        with patch.object(feature_set, "get_name_of_one_feature", return_value=mock_feature_name):
            with pytest.raises(ValueError) as exc_info:
                execution_plan.prepare_api_input_data(ApiInputDataFeatureGroupFixture, feature_set)

        error_message = str(exc_info.value)

        assert "no feature set name" in error_message.lower(), (
            f"Error should be about 'no feature set name', but got: {error_message}"
        )
        assert "<class" not in error_message, (
            f"Error message should NOT contain raw class representation '<class', but got: {error_message}"
        )


class TestNoMatchingApiDataClassErrorMessage:
    """Tests for the 'no matching api data class' error message format (Line 874)."""

    def test_error_message_contains_class_name(self) -> None:
        """Test that the 'no matching api data class' error includes the class name.

        The error message should contain the class name 'ApiInputDataFeatureGroupFixture'.
        """
        api_collection = ApiInputDataCollection()

        execution_plan = ExecutionPlan(
            api_input_data_collection=api_collection,
        )

        feature_set = FeatureSet()
        feature = Feature("unmatched_feature")
        feature_set.add(feature)

        with patch.object(api_collection, "get_name_cls_by_matching_column_name", return_value=("test", None)):
            with pytest.raises(ValueError) as exc_info:
                execution_plan.prepare_api_input_data(ApiInputDataFeatureGroupFixture, feature_set)

        error_message = str(exc_info.value)

        assert "no matching api data class" in error_message.lower(), (
            f"Error should be about 'no matching api data class', but got: {error_message}"
        )
        assert "ApiInputDataFeatureGroupFixture" in error_message, (
            f"Error message should contain class name 'ApiInputDataFeatureGroupFixture', but got: {error_message}"
        )

    def test_error_message_contains_module_path(self) -> None:
        """Test that the 'no matching api data class' error includes module path.

        The error message should contain the module path in parentheses like:
          ApiInputDataFeatureGroupFixture (tests.test_core.test_prepare.test_execution_plan_error_messages)

        """
        api_collection = ApiInputDataCollection()

        execution_plan = ExecutionPlan(
            api_input_data_collection=api_collection,
        )

        feature_set = FeatureSet()
        feature = Feature("unmatched_feature")
        feature_set.add(feature)

        with patch.object(api_collection, "get_name_cls_by_matching_column_name", return_value=("test", None)):
            with pytest.raises(ValueError) as exc_info:
                execution_plan.prepare_api_input_data(ApiInputDataFeatureGroupFixture, feature_set)

        error_message = str(exc_info.value)

        assert "no matching api data class" in error_message.lower(), (
            f"Error should be about 'no matching api data class', but got: {error_message}"
        )
        assert "(tests.test_core.test_prepare.test_execution_plan_error_messages)" in error_message, (
            f"Error message should contain module path in parentheses, but got: {error_message}"
        )

    def test_error_message_does_not_contain_raw_class_representation(self) -> None:
        """Test that the error message does NOT contain raw class representation.

        The raw representation looks like:
            Feature group <class '...'> has no matching api data class for feature.

        This raw '<class' representation should NOT appear in the formatted message.
        """
        api_collection = ApiInputDataCollection()

        execution_plan = ExecutionPlan(
            api_input_data_collection=api_collection,
        )

        feature_set = FeatureSet()
        feature = Feature("unmatched_feature")
        feature_set.add(feature)

        with patch.object(api_collection, "get_name_cls_by_matching_column_name", return_value=("test", None)):
            with pytest.raises(ValueError) as exc_info:
                execution_plan.prepare_api_input_data(ApiInputDataFeatureGroupFixture, feature_set)

        error_message = str(exc_info.value)

        assert "no matching api data class" in error_message.lower(), (
            f"Error should be about 'no matching api data class', but got: {error_message}"
        )
        assert "<class" not in error_message, (
            f"Error message should NOT contain raw class representation '<class', but got: {error_message}"
        )


class TestNoUuidErrorMessage:
    """Tests for the 'no uuid' error message format (Line 230).

    This error is triggered in add_tfs() when ep.features.any_uuid is None.
    This is a more complex error to trigger as it requires setting up
    the execution plan with a FeatureGroupStep that has no uuid.
    """

    def test_error_message_contains_class_name(self) -> None:
        """Test that the 'no uuid' error includes the class name.

        The error message should contain the class name 'NoUuidFeatureGroup'.
        """
        from mloda.core.core.step.feature_group_step import FeatureGroupStep
        from mloda.core.prepare.graph.graph import Graph

        execution_plan = ExecutionPlan()

        feature_set = FeatureSet()
        feature = Feature("no_uuid_test_feature")
        feature_set.add(feature)
        feature_set.any_uuid = None

        fg_step = FeatureGroupStep(
            feature_group=NoUuidFeatureGroup,
            features=feature_set,
            required_uuids=set(),
            compute_framework=MockComputeFramework,
        )
        fg_step.features.any_uuid = None

        graph = MagicMock(spec=Graph)
        graph.parent_to_children_mapping = {}

        with pytest.raises(ValueError) as exc_info:
            execution_plan.add_tfs([fg_step], graph)

        error_message = str(exc_info.value)

        assert "no uuid" in error_message.lower(), f"Error should be about 'no uuid', but got: {error_message}"
        assert "NoUuidFeatureGroup" in error_message, (
            f"Error message should contain class name 'NoUuidFeatureGroup', but got: {error_message}"
        )

    def test_error_message_contains_module_path(self) -> None:
        """Test that the 'no uuid' error includes module path.

        The error message should contain the module path in parentheses like:
          NoUuidFeatureGroup (tests.test_core.test_prepare.test_execution_plan_error_messages)

        """
        from mloda.core.core.step.feature_group_step import FeatureGroupStep
        from mloda.core.prepare.graph.graph import Graph

        execution_plan = ExecutionPlan()

        feature_set = FeatureSet()
        feature = Feature("no_uuid_test_feature")
        feature_set.add(feature)
        feature_set.any_uuid = None

        fg_step = FeatureGroupStep(
            feature_group=NoUuidFeatureGroup,
            features=feature_set,
            required_uuids=set(),
            compute_framework=MockComputeFramework,
        )
        fg_step.features.any_uuid = None

        graph = MagicMock(spec=Graph)
        graph.parent_to_children_mapping = {}

        with pytest.raises(ValueError) as exc_info:
            execution_plan.add_tfs([fg_step], graph)

        error_message = str(exc_info.value)

        assert "no uuid" in error_message.lower(), f"Error should be about 'no uuid', but got: {error_message}"
        assert "(tests.test_core.test_prepare.test_execution_plan_error_messages)" in error_message, (
            f"Error message should contain module path in parentheses, but got: {error_message}"
        )

    def test_error_message_does_not_contain_raw_class_representation(self) -> None:
        """Test that the error message does NOT contain raw class representation.

        The raw representation looks like:
            Feature group <class '...'> has no uuid.

        This raw '<class' representation should NOT appear in the formatted message.
        """
        from mloda.core.core.step.feature_group_step import FeatureGroupStep
        from mloda.core.prepare.graph.graph import Graph

        execution_plan = ExecutionPlan()

        feature_set = FeatureSet()
        feature = Feature("no_uuid_test_feature")
        feature_set.add(feature)
        feature_set.any_uuid = None

        fg_step = FeatureGroupStep(
            feature_group=NoUuidFeatureGroup,
            features=feature_set,
            required_uuids=set(),
            compute_framework=MockComputeFramework,
        )
        fg_step.features.any_uuid = None

        graph = MagicMock(spec=Graph)
        graph.parent_to_children_mapping = {}

        with pytest.raises(ValueError) as exc_info:
            execution_plan.add_tfs([fg_step], graph)

        error_message = str(exc_info.value)

        assert "no uuid" in error_message.lower(), f"Error should be about 'no uuid', but got: {error_message}"
        assert "<class" not in error_message, (
            f"Error message should NOT contain raw class representation '<class', but got: {error_message}"
        )
