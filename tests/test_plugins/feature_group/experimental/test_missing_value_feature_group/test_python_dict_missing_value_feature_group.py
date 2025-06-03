"""
Tests for the PythonDictMissingValueFeatureGroup implementation.
"""

import pytest
import pandas as pd
from typing import Any, Dict, List

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI

from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from mloda_plugins.feature_group.experimental.data_quality.missing_value.python_dict import (
    PythonDictMissingValueFeatureGroup,
)

from tests.test_plugins.feature_group.experimental.test_missing_value_feature_group.test_missing_value_utils import (
    validate_missing_value_features,
)
from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator


class PythonDictMissingValueTestDataCreator(ATestDataCreator):
    """Test data creator for PythonDict missing value tests."""

    compute_framework = PythonDictFramework

    @classmethod
    def get_raw_data(cls) -> Dict[str, Any]:
        """Return the raw data as a dictionary."""
        return {
            "income": [50000, None, 75000, None, 60000],
            "age": [30, 25, None, 45, None],
            "category": ["A", None, "B", "A", None],
            "temperature": [72.5, 68.3, None, None, 70.1],
            "group": ["X", "Y", "X", "Y", "X"],
        }

    @classmethod
    def transform_format_for_testing(cls, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Transform the data to PythonDict format (List[Dict[str, Any]])."""
        # Convert columnar format to row-based format
        if not data:
            return []

        # Get the length from the first column
        first_key = next(iter(data.keys()))
        length = len(data[first_key])

        # Verify all columns have the same length
        for key, values in data.items():
            if len(values) != length:
                raise ValueError(
                    f"All columns must have the same length. Column '{key}' has length {len(values)}, expected {length}"
                )

        return [{key: data[key][i] for key in data.keys()} for i in range(length)]


@pytest.fixture
def sample_data_with_missing() -> List[Dict[str, Any]]:
    """Create sample PythonDict data with missing values for testing."""
    return [
        {"income": 50000, "age": 30, "category": "A", "temperature": 72.5, "group": "X"},
        {"income": None, "age": 25, "category": None, "temperature": 68.3, "group": "Y"},
        {"income": 75000, "age": None, "category": "B", "temperature": None, "group": "X"},
        {"income": None, "age": 45, "category": "A", "temperature": None, "group": "Y"},
        {"income": 60000, "age": None, "category": None, "temperature": 70.1, "group": "X"},
    ]


@pytest.fixture
def feature_set_mean() -> FeatureSet:
    """Create a feature set with a mean imputation feature."""
    feature_set = FeatureSet()
    feature_set.add(Feature("mean_imputed__income"))
    return feature_set


@pytest.fixture
def feature_set_multiple() -> FeatureSet:
    """Create a feature set with multiple imputation features."""
    feature_set = FeatureSet()
    feature_set.add(Feature("mean_imputed__income"))
    feature_set.add(Feature("median_imputed__age"))
    feature_set.add(Feature("mode_imputed__category"))
    feature_set.add(Feature("ffill_imputed__temperature"))
    return feature_set


@pytest.fixture
def feature_set_constant() -> FeatureSet:
    """Create a feature set with a constant imputation feature and options."""
    feature_set = FeatureSet()
    feature_set.add(Feature("constant_imputed__category"))
    feature_set.options = Options({"constant_value": "Unknown"})
    return feature_set


@pytest.fixture
def feature_set_grouped() -> FeatureSet:
    """Create a feature set with a grouped imputation feature and options."""
    feature_set = FeatureSet()
    feature_set.add(Feature("mean_imputed__income"))
    feature_set.options = Options({"group_by_features": ["group"]})
    return feature_set


class TestPythonDictMissingValueFeatureGroup:
    """Tests for the PythonDictMissingValueFeatureGroup class."""

    def test_compute_framework_rule(self) -> None:
        """Test compute_framework_rule method."""
        assert PythonDictMissingValueFeatureGroup.compute_framework_rule() == {PythonDictFramework}

    def test_check_source_feature_exists(self, sample_data_with_missing: List[Dict[str, Any]]) -> None:
        """Test _check_source_feature_exists method."""
        # Feature exists
        PythonDictMissingValueFeatureGroup._check_source_feature_exists(sample_data_with_missing, "income")

        # Feature doesn't exist
        with pytest.raises(ValueError, match="Source feature 'nonexistent' not found in data"):
            PythonDictMissingValueFeatureGroup._check_source_feature_exists(sample_data_with_missing, "nonexistent")

        # Empty data
        with pytest.raises(ValueError, match="Data cannot be empty"):
            PythonDictMissingValueFeatureGroup._check_source_feature_exists([], "income")

    def test_add_result_to_data(self, sample_data_with_missing: List[Dict[str, Any]]) -> None:
        """Test _add_result_to_data method."""
        result = [1, 2, 3, 4, 5]
        data_copy = [row.copy() for row in sample_data_with_missing]

        updated_data = PythonDictMissingValueFeatureGroup._add_result_to_data(data_copy, "test_feature", result)

        assert len(updated_data) == len(sample_data_with_missing)
        for i, row in enumerate(updated_data):
            assert row["test_feature"] == result[i]

        # Test mismatched lengths
        with pytest.raises(ValueError, match="Result length 3 does not match data length 5"):
            PythonDictMissingValueFeatureGroup._add_result_to_data(data_copy, "test", [1, 2, 3])

    def test_impute_mean(self, sample_data_with_missing: List[Dict[str, Any]]) -> None:
        """Test _impute_mean method."""
        values = [50000, None, 75000, None, 60000]
        result = PythonDictMissingValueFeatureGroup._impute_mean(values)

        # Mean of [50000, 75000, 60000] = 61666.67
        expected_mean = 61666.67
        assert abs(result[1] - expected_mean) < 0.1
        assert abs(result[3] - expected_mean) < 0.1
        # Original values should be preserved
        assert result[0] == 50000
        assert result[2] == 75000
        assert result[4] == 60000

    def test_impute_median(self, sample_data_with_missing: List[Dict[str, Any]]) -> None:
        """Test _impute_median method."""
        values = [50000, None, 75000, None, 60000]
        result = PythonDictMissingValueFeatureGroup._impute_median(values)

        # Median of [50000, 75000, 60000] = 60000
        assert result[1] == 60000
        assert result[3] == 60000
        # Original values should be preserved
        assert result[0] == 50000
        assert result[2] == 75000
        assert result[4] == 60000

    def test_impute_mode(self, sample_data_with_missing: List[Dict[str, Any]]) -> None:
        """Test _impute_mode method."""
        values = ["A", None, "B", "A", None]
        result = PythonDictMissingValueFeatureGroup._impute_mode(values)

        # Mode of ["A", "B", "A"] = "A"
        assert result[1] == "A"
        assert result[4] == "A"
        # Original values should be preserved
        assert result[0] == "A"
        assert result[2] == "B"
        assert result[3] == "A"

    def test_impute_constant(self, sample_data_with_missing: List[Dict[str, Any]]) -> None:
        """Test _impute_constant method."""
        values = ["A", None, "B", None, "C"]
        result = PythonDictMissingValueFeatureGroup._impute_constant(values, "Unknown")

        assert result[1] == "Unknown"
        assert result[3] == "Unknown"
        # Original values should be preserved
        assert result[0] == "A"
        assert result[2] == "B"
        assert result[4] == "C"

    def test_impute_ffill(self, sample_data_with_missing: List[Dict[str, Any]]) -> None:
        """Test _impute_ffill method."""
        values = [72.5, 68.3, None, None, 70.1]
        result = PythonDictMissingValueFeatureGroup._impute_ffill(values)

        assert result[0] == 72.5
        assert result[1] == 68.3
        assert result[2] == 68.3  # Forward filled
        assert result[3] == 68.3  # Forward filled
        assert result[4] == 70.1

    def test_impute_bfill(self, sample_data_with_missing: List[Dict[str, Any]]) -> None:
        """Test _impute_bfill method."""
        values = [72.5, 68.3, None, None, 70.1]
        result = PythonDictMissingValueFeatureGroup._impute_bfill(values)

        assert result[0] == 72.5
        assert result[1] == 68.3
        assert result[2] == 70.1  # Backward filled
        assert result[3] == 70.1  # Backward filled
        assert result[4] == 70.1

    def test_perform_imputation_mean(self, sample_data_with_missing: List[Dict[str, Any]]) -> None:
        """Test _perform_imputation method with mean imputation."""
        result = PythonDictMissingValueFeatureGroup._perform_imputation(sample_data_with_missing, "mean", "income")

        # Mean of [50000, 75000, 60000] = 61666.67
        expected_mean = 61666.67
        assert abs(result[1] - expected_mean) < 0.1
        assert abs(result[3] - expected_mean) < 0.1
        # Original values should be preserved
        assert result[0] == 50000
        assert result[2] == 75000
        assert result[4] == 60000

    def test_perform_imputation_invalid(self, sample_data_with_missing: List[Dict[str, Any]]) -> None:
        """Test _perform_imputation method with invalid imputation type."""
        with pytest.raises(ValueError, match="Unsupported imputation method: invalid"):
            PythonDictMissingValueFeatureGroup._perform_imputation(sample_data_with_missing, "invalid", "income")

    def test_perform_grouped_imputation_mean(self, sample_data_with_missing: List[Dict[str, Any]]) -> None:
        """Test _perform_grouped_imputation method with mean imputation by group."""
        result = PythonDictMissingValueFeatureGroup._perform_grouped_imputation(
            sample_data_with_missing, "mean", "income", None, ["group"]
        )

        # Group X: [50000, 75000, 60000] -> mean = 61666.67
        # Group Y: [None, None] -> no mean, should get overall mean
        overall_mean = 61666.67
        assert abs(result[0] - 50000) < 0.1  # Original value
        assert abs(result[1] - overall_mean) < 0.1  # Imputed with overall mean
        assert abs(result[2] - 75000) < 0.1  # Original value
        assert abs(result[3] - overall_mean) < 0.1  # Imputed with overall mean
        assert abs(result[4] - 60000) < 0.1  # Original value

    def test_calculate_feature_single(
        self, sample_data_with_missing: List[Dict[str, Any]], feature_set_mean: FeatureSet
    ) -> None:
        """Test calculate_feature method with a single imputation."""
        data_copy = [row.copy() for row in sample_data_with_missing]
        result = PythonDictMissingValueFeatureGroup.calculate_feature(data_copy, feature_set_mean)

        # Check that the result contains the imputed feature
        assert "mean_imputed__income" in result[0]
        expected_mean = 61666.67
        assert abs(result[1]["mean_imputed__income"] - expected_mean) < 0.1
        assert abs(result[3]["mean_imputed__income"] - expected_mean) < 0.1

        # Check that the original data is preserved
        assert "income" in result[0]
        assert "age" in result[0]
        assert "category" in result[0]

    def test_calculate_feature_multiple(
        self, sample_data_with_missing: List[Dict[str, Any]], feature_set_multiple: FeatureSet
    ) -> None:
        """Test calculate_feature method with multiple imputations."""
        data_copy = [row.copy() for row in sample_data_with_missing]
        result = PythonDictMissingValueFeatureGroup.calculate_feature(data_copy, feature_set_multiple)

        # Check that all imputed features are present
        assert "mean_imputed__income" in result[0]
        assert "median_imputed__age" in result[0]
        assert "mode_imputed__category" in result[0]
        assert "ffill_imputed__temperature" in result[0]

        # Check some specific values
        expected_mean = 61666.67
        assert abs(result[1]["mean_imputed__income"] - expected_mean) < 0.1

        # Median age of [30, 25, 45] = 30
        assert result[2]["median_imputed__age"] == 30
        assert result[4]["median_imputed__age"] == 30

        # Mode category of ["A", "B", "A"] = "A"
        assert result[1]["mode_imputed__category"] == "A"
        assert result[4]["mode_imputed__category"] == "A"

    def test_calculate_feature_constant(
        self, sample_data_with_missing: List[Dict[str, Any]], feature_set_constant: FeatureSet
    ) -> None:
        """Test calculate_feature method with constant imputation."""
        data_copy = [row.copy() for row in sample_data_with_missing]
        result = PythonDictMissingValueFeatureGroup.calculate_feature(data_copy, feature_set_constant)

        # Check that the result contains the imputed feature
        assert "constant_imputed__category" in result[0]
        assert result[1]["constant_imputed__category"] == "Unknown"
        assert result[4]["constant_imputed__category"] == "Unknown"

    def test_calculate_feature_grouped(
        self, sample_data_with_missing: List[Dict[str, Any]], feature_set_grouped: FeatureSet
    ) -> None:
        """Test calculate_feature method with grouped imputation."""
        data_copy = [row.copy() for row in sample_data_with_missing]
        result = PythonDictMissingValueFeatureGroup.calculate_feature(data_copy, feature_set_grouped)

        # Check that the result contains the imputed feature
        assert "mean_imputed__income" in result[0]

        # Group X: [50000, 75000, 60000] -> mean = 61666.67
        # Group Y: [None, None] -> no mean, should get overall mean
        overall_mean = 61666.67
        assert abs(result[0]["mean_imputed__income"] - 50000) < 0.1  # Original value
        assert abs(result[1]["mean_imputed__income"] - overall_mean) < 0.1  # Imputed
        assert abs(result[2]["mean_imputed__income"] - 75000) < 0.1  # Original value
        assert abs(result[3]["mean_imputed__income"] - overall_mean) < 0.1  # Imputed
        assert abs(result[4]["mean_imputed__income"] - 60000) < 0.1  # Original value

    def test_calculate_feature_missing_source(self, sample_data_with_missing: List[Dict[str, Any]]) -> None:
        """Test calculate_feature method with missing source feature."""
        feature_set = FeatureSet()
        feature_set.add(Feature("mean_imputed__missing"))

        data_copy = [row.copy() for row in sample_data_with_missing]
        with pytest.raises(ValueError, match="Source feature 'missing' not found in data"):
            PythonDictMissingValueFeatureGroup.calculate_feature(data_copy, feature_set)

    def test_calculate_feature_constant_without_value(self, sample_data_with_missing: List[Dict[str, Any]]) -> None:
        """Test calculate_feature method with constant imputation but no constant value."""
        feature_set = FeatureSet()
        feature_set.add(Feature("constant_imputed__category"))

        data_copy = [row.copy() for row in sample_data_with_missing]
        with pytest.raises(ValueError, match="Constant value must be provided for constant imputation method"):
            PythonDictMissingValueFeatureGroup.calculate_feature(data_copy, feature_set)


class TestMissingValuePythonDictIntegration:
    """Integration tests for the missing value feature group using PythonDict framework."""

    def test_imputation_with_data_creator(self) -> None:
        """Test imputation features with mlodaAPI using DataCreator."""

        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {PythonDictMissingValueTestDataCreator, PythonDictMissingValueFeatureGroup}
        )

        # Create options with constant value for constant imputation
        options = Options({"constant_value": "Unknown"})

        feature_str = [
            "income",  # Source data with missing values
            "age",
            "category",
            "temperature",
            "group",
            "mean_imputed__income",  # Mean imputation
            "median_imputed__age",  # Median imputation
            "mode_imputed__category",  # Mode imputation
            "constant_imputed__category",  # Constant imputation
            "ffill_imputed__temperature",  # Forward fill imputation
        ]

        feature_list = [Feature(name=feature, options=options) for feature in feature_str]

        # Run the API with multiple imputation features
        result = mlodaAPI.run_all(
            feature_list,  # type: ignore
            compute_frameworks={PythonDictFramework},
            plugin_collector=plugin_collector,
        )

        # Transform PythonDict result to pandas DataFrame for validation
        # Result is a list containing the PythonDict data (List[Dict[str, Any]])
        assert len(result) == 2

        # Convert PythonDict format to pandas DataFrame
        df = pd.DataFrame(result[0])
        df2 = pd.DataFrame(result[1])

        # Use the existing validation function
        validate_missing_value_features([df, df2])
