import pandas as pd
import pytest
from typing import List

import mloda
from mloda import Feature
from mloda.provider import FeatureSet
from mloda import Options
from mloda.user import PluginCollector

from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.experimental.data_quality.missing_value.pandas import PandasMissingValueFeatureGroup

from tests.test_plugins.feature_group.experimental.test_missing_value_feature_group.test_missing_value_utils import (
    PandasMissingValueTestDataCreator,
    validate_missing_value_features,
)


@pytest.fixture
def sample_dataframe_with_missing() -> pd.DataFrame:
    """Create a sample pandas DataFrame with missing values for testing."""
    return pd.DataFrame(
        {
            "income": [50000, None, 75000, None, 60000],
            "age": [30, 25, None, 45, None],
            "category": ["A", None, "B", "A", None],
            "temperature": [72.5, 68.3, None, None, 70.1],
            "group": ["X", "Y", "X", "Y", "X"],
        }
    )


@pytest.fixture
def feature_set_mean() -> FeatureSet:
    """Create a feature set with a mean imputation feature."""
    feature_set = FeatureSet()
    feature_set.add(Feature("income__mean_imputed"))
    return feature_set


@pytest.fixture
def feature_set_multiple() -> FeatureSet:
    """Create a feature set with multiple imputation features."""
    feature_set = FeatureSet()
    feature_set.add(Feature("income__mean_imputed"))
    feature_set.add(Feature("age__median_imputed"))
    feature_set.add(Feature("category__mode_imputed"))
    feature_set.add(Feature("temperature__ffill_imputed"))
    return feature_set


@pytest.fixture
def feature_set_constant() -> FeatureSet:
    """Create a feature set with a constant imputation feature and options."""
    feature_set = FeatureSet()
    feature_set.add(Feature("category__constant_imputed"))

    for feature in feature_set.features:
        feature.options = Options({"constant_value": "Unknown"})

    return feature_set


@pytest.fixture
def feature_set_grouped() -> FeatureSet:
    """Create a feature set with a grouped imputation feature and options."""
    feature_set = FeatureSet()
    feature_set.add(Feature("income__mean_imputed"))

    for feature in feature_set.features:
        feature.options = Options({"group_by_features": ["group"]})

    return feature_set


class TestPandasMissingValueFeatureGroup:
    """Tests for the PandasMissingValueFeatureGroup class."""

    def test_compute_framework_rule(self) -> None:
        """Test compute_framework_rule method."""
        assert PandasMissingValueFeatureGroup.compute_framework_rule() == {PandasDataFrame}

    def test_perform_imputation_mean(self, sample_dataframe_with_missing: pd.DataFrame) -> None:
        """Test _perform_imputation method with mean imputation."""
        result = PandasMissingValueFeatureGroup._perform_imputation(sample_dataframe_with_missing, "mean", ["income"])
        # Mean of [50000, NaN, 75000, NaN, 60000] = 61666.67
        assert abs(result.iloc[1] - 61666.67) < 0.1
        assert abs(result.iloc[3] - 61666.67) < 0.1
        # Original values should be preserved
        assert result.iloc[0] == 50000
        assert result.iloc[2] == 75000
        assert result.iloc[4] == 60000

    def test_perform_imputation_median(self, sample_dataframe_with_missing: pd.DataFrame) -> None:
        """Test _perform_imputation method with median imputation."""
        result = PandasMissingValueFeatureGroup._perform_imputation(sample_dataframe_with_missing, "median", ["income"])
        # Median of [50000, NaN, 75000, NaN, 60000] = 60000
        assert result.iloc[1] == 60000
        assert result.iloc[3] == 60000
        # Original values should be preserved
        assert result.iloc[0] == 50000
        assert result.iloc[2] == 75000
        assert result.iloc[4] == 60000

    def test_perform_imputation_mode(self, sample_dataframe_with_missing: pd.DataFrame) -> None:
        """Test _perform_imputation method with mode imputation."""
        result = PandasMissingValueFeatureGroup._perform_imputation(sample_dataframe_with_missing, "mode", ["category"])
        # Mode of ["A", None, "B", "A", None] = "A"
        assert result.iloc[1] == "A"
        assert result.iloc[4] == "A"
        # Original values should be preserved
        assert result.iloc[0] == "A"
        assert result.iloc[2] == "B"
        assert result.iloc[3] == "A"

    def test_perform_imputation_constant(self, sample_dataframe_with_missing: pd.DataFrame) -> None:
        """Test _perform_imputation method with constant imputation."""
        result = PandasMissingValueFeatureGroup._perform_imputation(
            sample_dataframe_with_missing, "constant", ["category"], constant_value="Unknown"
        )
        # Constant imputation with "Unknown"
        assert result.iloc[1] == "Unknown"
        assert result.iloc[4] == "Unknown"
        # Original values should be preserved
        assert result.iloc[0] == "A"
        assert result.iloc[2] == "B"
        assert result.iloc[3] == "A"

    def test_perform_imputation_ffill(self, sample_dataframe_with_missing: pd.DataFrame) -> None:
        """Test _perform_imputation method with forward fill imputation."""
        # Sort by index to ensure consistent fill direction
        sorted_df = sample_dataframe_with_missing.sort_index()
        result = PandasMissingValueFeatureGroup._perform_imputation(sorted_df, "ffill", ["temperature"])
        # Forward fill ["A", None, "B", "A", None] -> ["A", "A", "B", "A", "A"]
        assert result.iloc[0] == 72.5
        assert result.iloc[1] == 68.3
        assert result.iloc[2] == 68.3  # Filled from previous value
        assert result.iloc[3] == 68.3  # Filled from previous value
        assert result.iloc[4] == 70.1

    def test_perform_imputation_bfill(self, sample_dataframe_with_missing: pd.DataFrame) -> None:
        """Test _perform_imputation method with backward fill imputation."""
        # Sort by index to ensure consistent fill direction
        sorted_df = sample_dataframe_with_missing.sort_index()
        result = PandasMissingValueFeatureGroup._perform_imputation(sorted_df, "bfill", ["temperature"])
        # Backward fill [72.5, 68.3, None, None, 70.1] -> [72.5, 68.3, 70.1, 70.1, 70.1]
        assert result.iloc[0] == 72.5
        assert result.iloc[1] == 68.3
        assert result.iloc[2] == 70.1  # Filled from next value
        assert result.iloc[3] == 70.1  # Filled from next value
        assert result.iloc[4] == 70.1

    def test_perform_imputation_invalid(self, sample_dataframe_with_missing: pd.DataFrame) -> None:
        """Test _perform_imputation method with invalid imputation type."""
        with pytest.raises(ValueError):
            PandasMissingValueFeatureGroup._perform_imputation(sample_dataframe_with_missing, "invalid", ["income"])

    def test_perform_grouped_imputation_mean(self, sample_dataframe_with_missing: pd.DataFrame) -> None:
        """Test _perform_grouped_imputation method with mean imputation by group."""
        result = PandasMissingValueFeatureGroup._perform_grouped_imputation(
            sample_dataframe_with_missing, "mean", "income", None, ["group"]
        )
        # Group X: [50000, 75000, 60000] -> mean = 61666.67
        # Group Y: [None, None] -> no mean, should remain None
        # But since we're imputing, we should get the overall mean for group Y
        assert abs(result.iloc[0] - 50000) < 0.1  # Original value
        assert abs(result.iloc[1] - 61666.67) < 0.1  # Imputed with overall mean
        assert abs(result.iloc[2] - 75000) < 0.1  # Original value
        assert abs(result.iloc[3] - 61666.67) < 0.1  # Imputed with overall mean
        assert abs(result.iloc[4] - 60000) < 0.1  # Original value

    def test_calculate_feature_single(
        self, sample_dataframe_with_missing: pd.DataFrame, feature_set_mean: FeatureSet
    ) -> None:
        """Test calculate_feature method with a single imputation."""
        result = PandasMissingValueFeatureGroup.calculate_feature(sample_dataframe_with_missing, feature_set_mean)

        # Check that the result contains the original data plus the imputed feature
        assert "income__mean_imputed" in result.columns
        assert abs(result["income__mean_imputed"].iloc[1] - 61666.67) < 0.1
        assert abs(result["income__mean_imputed"].iloc[3] - 61666.67) < 0.1

        # Check that the original data is preserved
        assert "income" in result.columns
        assert "age" in result.columns
        assert "category" in result.columns
        assert "temperature" in result.columns
        assert "group" in result.columns

    def test_calculate_feature_multiple(
        self, sample_dataframe_with_missing: pd.DataFrame, feature_set_multiple: FeatureSet
    ) -> None:
        """Test calculate_feature method with multiple imputations."""
        result = PandasMissingValueFeatureGroup.calculate_feature(sample_dataframe_with_missing, feature_set_multiple)

        # Check that the result contains all imputed features
        assert "income__mean_imputed" in result.columns
        assert abs(result["income__mean_imputed"].iloc[1] - 61666.67) < 0.1
        assert abs(result["income__mean_imputed"].iloc[3] - 61666.67) < 0.1

        assert "age__median_imputed" in result.columns
        assert result["age__median_imputed"].iloc[2] == 30
        assert result["age__median_imputed"].iloc[4] == 30

        assert "category__mode_imputed" in result.columns
        assert result["category__mode_imputed"].iloc[1] == "A"
        assert result["category__mode_imputed"].iloc[4] == "A"

        assert "temperature__ffill_imputed" in result.columns
        # Forward fill depends on the order, so we can't assert exact values

        # Check that the original data is preserved
        assert "income" in result.columns
        assert "age" in result.columns
        assert "category" in result.columns
        assert "temperature" in result.columns
        assert "group" in result.columns

    def test_calculate_feature_constant(
        self, sample_dataframe_with_missing: pd.DataFrame, feature_set_constant: FeatureSet
    ) -> None:
        """Test calculate_feature method with constant imputation."""

        result = PandasMissingValueFeatureGroup.calculate_feature(sample_dataframe_with_missing, feature_set_constant)

        # Check that the result contains the imputed feature
        assert "category__constant_imputed" in result.columns
        assert result["category__constant_imputed"].iloc[1] == "Unknown"
        assert result["category__constant_imputed"].iloc[4] == "Unknown"

        # Check that the original data is preserved
        assert "income" in result.columns
        assert "age" in result.columns
        assert "category" in result.columns
        assert "temperature" in result.columns
        assert "group" in result.columns

    def test_calculate_feature_grouped(
        self, sample_dataframe_with_missing: pd.DataFrame, feature_set_grouped: FeatureSet
    ) -> None:
        """Test calculate_feature method with grouped imputation."""
        result = PandasMissingValueFeatureGroup.calculate_feature(sample_dataframe_with_missing, feature_set_grouped)

        # Check that the result contains the imputed feature
        assert "income__mean_imputed" in result.columns
        # Group X: [50000, 75000, 60000] -> mean = 61666.67
        # Group Y: [None, None] -> no mean, should remain None
        # But since we're imputing, we should get the overall mean for group Y
        assert abs(result["income__mean_imputed"].iloc[0] - 50000) < 0.1  # Original value
        assert abs(result["income__mean_imputed"].iloc[1] - 61666.67) < 0.1  # Imputed with overall mean
        assert abs(result["income__mean_imputed"].iloc[2] - 75000) < 0.1  # Original value
        assert abs(result["income__mean_imputed"].iloc[3] - 61666.67) < 0.1  # Imputed with overall mean
        assert abs(result["income__mean_imputed"].iloc[4] - 60000) < 0.1  # Original value

        # Check that the original data is preserved
        assert "income" in result.columns
        assert "age" in result.columns
        assert "category" in result.columns
        assert "temperature" in result.columns
        assert "group" in result.columns

    def test_calculate_feature_missing_source(self, sample_dataframe_with_missing: pd.DataFrame) -> None:
        """Test calculate_feature method with missing source feature."""
        feature_set = FeatureSet()
        feature_set.add(Feature("missing__mean_imputed"))

        with pytest.raises(ValueError, match="Source features not found in data"):
            PandasMissingValueFeatureGroup.calculate_feature(sample_dataframe_with_missing, feature_set)

    def test_calculate_feature_constant_without_value(self, sample_dataframe_with_missing: pd.DataFrame) -> None:
        """Test calculate_feature method with constant imputation but no constant value."""
        feature_set = FeatureSet()
        feature_set.add(Feature("category__constant_imputed"))

        with pytest.raises(ValueError, match="Constant value must be provided for constant imputation method"):
            PandasMissingValueFeatureGroup.calculate_feature(sample_dataframe_with_missing, feature_set)


class TestMissingValuePandasIntegration:
    """Integration tests for the missing value feature group using DataCreator."""

    def test_imputation_with_data_creator(self) -> None:
        """Test imputation features with API using DataCreator."""

        # Enable the necessary feature groups
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PandasMissingValueTestDataCreator, PandasMissingValueFeatureGroup}
        )

        # Create options with constant value for constant imputation
        options = Options({"constant_value": "Unknown"})

        feature_str = [
            "income",  # Source data with missing values
            "age",
            "category",
            "temperature",
            "group",
            "income__mean_imputed",  # Mean imputation
            "age__median_imputed",  # Median imputation
            "category__mode_imputed",  # Mode imputation
            "category__constant_imputed",  # Constant imputation
            "temperature__ffill_imputed",  # Forward fill imputation
        ]

        feature_list: List[Feature | str] = [Feature(name=feature, options=options) for feature in feature_str]

        # Run the API with multiple imputation features
        result = mloda.run_all(
            feature_list,
            compute_frameworks={PandasDataFrame},
            plugin_collector=plugin_collector,
        )

        # Validate the missing value features
        validate_missing_value_features(result)
