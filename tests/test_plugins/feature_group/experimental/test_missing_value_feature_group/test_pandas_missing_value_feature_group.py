import pandas as pd
import pytest
from typing import List, Set, Type, Union

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_core.api.request import mlodaAPI

from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.experimental.data_quality.missing_value.pandas import PandasMissingValueFeatureGroup


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
    feature_set.add(Feature("mean_imputed_income"))
    return feature_set


@pytest.fixture
def feature_set_multiple() -> FeatureSet:
    """Create a feature set with multiple imputation features."""
    feature_set = FeatureSet()
    feature_set.add(Feature("mean_imputed_income"))
    feature_set.add(Feature("median_imputed_age"))
    feature_set.add(Feature("mode_imputed_category"))
    feature_set.add(Feature("ffill_imputed_temperature"))
    return feature_set


@pytest.fixture
def feature_set_constant() -> FeatureSet:
    """Create a feature set with a constant imputation feature and options."""
    feature_set = FeatureSet()
    feature_set.add(Feature("constant_imputed_category"))
    feature_set.options = Options({"constant_value": "Unknown"})
    return feature_set


@pytest.fixture
def feature_set_grouped() -> FeatureSet:
    """Create a feature set with a grouped imputation feature and options."""
    feature_set = FeatureSet()
    feature_set.add(Feature("mean_imputed_income"))
    feature_set.options = Options({"group_by_features": ["group"]})
    return feature_set


class TestPandasMissingValueFeatureGroup:
    """Tests for the PandasMissingValueFeatureGroup class."""

    def test_compute_framework_rule(self) -> None:
        """Test compute_framework_rule method."""
        assert PandasMissingValueFeatureGroup.compute_framework_rule() == {PandasDataframe}

    def test_perform_imputation_mean(self, sample_dataframe_with_missing: pd.DataFrame) -> None:
        """Test _perform_imputation method with mean imputation."""
        result = PandasMissingValueFeatureGroup._perform_imputation(sample_dataframe_with_missing, "mean", "income")
        # Mean of [50000, NaN, 75000, NaN, 60000] = 61666.67
        assert abs(result.iloc[1] - 61666.67) < 0.1
        assert abs(result.iloc[3] - 61666.67) < 0.1
        # Original values should be preserved
        assert result.iloc[0] == 50000
        assert result.iloc[2] == 75000
        assert result.iloc[4] == 60000

    def test_perform_imputation_median(self, sample_dataframe_with_missing: pd.DataFrame) -> None:
        """Test _perform_imputation method with median imputation."""
        result = PandasMissingValueFeatureGroup._perform_imputation(sample_dataframe_with_missing, "median", "income")
        # Median of [50000, NaN, 75000, NaN, 60000] = 60000
        assert result.iloc[1] == 60000
        assert result.iloc[3] == 60000
        # Original values should be preserved
        assert result.iloc[0] == 50000
        assert result.iloc[2] == 75000
        assert result.iloc[4] == 60000

    def test_perform_imputation_mode(self, sample_dataframe_with_missing: pd.DataFrame) -> None:
        """Test _perform_imputation method with mode imputation."""
        result = PandasMissingValueFeatureGroup._perform_imputation(sample_dataframe_with_missing, "mode", "category")
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
            sample_dataframe_with_missing, "constant", "category", constant_value="Unknown"
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
        result = PandasMissingValueFeatureGroup._perform_imputation(sorted_df, "ffill", "temperature")
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
        result = PandasMissingValueFeatureGroup._perform_imputation(sorted_df, "bfill", "temperature")
        # Backward fill [72.5, 68.3, None, None, 70.1] -> [72.5, 68.3, 70.1, 70.1, 70.1]
        assert result.iloc[0] == 72.5
        assert result.iloc[1] == 68.3
        assert result.iloc[2] == 70.1  # Filled from next value
        assert result.iloc[3] == 70.1  # Filled from next value
        assert result.iloc[4] == 70.1

    def test_perform_imputation_invalid(self, sample_dataframe_with_missing: pd.DataFrame) -> None:
        """Test _perform_imputation method with invalid imputation type."""
        with pytest.raises(ValueError):
            PandasMissingValueFeatureGroup._perform_imputation(sample_dataframe_with_missing, "invalid", "income")

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
        assert "mean_imputed_income" in result.columns
        assert abs(result["mean_imputed_income"].iloc[1] - 61666.67) < 0.1
        assert abs(result["mean_imputed_income"].iloc[3] - 61666.67) < 0.1

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
        assert "mean_imputed_income" in result.columns
        assert abs(result["mean_imputed_income"].iloc[1] - 61666.67) < 0.1
        assert abs(result["mean_imputed_income"].iloc[3] - 61666.67) < 0.1

        assert "median_imputed_age" in result.columns
        assert result["median_imputed_age"].iloc[2] == 30
        assert result["median_imputed_age"].iloc[4] == 30

        assert "mode_imputed_category" in result.columns
        assert result["mode_imputed_category"].iloc[1] == "A"
        assert result["mode_imputed_category"].iloc[4] == "A"

        assert "ffill_imputed_temperature" in result.columns
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
        assert "constant_imputed_category" in result.columns
        assert result["constant_imputed_category"].iloc[1] == "Unknown"
        assert result["constant_imputed_category"].iloc[4] == "Unknown"

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
        assert "mean_imputed_income" in result.columns
        # Group X: [50000, 75000, 60000] -> mean = 61666.67
        # Group Y: [None, None] -> no mean, should remain None
        # But since we're imputing, we should get the overall mean for group Y
        assert abs(result["mean_imputed_income"].iloc[0] - 50000) < 0.1  # Original value
        assert abs(result["mean_imputed_income"].iloc[1] - 61666.67) < 0.1  # Imputed with overall mean
        assert abs(result["mean_imputed_income"].iloc[2] - 75000) < 0.1  # Original value
        assert abs(result["mean_imputed_income"].iloc[3] - 61666.67) < 0.1  # Imputed with overall mean
        assert abs(result["mean_imputed_income"].iloc[4] - 60000) < 0.1  # Original value

        # Check that the original data is preserved
        assert "income" in result.columns
        assert "age" in result.columns
        assert "category" in result.columns
        assert "temperature" in result.columns
        assert "group" in result.columns

    def test_calculate_feature_missing_source(self, sample_dataframe_with_missing: pd.DataFrame) -> None:
        """Test calculate_feature method with missing source feature."""
        feature_set = FeatureSet()
        feature_set.add(Feature("mean_imputed_missing"))

        with pytest.raises(ValueError, match="Source feature 'missing' not found in data"):
            PandasMissingValueFeatureGroup.calculate_feature(sample_dataframe_with_missing, feature_set)

    def test_calculate_feature_constant_without_value(self, sample_dataframe_with_missing: pd.DataFrame) -> None:
        """Test calculate_feature method with constant imputation but no constant value."""
        feature_set = FeatureSet()
        feature_set.add(Feature("constant_imputed_category"))

        with pytest.raises(ValueError, match="Constant value must be provided for constant imputation method"):
            PandasMissingValueFeatureGroup.calculate_feature(sample_dataframe_with_missing, feature_set)


class TestMissingValuePandasIntegration:
    """Integration tests for the missing value feature group using DataCreator."""

    def test_imputation_with_data_creator(self) -> None:
        """Test imputation features with mlodaAPI using DataCreator."""

        # Create a feature group that uses DataCreator to provide test data
        class TestDataCreator(AbstractFeatureGroup):
            @classmethod
            def input_data(cls) -> BaseInputData:
                return DataCreator({"income", "age", "category", "temperature", "group"})

            @classmethod
            def calculate_feature(cls, data: pd.DataFrame, features: FeatureSet) -> pd.DataFrame:
                # Return the test data
                return pd.DataFrame(
                    {
                        "income": [50000, None, 75000, None, 60000],
                        "age": [30, 25, None, 45, None],
                        "category": ["A", None, "B", "A", None],
                        "temperature": [72.5, 68.3, None, None, 70.1],
                        "group": ["X", "Y", "X", "Y", "X"],
                    }
                )

            @classmethod
            def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
                return {PandasDataframe}

        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups({TestDataCreator, PandasMissingValueFeatureGroup})

        # Create options with constant value for constant imputation
        options = Options({"constant_value": "Unknown"})

        feature_str = [
            "income",  # Source data with missing values
            "age",
            "category",
            "temperature",
            "group",
            "mean_imputed_income",  # Mean imputation
            "median_imputed_age",  # Median imputation
            "mode_imputed_category",  # Mode imputation
            "constant_imputed_category",  # Constant imputation
            "ffill_imputed_temperature",  # Forward fill imputation
        ]

        feature_list: List[Feature | str] = [Feature(name=feature, options=options) for feature in feature_str]

        # Run the API with multiple imputation features
        result = mlodaAPI.run_all(
            feature_list,
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        # Verify the results
        assert len(result) == 2  # Two DataFrames: one for source data, one for imputed features

        # Find the DataFrame with the imputed features
        imputed_df = None
        for df in result:
            if "mean_imputed_income" in df.columns:
                imputed_df = df
                break

        assert imputed_df is not None, "DataFrame with imputed features not found"

        # Verify the imputed features
        assert "mean_imputed_income" in imputed_df.columns
        assert abs(imputed_df["mean_imputed_income"].iloc[1] - 61666.67) < 0.1
        assert abs(imputed_df["mean_imputed_income"].iloc[3] - 61666.67) < 0.1

        assert "median_imputed_age" in imputed_df.columns
        assert imputed_df["median_imputed_age"].iloc[2] == 30
        assert imputed_df["median_imputed_age"].iloc[4] == 30

        assert "mode_imputed_category" in imputed_df.columns
        assert imputed_df["mode_imputed_category"].iloc[1] == "A"
        assert imputed_df["mode_imputed_category"].iloc[4] == "A"

        assert "constant_imputed_category" in imputed_df.columns
        assert imputed_df["constant_imputed_category"].iloc[1] == "Unknown"
        assert imputed_df["constant_imputed_category"].iloc[4] == "Unknown"

        assert "ffill_imputed_temperature" in imputed_df.columns
        # Forward fill depends on the order, so we can't assert exact values
