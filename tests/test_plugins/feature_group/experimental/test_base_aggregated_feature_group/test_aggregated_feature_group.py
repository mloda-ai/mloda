import pandas as pd
import pytest

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import PandasAggregatedFeatureGroup

from tests.test_plugins.feature_group.experimental.test_base_aggregated_feature_group.test_aggregated_utils import (
    PandasAggregatedTestDataCreator,
    validate_aggregated_features,
)


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a sample pandas DataFrame for testing."""
    return pd.DataFrame(
        {
            "sales": [100, 200, 300, 400, 500],
            "quantity": [10, 20, 30, 40, 50],
            "price": [10.0, 9.5, 9.0, 8.5, 8.0],
            "discount": [0.1, 0.2, 0.15, 0.25, 0.1],
            "customer_rating": [4, 5, 3, 4, 5],
        }
    )


@pytest.fixture
def feature_set_sum() -> FeatureSet:
    """Create a feature set with a sum aggregation feature."""
    feature_set = FeatureSet()
    feature_set.add(Feature("sum_aggr__sales"))
    return feature_set


@pytest.fixture
def feature_set_multiple() -> FeatureSet:
    """Create a feature set with multiple aggregation features."""
    feature_set = FeatureSet()
    feature_set.add(Feature("sum_aggr__sales"))
    feature_set.add(Feature("avg_aggr__price"))
    feature_set.add(Feature("min_aggr__discount"))
    feature_set.add(Feature("max_aggr__customer_rating"))
    return feature_set


class TestAggregatedFeatureGroup:
    """Tests for the AggregatedFeatureGroup class."""

    def test_get_aggregation_type(self) -> None:
        """Test extraction of aggregation type from feature name."""
        assert AggregatedFeatureGroup.get_aggregation_type("sum_aggr__sales") == "sum"
        assert AggregatedFeatureGroup.get_aggregation_type("min_aggr__quantity") == "min"
        assert AggregatedFeatureGroup.get_aggregation_type("max_aggr__price") == "max"
        assert AggregatedFeatureGroup.get_aggregation_type("avg_aggr__discount") == "avg"

        # Test with invalid feature names
        with pytest.raises(ValueError):
            AggregatedFeatureGroup.get_aggregation_type("invalid_feature_name")

        with pytest.raises(ValueError):
            AggregatedFeatureGroup.get_aggregation_type("_aggr_sales")

        with pytest.raises(ValueError):
            AggregatedFeatureGroup.get_aggregation_type("sum_aggr_")

    def test_mloda_source_feature(self) -> None:
        """Test extraction of source feature from feature name."""
        assert AggregatedFeatureGroup.mloda_source_feature("sum_aggr__sales") == "sales"
        assert AggregatedFeatureGroup.mloda_source_feature("min_aggr__quantity") == "quantity"
        assert AggregatedFeatureGroup.mloda_source_feature("max_aggr__price") == "price"
        assert AggregatedFeatureGroup.mloda_source_feature("avg_aggr__discount") == "discount"

        # Test with invalid feature names
        with pytest.raises(ValueError):
            AggregatedFeatureGroup.mloda_source_feature("invalid_feature_name")

        with pytest.raises(ValueError):
            AggregatedFeatureGroup.mloda_source_feature("sum_aggr_")

        with pytest.raises(ValueError):
            AggregatedFeatureGroup.mloda_source_feature("_aggr_sales")

    def test_supports_aggregation_type(self) -> None:
        """Test _supports_aggregation_type method."""
        # Test with supported aggregation types
        assert AggregatedFeatureGroup._supports_aggregation_type("sum")
        assert AggregatedFeatureGroup._supports_aggregation_type("min")
        assert AggregatedFeatureGroup._supports_aggregation_type("max")
        assert AggregatedFeatureGroup._supports_aggregation_type("avg")
        assert AggregatedFeatureGroup._supports_aggregation_type("mean")
        assert AggregatedFeatureGroup._supports_aggregation_type("count")
        assert AggregatedFeatureGroup._supports_aggregation_type("std")
        assert AggregatedFeatureGroup._supports_aggregation_type("var")
        assert AggregatedFeatureGroup._supports_aggregation_type("median")

        # Test with unsupported aggregation type
        assert not AggregatedFeatureGroup._supports_aggregation_type("unsupported")

    def test_match_feature_group_criteria(self) -> None:
        """Test match_feature_group_criteria method."""
        options = Options()

        # Test with valid feature names
        assert AggregatedFeatureGroup.match_feature_group_criteria("sum_aggr__sales", options)
        assert AggregatedFeatureGroup.match_feature_group_criteria("min_aggr__quantity", options)
        assert AggregatedFeatureGroup.match_feature_group_criteria("max_aggr__price", options)
        assert AggregatedFeatureGroup.match_feature_group_criteria("avg_aggr__discount", options)

        # Test with FeatureName objects
        assert AggregatedFeatureGroup.match_feature_group_criteria(FeatureName("sum_aggr__sales"), options)
        assert AggregatedFeatureGroup.match_feature_group_criteria(FeatureName("min_aggr__quantity"), options)

        # Test with invalid feature names
        assert not AggregatedFeatureGroup.match_feature_group_criteria("invalid_feature_name", options)
        assert not AggregatedFeatureGroup.match_feature_group_criteria("sum_invalid_sales", options)
        assert not AggregatedFeatureGroup.match_feature_group_criteria("invalid_aggr_sales", options)

    def test_input_features(self) -> None:
        """Test input_features method."""
        options = Options()
        feature_group = AggregatedFeatureGroup()

        # Test with valid feature names
        input_features = feature_group.input_features(options, FeatureName("sum_aggr__sales"))
        assert input_features == {Feature("sales")}

        input_features = feature_group.input_features(options, FeatureName("min_aggr__quantity"))
        assert input_features == {Feature("quantity")}

        input_features = feature_group.input_features(options, FeatureName("max_aggr__price"))
        assert input_features == {Feature("price")}

        input_features = feature_group.input_features(options, FeatureName("avg_aggr__discount"))
        assert input_features == {Feature("discount")}


class TestPandasAggregatedFeatureGroup:
    """Tests for the PandasAggregatedFeatureGroup class."""

    def test_compute_framework_rule(self) -> None:
        """Test compute_framework_rule method."""
        assert PandasAggregatedFeatureGroup.compute_framework_rule() == {PandasDataframe}

    def test_perform_aggregation_sum(self, sample_dataframe: pd.DataFrame) -> None:
        """Test _perform_aggregation method with sum aggregation."""
        result = PandasAggregatedFeatureGroup._perform_aggregation(sample_dataframe, "sum", "sales")
        assert result == 1500  # Sum of [100, 200, 300, 400, 500]

    def test_perform_aggregation_min(self, sample_dataframe: pd.DataFrame) -> None:
        """Test _perform_aggregation method with min aggregation."""
        result = PandasAggregatedFeatureGroup._perform_aggregation(sample_dataframe, "min", "sales")
        assert result == 100  # Min of [100, 200, 300, 400, 500]

    def test_perform_aggregation_max(self, sample_dataframe: pd.DataFrame) -> None:
        """Test _perform_aggregation method with max aggregation."""
        result = PandasAggregatedFeatureGroup._perform_aggregation(sample_dataframe, "max", "sales")
        assert result == 500  # Max of [100, 200, 300, 400, 500]

    def test_perform_aggregation_avg(self, sample_dataframe: pd.DataFrame) -> None:
        """Test _perform_aggregation method with avg aggregation."""
        result = PandasAggregatedFeatureGroup._perform_aggregation(sample_dataframe, "avg", "sales")
        assert result == 300  # Avg of [100, 200, 300, 400, 500]

    def test_perform_aggregation_mean(self, sample_dataframe: pd.DataFrame) -> None:
        """Test _perform_aggregation method with mean aggregation."""
        result = PandasAggregatedFeatureGroup._perform_aggregation(sample_dataframe, "mean", "sales")
        assert result == 300  # Mean of [100, 200, 300, 400, 500]

    def test_perform_aggregation_count(self, sample_dataframe: pd.DataFrame) -> None:
        """Test _perform_aggregation method with count aggregation."""
        result = PandasAggregatedFeatureGroup._perform_aggregation(sample_dataframe, "count", "sales")
        assert result == 5  # Count of [100, 200, 300, 400, 500]

    def test_perform_aggregation_std(self, sample_dataframe: pd.DataFrame) -> None:
        """Test _perform_aggregation method with std aggregation."""
        result = PandasAggregatedFeatureGroup._perform_aggregation(sample_dataframe, "std", "sales")
        assert abs(result - 158.11) < 0.1  # Std of [100, 200, 300, 400, 500]

    def test_perform_aggregation_var(self, sample_dataframe: pd.DataFrame) -> None:
        """Test _perform_aggregation method with var aggregation."""
        result = PandasAggregatedFeatureGroup._perform_aggregation(sample_dataframe, "var", "sales")
        assert abs(result - 25000) < 0.1  # Var of [100, 200, 300, 400, 500]

    def test_perform_aggregation_median(self, sample_dataframe: pd.DataFrame) -> None:
        """Test _perform_aggregation method with median aggregation."""
        result = PandasAggregatedFeatureGroup._perform_aggregation(sample_dataframe, "median", "sales")
        assert result == 300  # Median of [100, 200, 300, 400, 500]

    def test_perform_aggregation_invalid(self, sample_dataframe: pd.DataFrame) -> None:
        """Test _perform_aggregation method with invalid aggregation type."""
        with pytest.raises(ValueError):
            PandasAggregatedFeatureGroup._perform_aggregation(sample_dataframe, "invalid", "sales")

    def test_calculate_feature_single(self, sample_dataframe: pd.DataFrame, feature_set_sum: FeatureSet) -> None:
        """Test calculate_feature method with a single aggregation."""
        result = PandasAggregatedFeatureGroup.calculate_feature(sample_dataframe, feature_set_sum)

        # Check that the result contains the original data plus the aggregated feature
        assert "sum_aggr__sales" in result.columns
        assert result["sum_aggr__sales"].iloc[0] == 1500  # Sum of [100, 200, 300, 400, 500]

        # Check that the original data is preserved
        assert "sales" in result.columns
        assert "quantity" in result.columns
        assert "price" in result.columns
        assert "discount" in result.columns
        assert "customer_rating" in result.columns

    def test_calculate_feature_multiple(self, sample_dataframe: pd.DataFrame, feature_set_multiple: FeatureSet) -> None:
        """Test calculate_feature method with multiple aggregations."""
        result = PandasAggregatedFeatureGroup.calculate_feature(sample_dataframe, feature_set_multiple)

        # Check that the result contains all aggregated features
        assert "sum_aggr__sales" in result.columns
        assert result["sum_aggr__sales"].iloc[0] == 1500  # Sum of [100, 200, 300, 400, 500]

        assert "avg_aggr__price" in result.columns
        assert result["avg_aggr__price"].iloc[0] == 9.0  # Avg of [10.0, 9.5, 9.0, 8.5, 8.0]

        assert "min_aggr__discount" in result.columns
        assert result["min_aggr__discount"].iloc[0] == 0.1  # Min of [0.1, 0.2, 0.15, 0.25, 0.1]

        assert "max_aggr__customer_rating" in result.columns
        assert result["max_aggr__customer_rating"].iloc[0] == 5  # Max of [4, 5, 3, 4, 5]

        # Check that the original data is preserved
        assert "sales" in result.columns
        assert "quantity" in result.columns
        assert "price" in result.columns
        assert "discount" in result.columns
        assert "customer_rating" in result.columns

    def test_calculate_feature_missing_source(self, sample_dataframe: pd.DataFrame) -> None:
        """Test calculate_feature method with missing source feature."""
        feature_set = FeatureSet()
        feature_set.add(Feature("sum_aggr__missing"))

        with pytest.raises(ValueError, match="Source feature 'missing' not found in data"):
            PandasAggregatedFeatureGroup.calculate_feature(sample_dataframe, feature_set)

    def test_calculate_feature_invalid_aggregation(self, sample_dataframe: pd.DataFrame) -> None:
        """Test calculate_feature method with invalid aggregation type."""
        # Temporarily modify the AGGREGATION_TYPES to simulate an invalid aggregation type
        original_types = AggregatedFeatureGroup.AGGREGATION_TYPES.copy()
        try:
            AggregatedFeatureGroup.AGGREGATION_TYPES = {"sum": "Sum of values"}

            feature_set = FeatureSet()
            feature_set.add(Feature("min_aggr__sales"))

            with pytest.raises(ValueError, match="Unsupported aggregation type: min"):
                PandasAggregatedFeatureGroup.calculate_feature(sample_dataframe, feature_set)
        finally:
            # Restore the original AGGREGATION_TYPES
            AggregatedFeatureGroup.AGGREGATION_TYPES = original_types


class TestAggPandasIntegration:
    """Integration tests for the aggregated feature group using DataCreator."""

    def test_aggregation_with_data_creator(self) -> None:
        """Test aggregation features with mlodaAPI using DataCreator."""

        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {PandasAggregatedTestDataCreator, PandasAggregatedFeatureGroup}
        )

        # Run the API with multiple aggregation features
        result = mlodaAPI.run_all(
            [
                "sales",
                "sum_aggr__sales",
                "avg_aggr__price",
                "min_aggr__discount",
                "max_aggr__customer_rating",
            ],
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        validate_aggregated_features(result)
