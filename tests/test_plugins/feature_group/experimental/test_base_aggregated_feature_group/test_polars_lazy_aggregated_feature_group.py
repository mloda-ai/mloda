from typing import Any
import pytest

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataframe
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.aggregated_feature_group.polars_lazy import (
    PolarsLazyAggregatedFeatureGroup,
)

from tests.test_plugins.feature_group.experimental.test_base_aggregated_feature_group.test_aggregated_utils import (
    AggregatedTestDataCreator,
    validate_aggregated_features,
)

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None  # type: ignore


# Test data creator for Polars Lazy
class PolarsLazyAggregatedTestDataCreator(AggregatedTestDataCreator):
    """Test data creator for Polars Lazy aggregated feature group tests."""

    compute_framework = PolarsLazyDataframe

    # Add Polars LazyFrame conversion to the conversion dictionary
    conversion = AggregatedTestDataCreator.conversion.copy()
    if POLARS_AVAILABLE:
        conversion[PolarsLazyDataframe] = lambda data: pl.LazyFrame(data)


@pytest.fixture
def sample_lazy_dataframe() -> Any:
    """Create a sample Polars LazyFrame for testing."""
    if not POLARS_AVAILABLE:
        pytest.skip("Polars not available")

    data = {
        "sales": [100, 200, 300, 400, 500],
        "quantity": [10, 20, 30, 40, 50],
        "price": [10.0, 9.5, 9.0, 8.5, 8.0],
        "discount": [0.1, 0.2, 0.15, 0.25, 0.1],
        "customer_rating": [4, 5, 3, 4, 5],
    }
    return pl.LazyFrame(data)


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


@pytest.mark.skipif(pl is None, reason="Polars not available")
class TestPolarsLazyAggregatedFeatureGroup:
    """Tests for the PolarsLazyAggregatedFeatureGroup class."""

    def test_compute_framework_rule(self) -> None:
        """Test compute_framework_rule method."""
        assert PolarsLazyAggregatedFeatureGroup.compute_framework_rule() == {PolarsLazyDataframe}

    def test_check_source_feature_exists_valid(self, sample_lazy_dataframe: Any) -> None:
        """Test _check_source_feature_exists with valid feature."""
        # Should not raise an exception
        PolarsLazyAggregatedFeatureGroup._check_source_feature_exists(sample_lazy_dataframe, "sales")
        PolarsLazyAggregatedFeatureGroup._check_source_feature_exists(sample_lazy_dataframe, "price")

    def test_check_source_feature_exists_invalid(self, sample_lazy_dataframe: Any) -> None:
        """Test _check_source_feature_exists with invalid feature."""
        with pytest.raises(ValueError, match="Source feature 'missing' not found in data"):
            PolarsLazyAggregatedFeatureGroup._check_source_feature_exists(sample_lazy_dataframe, "missing")

    def test_perform_aggregation_sum(self, sample_lazy_dataframe: Any) -> None:
        """Test _perform_aggregation method with sum aggregation."""
        result_expr = PolarsLazyAggregatedFeatureGroup._perform_aggregation(sample_lazy_dataframe, "sum", "sales")

        # The result should be a Polars expression
        assert hasattr(result_expr, "alias")  # Polars expressions have alias method

        # Test the actual aggregation by adding it to the dataframe and collecting
        result_df = sample_lazy_dataframe.with_columns(result_expr.alias("test_sum")).collect()
        assert result_df["test_sum"][0] == 1500  # Sum of [100, 200, 300, 400, 500]

    def test_perform_aggregation_min(self, sample_lazy_dataframe: Any) -> None:
        """Test _perform_aggregation method with min aggregation."""
        result_expr = PolarsLazyAggregatedFeatureGroup._perform_aggregation(sample_lazy_dataframe, "min", "sales")
        result_df = sample_lazy_dataframe.with_columns(result_expr.alias("test_min")).collect()
        assert result_df["test_min"][0] == 100  # Min of [100, 200, 300, 400, 500]

    def test_perform_aggregation_max(self, sample_lazy_dataframe: Any) -> None:
        """Test _perform_aggregation method with max aggregation."""
        result_expr = PolarsLazyAggregatedFeatureGroup._perform_aggregation(sample_lazy_dataframe, "max", "sales")
        result_df = sample_lazy_dataframe.with_columns(result_expr.alias("test_max")).collect()
        assert result_df["test_max"][0] == 500  # Max of [100, 200, 300, 400, 500]

    def test_perform_aggregation_avg(self, sample_lazy_dataframe: Any) -> None:
        """Test _perform_aggregation method with avg aggregation."""
        result_expr = PolarsLazyAggregatedFeatureGroup._perform_aggregation(sample_lazy_dataframe, "avg", "sales")
        result_df = sample_lazy_dataframe.with_columns(result_expr.alias("test_avg")).collect()
        assert result_df["test_avg"][0] == 300  # Avg of [100, 200, 300, 400, 500]

    def test_perform_aggregation_mean(self, sample_lazy_dataframe: Any) -> None:
        """Test _perform_aggregation method with mean aggregation."""
        result_expr = PolarsLazyAggregatedFeatureGroup._perform_aggregation(sample_lazy_dataframe, "mean", "sales")
        result_df = sample_lazy_dataframe.with_columns(result_expr.alias("test_mean")).collect()
        assert result_df["test_mean"][0] == 300  # Mean of [100, 200, 300, 400, 500]

    def test_perform_aggregation_count(self, sample_lazy_dataframe: Any) -> None:
        """Test _perform_aggregation method with count aggregation."""
        result_expr = PolarsLazyAggregatedFeatureGroup._perform_aggregation(sample_lazy_dataframe, "count", "sales")
        result_df = sample_lazy_dataframe.with_columns(result_expr.alias("test_count")).collect()
        assert result_df["test_count"][0] == 5  # Count of [100, 200, 300, 400, 500]

    def test_perform_aggregation_std(self, sample_lazy_dataframe: Any) -> None:
        """Test _perform_aggregation method with std aggregation."""
        result_expr = PolarsLazyAggregatedFeatureGroup._perform_aggregation(sample_lazy_dataframe, "std", "sales")
        result_df = sample_lazy_dataframe.with_columns(result_expr.alias("test_std")).collect()
        assert abs(result_df["test_std"][0] - 158.11) < 0.1  # Std of [100, 200, 300, 400, 500]

    def test_perform_aggregation_var(self, sample_lazy_dataframe: Any) -> None:
        """Test _perform_aggregation method with var aggregation."""
        result_expr = PolarsLazyAggregatedFeatureGroup._perform_aggregation(sample_lazy_dataframe, "var", "sales")
        result_df = sample_lazy_dataframe.with_columns(result_expr.alias("test_var")).collect()
        assert abs(result_df["test_var"][0] - 25000) < 0.1  # Var of [100, 200, 300, 400, 500]

    def test_perform_aggregation_median(self, sample_lazy_dataframe: Any) -> None:
        """Test _perform_aggregation method with median aggregation."""
        result_expr = PolarsLazyAggregatedFeatureGroup._perform_aggregation(sample_lazy_dataframe, "median", "sales")
        result_df = sample_lazy_dataframe.with_columns(result_expr.alias("test_median")).collect()
        assert result_df["test_median"][0] == 300  # Median of [100, 200, 300, 400, 500]

    def test_perform_aggregation_invalid(self, sample_lazy_dataframe: Any) -> None:
        """Test _perform_aggregation method with invalid aggregation type."""
        with pytest.raises(ValueError, match="Unsupported aggregation type: invalid"):
            PolarsLazyAggregatedFeatureGroup._perform_aggregation(sample_lazy_dataframe, "invalid", "sales")

    def test_calculate_feature_single(self, sample_lazy_dataframe: Any, feature_set_sum: FeatureSet) -> None:
        """Test calculate_feature method with a single aggregation."""
        result = PolarsLazyAggregatedFeatureGroup.calculate_feature(sample_lazy_dataframe, feature_set_sum)

        # The result should still be a LazyFrame
        assert hasattr(result, "collect_schema")

        # Check that the aggregated feature was added to the schema
        schema_names = set(result.collect_schema().names())
        assert "sum_aggr__sales" in schema_names

        # Collect to verify the actual values
        collected = result.collect()
        assert collected["sum_aggr__sales"][0] == 1500  # Sum of [100, 200, 300, 400, 500]

        # Check that the original data is preserved
        assert "sales" in schema_names
        assert "quantity" in schema_names
        assert "price" in schema_names
        assert "discount" in schema_names
        assert "customer_rating" in schema_names

    def test_calculate_feature_multiple(self, sample_lazy_dataframe: Any, feature_set_multiple: FeatureSet) -> None:
        """Test calculate_feature method with multiple aggregations."""
        result = PolarsLazyAggregatedFeatureGroup.calculate_feature(sample_lazy_dataframe, feature_set_multiple)

        # The result should still be a LazyFrame
        assert hasattr(result, "collect_schema")

        # Check that all aggregated features were added to the schema
        schema_names = set(result.collect_schema().names())
        assert "sum_aggr__sales" in schema_names
        assert "avg_aggr__price" in schema_names
        assert "min_aggr__discount" in schema_names
        assert "max_aggr__customer_rating" in schema_names

        # Collect to verify the actual values
        collected = result.collect()
        assert collected["sum_aggr__sales"][0] == 1500  # Sum of [100, 200, 300, 400, 500]
        assert collected["avg_aggr__price"][0] == 9.0  # Avg of [10.0, 9.5, 9.0, 8.5, 8.0]
        assert collected["min_aggr__discount"][0] == 0.1  # Min of [0.1, 0.2, 0.15, 0.25, 0.1]
        assert collected["max_aggr__customer_rating"][0] == 5  # Max of [4, 5, 3, 4, 5]

        # Check that the original data is preserved
        assert "sales" in schema_names
        assert "quantity" in schema_names
        assert "price" in schema_names
        assert "discount" in schema_names
        assert "customer_rating" in schema_names

    def test_calculate_feature_missing_source(self, sample_lazy_dataframe: Any) -> None:
        """Test calculate_feature method with missing source feature."""
        feature_set = FeatureSet()
        feature_set.add(Feature("sum_aggr__missing"))

        with pytest.raises(ValueError, match="Source feature 'missing' not found in data"):
            PolarsLazyAggregatedFeatureGroup.calculate_feature(sample_lazy_dataframe, feature_set)

    def test_calculate_feature_invalid_aggregation(self, sample_lazy_dataframe: Any) -> None:
        """Test calculate_feature method with invalid aggregation type."""
        # Temporarily modify the AGGREGATION_TYPES to simulate an invalid aggregation type
        original_types = AggregatedFeatureGroup.AGGREGATION_TYPES.copy()
        try:
            AggregatedFeatureGroup.AGGREGATION_TYPES = {"sum": "Sum of values"}

            feature_set = FeatureSet()
            feature_set.add(Feature("min_aggr__sales"))

            with pytest.raises(ValueError, match="Unsupported aggregation type: min"):
                PolarsLazyAggregatedFeatureGroup.calculate_feature(sample_lazy_dataframe, feature_set)
        finally:
            # Restore the original AGGREGATION_TYPES
            AggregatedFeatureGroup.AGGREGATION_TYPES = original_types


@pytest.mark.skipif(pl is None, reason="Polars not available")
class TestPolarsLazyAggregationIntegration:
    """Integration tests for the Polars Lazy aggregated feature group using DataCreator."""

    def test_aggregation_with_data_creator(self) -> None:
        """Test aggregation features with mlodaAPI using DataCreator."""

        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {PolarsLazyAggregatedTestDataCreator, PolarsLazyAggregatedFeatureGroup}
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
            compute_frameworks={PolarsLazyDataframe},
            plugin_collector=plugin_collector,
        )

        # Convert Polars DataFrames to pandas for validation
        pandas_result = []
        for df in result:
            pandas_result.append(df.to_pandas())

        # The result should be collected (since this is the final output)
        # and should contain all the expected features
        validate_aggregated_features(pandas_result)

    def test_lazy_evaluation_maintained(self) -> None:
        """Test that lazy evaluation is maintained throughout the process."""

        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {PolarsLazyAggregatedTestDataCreator, PolarsLazyAggregatedFeatureGroup}
        )

        # Create a larger dataset to better test lazy evaluation
        large_data = {
            "sales": list(range(1000)),
            "price": [10.0 - (i * 0.001) for i in range(1000)],
        }
        lazy_frame = pl.LazyFrame(large_data)

        # Test that we can create aggregated features without immediate execution
        feature_set = FeatureSet()
        feature_set.add(Feature("sum_aggr__sales"))
        feature_set.add(Feature("avg_aggr__price"))

        result = PolarsLazyAggregatedFeatureGroup.calculate_feature(lazy_frame, feature_set)

        # The result should still be a LazyFrame (not collected)
        assert hasattr(result, "collect_schema")
        assert hasattr(result, "collect")

        # Only when we collect should we get the actual values
        collected = result.collect()
        assert len(collected) == 1000
        assert "sum_aggr__sales" in collected.columns
        assert "avg_aggr__price" in collected.columns
