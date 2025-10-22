import pyarrow as pa
import pandas as pd
import pytest
from typing import List

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyarrowTable
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pyarrow import PyArrowAggregatedFeatureGroup

from tests.test_plugins.feature_group.experimental.test_base_aggregated_feature_group.test_aggregated_utils import (
    PyArrowAggregatedTestDataCreator,
    validate_aggregated_features,
)


@pytest.fixture
def sample_table() -> pa.Table:
    """Create a sample PyArrow Table for testing."""
    return pa.table(
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


class TestPyArrowAggregatedFeatureGroup:
    """Tests for the PyArrowAggregatedFeatureGroup class."""

    def test_compute_framework_rule(self) -> None:
        """Test compute_framework_rule method."""
        assert PyArrowAggregatedFeatureGroup.compute_framework_rule() == {PyarrowTable}

    def test_perform_aggregation_sum(self, sample_table: pa.Table) -> None:
        """Test _perform_aggregation method with sum aggregation."""
        result = PyArrowAggregatedFeatureGroup._perform_aggregation(sample_table, "sum", ["sales"])
        assert result == 1500  # Sum of [100, 200, 300, 400, 500]

    def test_perform_aggregation_min(self, sample_table: pa.Table) -> None:
        """Test _perform_aggregation method with min aggregation."""
        result = PyArrowAggregatedFeatureGroup._perform_aggregation(sample_table, "min", ["sales"])
        assert result == 100  # Min of [100, 200, 300, 400, 500]

    def test_perform_aggregation_max(self, sample_table: pa.Table) -> None:
        """Test _perform_aggregation method with max aggregation."""
        result = PyArrowAggregatedFeatureGroup._perform_aggregation(sample_table, "max", ["sales"])
        assert result == 500  # Max of [100, 200, 300, 400, 500]

    def test_perform_aggregation_avg(self, sample_table: pa.Table) -> None:
        """Test _perform_aggregation method with avg aggregation."""
        result = PyArrowAggregatedFeatureGroup._perform_aggregation(sample_table, "avg", ["sales"])
        assert result == 300  # Avg of [100, 200, 300, 400, 500]

    def test_perform_aggregation_mean(self, sample_table: pa.Table) -> None:
        """Test _perform_aggregation method with mean aggregation."""
        result = PyArrowAggregatedFeatureGroup._perform_aggregation(sample_table, "mean", ["sales"])
        assert result == 300  # Mean of [100, 200, 300, 400, 500]

    def test_perform_aggregation_count(self, sample_table: pa.Table) -> None:
        """Test _perform_aggregation method with count aggregation."""
        result = PyArrowAggregatedFeatureGroup._perform_aggregation(sample_table, "count", ["sales"])
        assert result == 5  # Count of [100, 200, 300, 400, 500]

    def test_perform_aggregation_std(self, sample_table: pa.Table) -> None:
        """Test _perform_aggregation method with std aggregation."""
        result = PyArrowAggregatedFeatureGroup._perform_aggregation(sample_table, "std", ["sales"])
        # PyArrow uses a different formula for standard deviation than Pandas
        # PyArrow uses the population standard deviation (n), while Pandas uses the sample standard deviation (n-1)
        assert abs(result - 141.42) < 0.1  # Std of [100, 200, 300, 400, 500] with population formula

    def test_perform_aggregation_var(self, sample_table: pa.Table) -> None:
        """Test _perform_aggregation method with var aggregation."""
        result = PyArrowAggregatedFeatureGroup._perform_aggregation(sample_table, "var", ["sales"])
        # PyArrow uses a different formula for variance than Pandas
        # PyArrow uses the population variance (n), while Pandas uses the sample variance (n-1)
        assert abs(result - 20000) < 0.1  # Var of [100, 200, 300, 400, 500] with population formula

    def test_perform_aggregation_median(self, sample_table: pa.Table) -> None:
        """Test _perform_aggregation method with median aggregation."""
        result = PyArrowAggregatedFeatureGroup._perform_aggregation(sample_table, "median", ["sales"])
        assert result == 300  # Median of [100, 200, 300, 400, 500]

    def test_perform_aggregation_invalid(self, sample_table: pa.Table) -> None:
        """Test _perform_aggregation method with invalid aggregation type."""
        with pytest.raises(ValueError):
            PyArrowAggregatedFeatureGroup._perform_aggregation(sample_table, "invalid", ["sales"])

    def test_calculate_feature_single(self, sample_table: pa.Table, feature_set_sum: FeatureSet) -> None:
        """Test calculate_feature method with a single aggregation."""
        result = PyArrowAggregatedFeatureGroup.calculate_feature(sample_table, feature_set_sum)

        # Check that the result contains the original data plus the aggregated feature
        assert "sum_aggr__sales" in result.schema.names
        assert result.column("sum_aggr__sales")[0].as_py() == 1500  # Sum of [100, 200, 300, 400, 500]

        # Check that the original data is preserved
        assert "sales" in result.schema.names
        assert "quantity" in result.schema.names
        assert "price" in result.schema.names
        assert "discount" in result.schema.names
        assert "customer_rating" in result.schema.names

    def test_calculate_feature_multiple(self, sample_table: pa.Table, feature_set_multiple: FeatureSet) -> None:
        """Test calculate_feature method with multiple aggregations."""
        result = PyArrowAggregatedFeatureGroup.calculate_feature(sample_table, feature_set_multiple)

        # Check that the result contains all aggregated features
        assert "sum_aggr__sales" in result.schema.names
        assert result.column("sum_aggr__sales")[0].as_py() == 1500  # Sum of [100, 200, 300, 400, 500]

        assert "avg_aggr__price" in result.schema.names
        assert result.column("avg_aggr__price")[0].as_py() == 9.0  # Avg of [10.0, 9.5, 9.0, 8.5, 8.0]

        assert "min_aggr__discount" in result.schema.names
        assert result.column("min_aggr__discount")[0].as_py() == 0.1  # Min of [0.1, 0.2, 0.15, 0.25, 0.1]

        assert "max_aggr__customer_rating" in result.schema.names
        assert result.column("max_aggr__customer_rating")[0].as_py() == 5  # Max of [4, 5, 3, 4, 5]

        # Check that the original data is preserved
        assert "sales" in result.schema.names
        assert "quantity" in result.schema.names
        assert "price" in result.schema.names
        assert "discount" in result.schema.names
        assert "customer_rating" in result.schema.names

    def test_calculate_feature_missing_source(self, sample_table: pa.Table) -> None:
        """Test calculate_feature method with missing source feature."""
        feature_set = FeatureSet()
        feature_set.add(Feature("sum_aggr__missing"))

        with pytest.raises(ValueError, match="None of the source features"):
            PyArrowAggregatedFeatureGroup.calculate_feature(sample_table, feature_set)

    def test_calculate_feature_invalid_aggregation(self, sample_table: pa.Table) -> None:
        """Test calculate_feature method with invalid aggregation type."""
        # Temporarily modify the AGGREGATION_TYPES to simulate an invalid aggregation type
        original_types = AggregatedFeatureGroup.AGGREGATION_TYPES.copy()
        try:
            AggregatedFeatureGroup.AGGREGATION_TYPES = {"sum": "Sum of values"}

            feature_set = FeatureSet()
            feature_set.add(Feature("min_aggr__sales"))

            with pytest.raises(ValueError, match="Unsupported aggregation type: min"):
                PyArrowAggregatedFeatureGroup.calculate_feature(sample_table, feature_set)
        finally:
            # Restore the original AGGREGATION_TYPES
            AggregatedFeatureGroup.AGGREGATION_TYPES = original_types


class TestAggPyArrowIntegration:
    """Integration tests for the aggregated feature group using DataCreator."""

    def test_aggregation_with_data_creator(self) -> None:
        """Test aggregation features with mlodaAPI using DataCreator."""

        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {PyArrowAggregatedTestDataCreator, PyArrowAggregatedFeatureGroup}
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
            compute_frameworks={PyarrowTable},
            plugin_collector=plugin_collector,
        )

        new_res: List[pd.DataFrame] = []
        for res in result:
            new_res.append(res.to_pandas())

        validate_aggregated_features(new_res)
