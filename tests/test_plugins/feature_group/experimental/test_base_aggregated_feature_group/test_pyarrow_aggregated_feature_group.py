import pyarrow as pa
import pandas as pd
import pytest

from mloda.user import mloda
from mloda.user import Feature
from mloda.provider import FeatureSet
from mloda.user import PluginCollector
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
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
    feature_set.add(Feature("sales__sum_aggr"))
    return feature_set


@pytest.fixture
def feature_set_multiple() -> FeatureSet:
    """Create a feature set with multiple aggregation features."""
    feature_set = FeatureSet()
    feature_set.add(Feature("sales__sum_aggr"))
    feature_set.add(Feature("price__avg_aggr"))
    feature_set.add(Feature("discount__min_aggr"))
    feature_set.add(Feature("customer_rating__max_aggr"))
    return feature_set


@pytest.fixture
def multi_source_table() -> pa.Table:
    """Table with multi-column source features (metrics~0, metrics~1) for row-wise aggregation."""
    return pa.table(
        {
            "metrics~0": [1, 2, 3, 4],
            "metrics~1": [10, 20, 30, 40],
        }
    )


@pytest.fixture
def multi_source_table_with_null() -> pa.Table:
    """Two source columns, row index 2 has a null in metrics~0 only."""
    return pa.table(
        {
            "metrics~0": pa.array([1.0, 2.0, None, 4.0], type=pa.float64()),
            "metrics~1": pa.array([10.0, 20.0, 30.0, 40.0], type=pa.float64()),
        }
    )


@pytest.fixture
def multi_source_table_three_columns() -> pa.Table:
    """Three source columns, no nulls, distinct values so std/var is meaningful."""
    return pa.table(
        {
            "metrics~0": [1.0, 4.0, 100.0, 8.0],
            "metrics~1": [2.0, 10.0, 50.0, 8.0],
            "metrics~2": [3.0, 7.0, 25.0, 9.0],
        }
    )


class TestPyArrowAggregatedFeatureGroup:
    """Tests for the PyArrowAggregatedFeatureGroup class."""

    def test_compute_framework_rule(self) -> None:
        """Test compute_framework_rule method."""
        assert PyArrowAggregatedFeatureGroup.compute_framework_rule() == {PyArrowTable}

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
        # ddof=1 sample standard deviation, matching pandas' default.
        assert abs(result - 158.11) < 0.1  # Std of [100, 200, 300, 400, 500] with sample formula

    def test_perform_aggregation_var(self, sample_table: pa.Table) -> None:
        """Test _perform_aggregation method with var aggregation."""
        result = PyArrowAggregatedFeatureGroup._perform_aggregation(sample_table, "var", ["sales"])
        # ddof=1 sample variance, matching pandas' default.
        assert abs(result - 25000) < 0.1  # Var of [100, 200, 300, 400, 500] with sample formula

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
        assert "sales__sum_aggr" in result.schema.names
        assert result.column("sales__sum_aggr")[0].as_py() == 1500  # Sum of [100, 200, 300, 400, 500]

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
        assert "sales__sum_aggr" in result.schema.names
        assert result.column("sales__sum_aggr")[0].as_py() == 1500  # Sum of [100, 200, 300, 400, 500]

        assert "price__avg_aggr" in result.schema.names
        assert result.column("price__avg_aggr")[0].as_py() == 9.0  # Avg of [10.0, 9.5, 9.0, 8.5, 8.0]

        assert "discount__min_aggr" in result.schema.names
        assert result.column("discount__min_aggr")[0].as_py() == 0.1  # Min of [0.1, 0.2, 0.15, 0.25, 0.1]

        assert "customer_rating__max_aggr" in result.schema.names
        assert result.column("customer_rating__max_aggr")[0].as_py() == 5  # Max of [4, 5, 3, 4, 5]

        # Check that the original data is preserved
        assert "sales" in result.schema.names
        assert "quantity" in result.schema.names
        assert "price" in result.schema.names
        assert "discount" in result.schema.names
        assert "customer_rating" in result.schema.names

    def test_calculate_feature_missing_source(self, sample_table: pa.Table) -> None:
        """Test calculate_feature method with missing source feature."""
        feature_set = FeatureSet()
        feature_set.add(Feature("missing__sum_aggr"))

        with pytest.raises(ValueError, match="None of the source features"):
            PyArrowAggregatedFeatureGroup.calculate_feature(sample_table, feature_set)

    def test_calculate_feature_invalid_aggregation(self, sample_table: pa.Table) -> None:
        """Test calculate_feature method with invalid aggregation type."""
        # Temporarily modify the AGGREGATION_TYPES to simulate an invalid aggregation type
        original_types = AggregatedFeatureGroup.AGGREGATION_TYPES.copy()
        try:
            AggregatedFeatureGroup.AGGREGATION_TYPES = {"sum": "Sum of values"}

            feature_set = FeatureSet()
            feature_set.add(Feature("sales__min_aggr"))

            with pytest.raises(ValueError, match="Unsupported aggregation type: min"):
                PyArrowAggregatedFeatureGroup.calculate_feature(sample_table, feature_set)
        finally:
            # Restore the original AGGREGATION_TYPES
            AggregatedFeatureGroup.AGGREGATION_TYPES = original_types


class TestPyArrowAggregatedFeatureGroupMultiColumn:
    """Pins down bugs in _add_result_to_data for multi-column (row-wise) aggregation results."""

    def test_add_result_to_data_multi_column_result_is_flat_not_nested(self, multi_source_table: pa.Table) -> None:
        """A row-wise numpy result must become a flat length-n column, not an n-by-n nested ListArray."""
        result = PyArrowAggregatedFeatureGroup._perform_aggregation(
            multi_source_table, "sum", ["metrics~0", "metrics~1"]
        )

        updated = PyArrowAggregatedFeatureGroup._add_result_to_data(multi_source_table, "metrics__sum_aggr", result)

        result_col = updated.column("metrics__sum_aggr")
        assert not pa.types.is_list(result_col.type), (
            f"expected a flat scalar column, got nested list type {result_col.type}"
        )
        assert len(result_col) == multi_source_table.num_rows
        assert result_col.to_pylist() == [11, 22, 33, 44]

    def test_add_result_to_data_recompute_does_not_duplicate_column(self, multi_source_table: pa.Table) -> None:
        """Calling _add_result_to_data twice for the same feature_name must not duplicate the column."""
        result = PyArrowAggregatedFeatureGroup._perform_aggregation(
            multi_source_table, "sum", ["metrics~0", "metrics~1"]
        )

        once = PyArrowAggregatedFeatureGroup._add_result_to_data(multi_source_table, "metrics__sum_aggr", result)
        twice = PyArrowAggregatedFeatureGroup._add_result_to_data(once, "metrics__sum_aggr", result)

        assert twice.schema.names.count("metrics__sum_aggr") == 1

    def test_calculate_feature_multi_column_row_wise_sum(self, multi_source_table: pa.Table) -> None:
        """End-to-end: calculate_feature must produce a flat per-row column and not duplicate on recompute."""
        feature_set = FeatureSet()
        feature_set.add(Feature("metrics__sum_aggr"))

        result = PyArrowAggregatedFeatureGroup.calculate_feature(multi_source_table, feature_set)

        result_col = result.column("metrics__sum_aggr")
        assert not pa.types.is_list(result_col.type), (
            f"expected a flat scalar column, got nested list type {result_col.type}"
        )
        assert len(result_col) == multi_source_table.num_rows
        assert result_col.to_pylist() == [11, 22, 33, 44]

        recomputed = PyArrowAggregatedFeatureGroup.calculate_feature(result, feature_set)
        assert recomputed.schema.names.count("metrics__sum_aggr") == 1


class TestPyArrowAggregatedFeatureGroupDdofAndNullSkip:
    """Pins down ddof=1 (sample statistics) and null-skip semantics for std/var/sum/count.

    Ground truth for every assertion is computed from pandas inline (never hardcoded),
    since pandas' single-column .std()/.var() and multi-column .sum(axis=1, skipna=True)/
    .std(axis=1, skipna=True) already implement the target semantics.
    """

    def test_perform_aggregation_std_single_column_matches_pandas_ddof1(self, sample_table: pa.Table) -> None:
        """PyArrow single-column std must use ddof=1 (sample), not ddof=0 (population)."""
        result = PyArrowAggregatedFeatureGroup._perform_aggregation(sample_table, "std", ["sales"])

        expected = pd.Series(sample_table.column("sales").to_pylist()).std()  # ddof=1 by default

        assert abs(result - expected) < 1e-6

    def test_perform_aggregation_var_single_column_matches_pandas_ddof1(self, sample_table: pa.Table) -> None:
        """PyArrow single-column var must use ddof=1 (sample), not ddof=0 (population)."""
        result = PyArrowAggregatedFeatureGroup._perform_aggregation(sample_table, "var", ["sales"])

        expected = pd.Series(sample_table.column("sales").to_pylist()).var()  # ddof=1 by default

        assert abs(result - expected) < 1e-6

    def test_perform_aggregation_sum_multi_column_skips_null(self, multi_source_table_with_null: pa.Table) -> None:
        """A null in one source column must be skipped, not propagated as NaN, for the row's sum."""
        result = PyArrowAggregatedFeatureGroup._perform_aggregation(
            multi_source_table_with_null, "sum", ["metrics~0", "metrics~1"]
        )

        expected = pd.DataFrame(
            {
                "metrics~0": multi_source_table_with_null.column("metrics~0").to_pylist(),
                "metrics~1": multi_source_table_with_null.column("metrics~1").to_pylist(),
            }
        ).sum(axis=1, skipna=True)

        for row_index in range(len(expected)):
            assert not pd.isna(result[row_index]), (
                f"row {row_index}: expected a skip-null sum, got NaN (null propagated instead of skipped)"
            )
            assert abs(result[row_index] - expected[row_index]) < 1e-9

    def test_perform_aggregation_count_multi_column_excludes_null_without_crashing(
        self, multi_source_table_with_null: pa.Table
    ) -> None:
        """Count across columns must not crash and must exclude the null value for the affected row."""
        result = PyArrowAggregatedFeatureGroup._perform_aggregation(
            multi_source_table_with_null, "count", ["metrics~0", "metrics~1"]
        )

        # Row index 2 has a null in metrics~0, so only metrics~1 counts.
        assert result[2] == 1

    def test_perform_aggregation_std_multi_column_matches_pandas_ddof1(
        self, multi_source_table_three_columns: pa.Table
    ) -> None:
        """PyArrow row-wise std across columns must use ddof=1 (sample), not ddof=0."""
        result = PyArrowAggregatedFeatureGroup._perform_aggregation(
            multi_source_table_three_columns, "std", ["metrics~0", "metrics~1", "metrics~2"]
        )

        expected = pd.DataFrame(
            {
                "metrics~0": multi_source_table_three_columns.column("metrics~0").to_pylist(),
                "metrics~1": multi_source_table_three_columns.column("metrics~1").to_pylist(),
                "metrics~2": multi_source_table_three_columns.column("metrics~2").to_pylist(),
            }
        ).std(axis=1)  # ddof=1 by default

        for row_index in range(len(expected)):
            assert abs(result[row_index] - expected[row_index]) < 1e-6

    def test_perform_aggregation_var_multi_column_matches_pandas_ddof1(
        self, multi_source_table_three_columns: pa.Table
    ) -> None:
        """PyArrow row-wise var across columns must use ddof=1 (sample), not ddof=0."""
        result = PyArrowAggregatedFeatureGroup._perform_aggregation(
            multi_source_table_three_columns, "var", ["metrics~0", "metrics~1", "metrics~2"]
        )

        expected = pd.DataFrame(
            {
                "metrics~0": multi_source_table_three_columns.column("metrics~0").to_pylist(),
                "metrics~1": multi_source_table_three_columns.column("metrics~1").to_pylist(),
                "metrics~2": multi_source_table_three_columns.column("metrics~2").to_pylist(),
            }
        ).var(axis=1)  # ddof=1 by default

        for row_index in range(len(expected)):
            assert abs(result[row_index] - expected[row_index]) < 1e-6


class TestAggPyArrowIntegration:
    """Integration tests for the aggregated feature group using DataCreator."""

    def test_aggregation_with_data_creator(self) -> None:
        """Test aggregation features with mloda using DataCreator."""

        # Enable the necessary feature groups
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowAggregatedTestDataCreator, PyArrowAggregatedFeatureGroup}
        )

        # Run the mloda with multiple aggregation features
        result = mloda.run_all(
            [
                "sales",
                "sales__sum_aggr",
                "price__avg_aggr",
                "discount__min_aggr",
                "customer_rating__max_aggr",
            ],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        new_res: list[pd.DataFrame] = []
        for res in result:
            new_res.append(res.to_pandas())

        validate_aggregated_features(new_res)
