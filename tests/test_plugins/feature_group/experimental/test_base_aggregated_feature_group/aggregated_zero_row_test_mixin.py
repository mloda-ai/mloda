"""Aggregated family zero-row contract: typed results on zero rows and all-null input, int32 width kept by min/max.

A framework rejecting an aggregation across multiple columns maps it to its ValueError regex in
unsupported_multi_column_aggregations."""

from __future__ import annotations

from typing import ClassVar

import pyarrow as pa
import pytest

from mloda.user import Feature
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from tests.test_plugins.feature_group.experimental.zero_row_result_type_test_mixin import ZeroRowResultTypeTestMixin

AGGREGATION_TYPES = sorted(AggregatedFeatureGroup.AGGREGATION_TYPES)

ZERO_ROWS = pa.table(
    {
        "sales": pa.array([], type=pa.int64()),
        "price": pa.array([], type=pa.float64()),
        "metrics~0": pa.array([], type=pa.int64()),
        "metrics~1": pa.array([], type=pa.int64()),
    }
)
ALL_NULL_PRICE = pa.table({"price": pa.array([None, None, None], type=pa.float64())})
INT32_SALES = pa.table({"sales": pa.array([100, 200, 300], type=pa.int32())})


class AggregatedZeroRowTestMixin(ZeroRowResultTypeTestMixin):
    """Zero-row, all-null and int32-width tests over every aggregation type."""

    unsupported_multi_column_aggregations: ClassVar[dict[str, str]] = {}

    @pytest.mark.parametrize("source", ["sales", "price"])
    @pytest.mark.parametrize("aggregation", AGGREGATION_TYPES)
    def test_zero_rows_single_column_result_is_typed(self, aggregation: str, source: str) -> None:
        feature_name = f"{source}__{aggregation}_aggr"
        self.assert_typed(self.calculate(ZERO_ROWS, Feature(feature_name)), feature_name, 0)

    @pytest.mark.parametrize("aggregation", AGGREGATION_TYPES)
    def test_zero_rows_multi_column_result_is_typed(self, aggregation: str) -> None:
        feature_name = f"metrics__{aggregation}_aggr"
        if aggregation in self.unsupported_multi_column_aggregations:
            with pytest.raises(ValueError, match=self.unsupported_multi_column_aggregations[aggregation]):
                self.calculate(ZERO_ROWS, Feature(feature_name))
            return
        self.assert_typed(self.calculate(ZERO_ROWS, Feature(feature_name)), feature_name, 0)

    @pytest.mark.parametrize("aggregation", AGGREGATION_TYPES)
    def test_all_null_input_result_is_typed(self, aggregation: str) -> None:
        feature_name = f"price__{aggregation}_aggr"
        self.assert_typed(self.calculate(ALL_NULL_PRICE, Feature(feature_name)), feature_name, 3)

    @pytest.mark.parametrize("aggregation", ["min", "max"])
    def test_min_max_preserve_int32_source_width(self, aggregation: str) -> None:
        feature_name = f"sales__{aggregation}_aggr"
        result = self.calculate(INT32_SALES, Feature(feature_name))
        assert self.column_type(result, feature_name) == self.column_type(self.from_arrow(INT32_SALES), "sales")
