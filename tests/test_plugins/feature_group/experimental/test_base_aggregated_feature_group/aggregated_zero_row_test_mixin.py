"""Shared contract: aggregations over zero rows or all-null input yield typed result columns.

Each framework-specific test class inherits AggregatedZeroRowTestMixin and provides:
- feature_group fixture: the aggregated feature group class under test
- zero_row_data fixture: zero rows, sales int64, price float64, and multi-column sources
  metrics~0 and metrics~1 int64
- all_null_data fixture: three rows, price float64, every value null
- int32_data fixture: sales int32 = [100, 200, 300]
- row_count(result): the number of rows in a result
- column_type(result, column): the framework-native type of a result column
- int32_type class attribute: the framework-native int32 type
- untyped_column_types class attribute: the types that mean the result lost its type
- unsupported_multi_column_aggregations class attribute: aggregation types the framework rejects
  across multiple columns, mapped to a regex their ValueError message must match
"""

from abc import abstractmethod
from typing import Any

import pytest

from mloda.provider import FeatureSet
from mloda.user import Feature
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup

AGGREGATION_TYPES = sorted(AggregatedFeatureGroup.AGGREGATION_TYPES)


def calculate(feature_group: type[AggregatedFeatureGroup], data: Any, feature_name: str) -> Any:
    """Run calculate_feature for a single feature name."""
    feature_set = FeatureSet()
    feature_set.add(Feature(feature_name))
    return feature_group.calculate_feature(data, feature_set)


class AggregatedZeroRowTestMixin:
    """Shared zero-row, all-null, and min/max integer-width tests for all aggregated feature groups."""

    int32_type: Any
    untyped_column_types: tuple[Any, ...]
    unsupported_multi_column_aggregations: dict[str, str] = {}

    @pytest.fixture
    @abstractmethod
    def feature_group(self) -> type[AggregatedFeatureGroup]:
        raise NotImplementedError

    @pytest.fixture
    @abstractmethod
    def zero_row_data(self) -> Any:
        raise NotImplementedError

    @pytest.fixture
    @abstractmethod
    def all_null_data(self) -> Any:
        raise NotImplementedError

    @pytest.fixture
    @abstractmethod
    def int32_data(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def row_count(self, result: Any) -> int:
        raise NotImplementedError

    @abstractmethod
    def column_type(self, result: Any, column: str) -> Any:
        raise NotImplementedError

    def assert_typed(self, result: Any, column: str, expected_rows: int) -> None:
        assert self.row_count(result) == expected_rows
        column_type = self.column_type(result, column)
        assert column_type not in self.untyped_column_types, f"{column} lost its type: {column_type}"

    @pytest.mark.parametrize("source", ["sales", "price"])
    @pytest.mark.parametrize("aggregation", AGGREGATION_TYPES)
    def test_zero_rows_single_column_result_is_typed(
        self, feature_group: type[AggregatedFeatureGroup], zero_row_data: Any, aggregation: str, source: str
    ) -> None:
        feature_name = f"{source}__{aggregation}_aggr"
        result = calculate(feature_group, zero_row_data, feature_name)
        self.assert_typed(result, feature_name, 0)

    @pytest.mark.parametrize("aggregation", AGGREGATION_TYPES)
    def test_zero_rows_multi_column_result_is_typed(
        self, feature_group: type[AggregatedFeatureGroup], zero_row_data: Any, aggregation: str
    ) -> None:
        feature_name = f"metrics__{aggregation}_aggr"
        if aggregation in self.unsupported_multi_column_aggregations:
            with pytest.raises(ValueError, match=self.unsupported_multi_column_aggregations[aggregation]):
                calculate(feature_group, zero_row_data, feature_name)
            return
        result = calculate(feature_group, zero_row_data, feature_name)
        self.assert_typed(result, feature_name, 0)

    @pytest.mark.parametrize("aggregation", AGGREGATION_TYPES)
    def test_all_null_input_result_is_typed(
        self, feature_group: type[AggregatedFeatureGroup], all_null_data: Any, aggregation: str
    ) -> None:
        feature_name = f"price__{aggregation}_aggr"
        result = calculate(feature_group, all_null_data, feature_name)
        self.assert_typed(result, feature_name, 3)

    @pytest.mark.parametrize("aggregation", ["min", "max"])
    def test_min_max_preserve_int32_source_width(
        self, feature_group: type[AggregatedFeatureGroup], int32_data: Any, aggregation: str
    ) -> None:
        feature_name = f"sales__{aggregation}_aggr"
        result = calculate(feature_group, int32_data, feature_name)
        assert self.column_type(result, feature_name) == self.int32_type
