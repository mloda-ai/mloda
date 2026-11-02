"""Aggregated family contract: broadcast, row-wise, recompute, several features, unsupported type.

Every check except the missing-source one runs the implementation's own calculate on framework-native input.
The missing-source check runs the core input check."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import pyarrow as pa
import pytest

from mloda.user import Feature
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from tests.test_plugins.feature_group.experimental.core_input_check import assert_core_rejects_missing_input
from tests.test_plugins.feature_group.experimental.test_base_aggregated_feature_group.test_aggregated_utils import (
    AggregatedTestDataCreator,
)
from tests.test_plugins.feature_group.experimental.zero_row_result_type_test_mixin import ZeroRowResultTypeTestMixin

SAMPLE = pa.table(AggregatedTestDataCreator.get_raw_data())
METRICS = pa.table(
    {
        "metrics~0": pa.array([1, 2, 3, 4], type=pa.int64()),
        "metrics~1": pa.array([10, 20, 30, 40], type=pa.int64()),
    }
)
EXPECTED = {
    "sales__sum_aggr": 1500,
    "price__avg_aggr": 9.0,
    "discount__min_aggr": 0.1,
    "customer_rating__max_aggr": 5,
    "quantity__count_aggr": 5,
    "discount__median_aggr": 0.15,
}


class AggregatedCalculateTestMixin(ZeroRowResultTypeTestMixin):
    """Calculate, unsupported-type and missing-source checks shared by every aggregation implementation."""

    @abstractmethod
    def column_names(self, result: Any) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def column_values(self, result: Any, column: str) -> list[Any]:
        raise NotImplementedError

    def test_scalar_result_is_broadcast_to_every_row(self) -> None:
        result = self.calculate(SAMPLE, Feature("sales__sum_aggr"))
        assert self.column_values(result, "sales__sum_aggr") == [1500] * SAMPLE.num_rows
        for name in SAMPLE.column_names:
            assert self.column_values(result, name) == SAMPLE.column(name).to_pylist()

    def test_row_wise_result_is_a_flat_column(self) -> None:
        result = self.calculate(METRICS, Feature("metrics__sum_aggr"))
        assert self.column_values(result, "metrics__sum_aggr") == [11, 22, 33, 44]

    @pytest.mark.parametrize(
        ("table", "name", "expected"),
        [
            (SAMPLE, "sales__sum_aggr", [1500] * 5),
            (METRICS, "metrics__sum_aggr", [11, 22, 33, 44]),
        ],
        ids=["scalar", "row_wise"],
    )
    def test_recompute_replaces_the_column(self, table: pa.Table, name: str, expected: list[Any]) -> None:
        stale = table.append_column(name, pa.array([0] * table.num_rows, type=pa.int64()))
        result = self.calculate(stale, Feature(name))
        assert self.column_names(result).count(name) == 1
        assert self.column_values(result, name) == expected

    def test_calculate_several_features(self) -> None:
        result = self.calculate(SAMPLE, *[Feature(name) for name in EXPECTED])
        for name, value in EXPECTED.items():
            assert self.column_values(result, name) == [value] * SAMPLE.num_rows
        for name in SAMPLE.column_names:
            assert self.column_values(result, name) == SAMPLE.column(name).to_pylist()

    def test_calculate_rejects_an_unsupported_aggregation_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(AggregatedFeatureGroup, "AGGREGATION_TYPES", {"sum": "Sum of values"})
        with pytest.raises(ValueError, match="Unsupported aggregation type: min"):
            self.calculate(SAMPLE, Feature("sales__min_aggr"))

    def test_core_rejects_a_missing_source(self) -> None:
        error = assert_core_rejects_missing_input(
            self.feature_group_class, self.from_arrow(SAMPLE), "missing__sum_aggr", ("missing",)
        )
        assert "sales" in error.available_columns
