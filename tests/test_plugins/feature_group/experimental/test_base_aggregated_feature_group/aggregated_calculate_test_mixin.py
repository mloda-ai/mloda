"""Aggregated family contract: broadcast, row-wise, recompute, several features, unsupported type.

Every check except the missing-source one runs the implementation's own calculate on framework-native input.
The missing-source check runs the core input check."""

from __future__ import annotations

from typing import Any

import pytest

from mloda._testing.typed_data import TypedTestData
from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.user import Feature
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from tests.test_plugins.feature_group.experimental.core_input_check import assert_core_rejects_missing_input
from tests.test_plugins.feature_group.experimental.test_base_aggregated_feature_group.test_aggregated_utils import (
    AggregatedTestDataCreator,
)
from tests.test_plugins.feature_group.experimental.zero_row_result_type_test_mixin import (
    ZeroRowResultTypeTestMixin,
    edit_columns,
)

_SAMPLE_TYPES = {
    "sales": DataType.INT64,
    "quantity": DataType.INT64,
    "price": DataType.DOUBLE,
    "discount": DataType.DOUBLE,
    "customer_rating": DataType.INT64,
}
SAMPLE = TypedTestData(
    {name: (_SAMPLE_TYPES[name], values) for name, values in AggregatedTestDataCreator.get_raw_data().items()}
)
METRICS = TypedTestData(
    {
        "metrics~0": (DataType.INT64, [1, 2, 3, 4]),
        "metrics~1": (DataType.INT64, [10, 20, 30, 40]),
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

    def test_scalar_result_is_broadcast_to_every_row(self) -> None:
        result = self.calculate(SAMPLE, Feature("sales__sum_aggr"))
        assert self.adapter.column_values(result, "sales__sum_aggr") == [1500] * SAMPLE.row_count
        for name, values in SAMPLE.to_python_dict().items():
            assert self.adapter.column_values(result, name) == values

    def test_row_wise_result_is_a_flat_column(self) -> None:
        result = self.calculate(METRICS, Feature("metrics__sum_aggr"))
        assert self.adapter.column_values(result, "metrics__sum_aggr") == [11, 22, 33, 44]

    @pytest.mark.parametrize(
        ("table", "name", "expected"),
        [
            (SAMPLE, "sales__sum_aggr", [1500] * 5),
            (METRICS, "metrics__sum_aggr", [11, 22, 33, 44]),
        ],
        ids=["scalar", "row_wise"],
    )
    def test_recompute_replaces_the_column(self, table: TypedTestData, name: str, expected: list[Any]) -> None:
        stale = edit_columns(table, add={name: (DataType.INT64, [0] * table.row_count)})
        result = self.calculate(stale, Feature(name))
        assert self.adapter.column_names(result).count(name) == 1
        assert self.adapter.column_values(result, name) == expected

    def test_calculate_several_features(self) -> None:
        result = self.calculate(SAMPLE, *[Feature(name) for name in EXPECTED])
        for name, value in EXPECTED.items():
            assert self.adapter.column_values(result, name) == [value] * SAMPLE.row_count
        for name, values in SAMPLE.to_python_dict().items():
            assert self.adapter.column_values(result, name) == values

    def test_calculate_rejects_an_unsupported_aggregation_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(AggregatedFeatureGroup, "AGGREGATION_TYPES", {"sum": "Sum of values"})
        with pytest.raises(ValueError, match="Unsupported aggregation type: min"):
            self.calculate(SAMPLE, Feature("sales__min_aggr"))

    def test_core_rejects_a_missing_source(self) -> None:
        error = assert_core_rejects_missing_input(
            self.feature_group_class, self.adapter.from_typed(SAMPLE), "missing__sum_aggr", ("missing",)
        )
        assert "sales" in error.available_columns
