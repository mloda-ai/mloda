"""Aggregated family calculate contract: single and multiple features on the sample data, source columns kept,
unsupported aggregation types rejected."""

from __future__ import annotations

import pyarrow as pa
import pytest

from mloda.user import Feature
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from tests.test_plugins.feature_group.experimental.zero_row_result_type_test_mixin import ZeroRowResultTypeTestMixin

SAMPLE = pa.table(
    {
        "sales": [100, 200, 300, 400, 500],
        "quantity": [10, 20, 30, 40, 50],
        "price": [10.0, 9.5, 9.0, 8.5, 8.0],
        "discount": [0.1, 0.2, 0.15, 0.25, 0.1],
        "customer_rating": [4, 5, 3, 4, 5],
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
    """Calculate tests on the sample table, run through the framework adapter."""

    def _assert_aggregated(self, result: object, expected: dict[str, float]) -> None:
        for name, value in expected.items():
            assert self.column_values(result, name) == [pytest.approx(value)] * SAMPLE.num_rows
        names = self.column_names(result)
        assert all(source in names for source in SAMPLE.column_names)

    def test_calculate_single_feature(self) -> None:
        result = self.calculate(SAMPLE, Feature("sales__sum_aggr"))
        self._assert_aggregated(result, {"sales__sum_aggr": EXPECTED["sales__sum_aggr"]})

    def test_calculate_multiple_features(self) -> None:
        result = self.calculate(SAMPLE, *(Feature(name) for name in EXPECTED))
        self._assert_aggregated(result, EXPECTED)

    def test_calculate_unsupported_aggregation_type_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(AggregatedFeatureGroup, "AGGREGATION_TYPES", {"sum": "Sum of values"})
        with pytest.raises(ValueError, match="Unsupported aggregation type: min"):
            self.calculate(SAMPLE, Feature("sales__min_aggr"))
