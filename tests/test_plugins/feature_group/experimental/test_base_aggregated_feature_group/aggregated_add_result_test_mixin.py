"""Aggregated family add-result contract: scalar broadcast, flat row-wise column, recompute without duplicates.

Every check runs the implementation's own calculate on framework-native input."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import pyarrow as pa
import pytest

from mloda.user import Feature
from tests.test_plugins.feature_group.experimental.zero_row_result_type_test_mixin import ZeroRowResultTypeTestMixin

SALES = pa.table({"sales": pa.array([100, 200, 300, 400, 500], type=pa.int64())})
METRICS = pa.table(
    {
        "metrics~0": pa.array([1, 2, 3, 4], type=pa.int64()),
        "metrics~1": pa.array([10, 20, 30, 40], type=pa.int64()),
    }
)


class AggregatedAddResultTestMixin(ZeroRowResultTypeTestMixin):
    """Broadcast, flat row-wise column and no-duplicate checks over calculate."""

    @abstractmethod
    def column_names(self, result: Any) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def column_values(self, result: Any, column: str) -> list[Any]:
        raise NotImplementedError

    def test_scalar_result_is_broadcast_to_every_row(self) -> None:
        result = self.calculate(SALES, Feature("sales__sum_aggr"))
        assert self.column_values(result, "sales__sum_aggr") == [1500] * 5
        assert self.column_values(result, "sales") == [100, 200, 300, 400, 500]

    def test_row_wise_result_is_a_flat_column(self) -> None:
        result = self.calculate(METRICS, Feature("metrics__sum_aggr"))
        assert self.column_values(result, "metrics__sum_aggr") == [11, 22, 33, 44]

    @pytest.mark.parametrize(
        ("table", "name", "expected"),
        [
            (SALES, "sales__sum_aggr", [1500] * 5),
            (METRICS, "metrics__sum_aggr", [11, 22, 33, 44]),
        ],
        ids=["scalar", "row_wise"],
    )
    def test_recompute_replaces_the_column(self, table: pa.Table, name: str, expected: list[Any]) -> None:
        stale = table.append_column(name, pa.array([0] * table.num_rows, type=pa.int64()))
        result = self.calculate(stale, Feature(name))
        assert self.column_names(result).count(name) == 1
        assert self.column_values(result, name) == expected
