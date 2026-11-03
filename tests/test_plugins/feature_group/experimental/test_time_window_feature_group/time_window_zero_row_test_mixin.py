"""Time window family zero-row contract: every window function yields a typed result column with zero rows."""

from __future__ import annotations

import pytest

from mloda._testing.typed_data import TypedTestData
from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.provider import DefaultOptionKeys
from mloda.user import Feature
from mloda_plugins.feature_group.experimental.time_window.base import TimeWindowFeatureGroup
from tests.test_plugins.feature_group.experimental.zero_row_result_type_test_mixin import ZeroRowResultTypeTestMixin

WINDOW_FUNCTIONS = sorted(TimeWindowFeatureGroup.WINDOW_FUNCTIONS)

ZERO_ROWS = TypedTestData(
    {
        "temperature": (DataType.INT64, []),
        "humidity": (DataType.DOUBLE, []),
        DefaultOptionKeys.reference_time.value: (DataType.TIMESTAMP_MICROS, []),
    }
)


class TimeWindowZeroRowTestMixin(ZeroRowResultTypeTestMixin):
    """Zero-row int64 and float64 sources, for every window function."""

    @pytest.mark.parametrize("source", ["temperature", "humidity"])
    @pytest.mark.parametrize("window_function", WINDOW_FUNCTIONS)
    def test_zero_rows_result_is_typed(self, window_function: str, source: str) -> None:
        feature_name = f"{source}__{window_function}_3_day_window"
        pa = pytest.importorskip("pyarrow")
        arrow_types = {DefaultOptionKeys.reference_time.value: pa.timestamp("ns")}
        result = self.calculate(ZERO_ROWS, Feature(feature_name), arrow_types=arrow_types)
        self.assert_typed(result, feature_name, 0)
