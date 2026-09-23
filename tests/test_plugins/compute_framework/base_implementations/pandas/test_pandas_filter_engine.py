"""Unit tests for the PandasFilterEngine class."""

from decimal import Decimal
from typing import Any

import pytest
import numpy as np
import pandas as pd
import pyarrow as pa

from mloda.user import Feature
from mloda.user import SingleFilter
from mloda.user import FilterType

from mloda_plugins.compute_framework.base_implementations.pandas.pandas_filter_engine import PandasFilterEngine

from tests.test_plugins.compute_framework.base_implementations.filter_engine_test_mixin import (
    FilterEngineTestMixin,
)
from tests.test_plugins.compute_framework.base_implementations.time_range_filter_engine_test_mixin import (
    SAMPLE_IDS,
    SAMPLE_TIMESTAMPS,
    TimeRangeFilterEngineTestMixin,
)


class TestPandasFilterEngine(FilterEngineTestMixin, TimeRangeFilterEngineTestMixin):
    """Unit tests for the PandasFilterEngine class using shared mixins."""

    @pytest.fixture
    def filter_engine(self) -> Any:
        """Return the PandasFilterEngine class."""
        return PandasFilterEngine

    @pytest.fixture
    def sample_data(self) -> Any:
        """Create a sample pandas DataFrame for testing."""
        return pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "age": [25, 30, 35, 40, 45],
                "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
                "category": ["A", "B", "A", "C", "B"],
            }
        )

    @pytest.fixture
    def nullable_category_sample_data(self) -> Any:
        """Create a sample pandas DataFrame with null categories for testing."""
        return pd.DataFrame(
            {"id": [1, 2, 3, 4, 5], "category": ["A", None, "B", None, "C"], "score": [1, None, 2, None, 3]}
        )

    def get_column_values(self, result: Any, column: str) -> list[Any]:
        """Extract column values from pandas DataFrame, mapping missing values to None."""
        return [None if pd.isna(value) else value for value in result[column].tolist()]

    @pytest.fixture
    def decimal_sample_data(self) -> Any:
        values = [Decimal("12.34"), Decimal("5.50"), Decimal("99.99"), None]
        return pd.DataFrame({"d": pd.Series(values, dtype=pd.ArrowDtype(pa.decimal128(10, 2)))})

    def get_decimal_column_dtype(self, data: Any) -> Any:
        return data["d"].dtype

    @pytest.fixture
    def sample_time_data(self) -> Any:
        return pd.DataFrame({"id": SAMPLE_IDS, "ts": pd.to_datetime(SAMPLE_TIMESTAMPS, utc=True)})

    def get_id_column_values(self, result: Any) -> list[int]:
        return list(result["id"].tolist())

    @pytest.mark.parametrize(
        ("column", "series", "values", "expected_ids"),
        [
            pytest.param(
                "nullable_int",
                pd.Series([1, None, 2, None, 3], dtype="Int64"),
                [1, None],
                [1, 2, 4],
                id="Int64_pd_NA",
            ),
            pytest.param(
                "ts",
                pd.Series(
                    [pd.Timestamp("2024-01-01"), None, pd.Timestamp("2024-01-02"), None, pd.Timestamp("2024-01-03")]
                ),
                [None],
                [2, 4],
                id="datetime64_NaT_only",
            ),
        ],
    )
    def test_do_categorical_inclusion_keeps_null_pandas_only_dtypes(
        self, column: str, series: Any, values: list[Any], expected_ids: list[int]
    ) -> None:
        """Nullable pandas-only dtypes (Int64/pd.NA, datetime64/NaT) must keep null rows when None is allowed."""
        data = pd.DataFrame({"id": [1, 2, 3, 4, 5], column: series})
        single_filter = SingleFilter(Feature(column), FilterType.CATEGORICAL_INCLUSION, {"values": values})

        result = PandasFilterEngine.do_categorical_inclusion_filter(data, single_filter)

        assert sorted(result["id"].tolist()) == expected_ids

    def test_do_regex_filter_excludes_null_rows(self) -> None:
        """A regex matching "a" must not match null cells that stringify to "nan" via astype(str).

        pandas >= 3.0 defaults ``future.infer_string`` to True, under which ``.astype(str)``
        preserves missing values instead of stringifying them, masking this bug. Pin the option
        explicitly so the test exercises the buggy code path regardless of installed pandas version.
        """
        with pd.option_context("future.infer_string", False):
            data = pd.DataFrame(
                {
                    "id": [1, 2, 3],
                    "name": ["cat", "dog", np.nan],
                }
            )
            feature = Feature("name")
            filter_type = FilterType.REGEX
            parameter = {"value": "a"}
            single_filter = SingleFilter(feature, filter_type, parameter)

            result = PandasFilterEngine.do_regex_filter(data, single_filter)

        assert result["id"].tolist() == [1]
        assert result["name"].tolist() == ["cat"]
