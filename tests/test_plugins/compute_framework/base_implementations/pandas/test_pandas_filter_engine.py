"""Unit tests for the PandasFilterEngine class."""

from typing import Any

import pytest
import numpy as np
import pandas as pd
import pyarrow as pa

from mloda._testing.compute_framework.filter_suite import FilterEngineSuite
from mloda._testing.framework_adapter import FrameworkAdapter, PandasAdapter
from mloda._testing.typed_data import TypedTestData
from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.user import Feature
from mloda.user import SingleFilter
from mloda.user import FilterType

from mloda_plugins.compute_framework.base_implementations.pandas.pandas_filter_engine import PandasFilterEngine


class TestPandasFilterEngine(FilterEngineSuite):
    """Unit tests for the PandasFilterEngine class using the shared suite."""

    adapter_class = PandasAdapter

    def frame(self, adapter: FrameworkAdapter, name: str, data: TypedTestData) -> Any:
        """Strings use pandas' default inference as in production; NaN/null and decimal survive via ArrowDtype."""
        frame = super().frame(adapter, name, data)
        columns = data.to_python_dict()
        for column, data_type in data.types.items():
            if data_type == DataType.STRING:
                frame[column] = pd.Series(columns[column], index=frame.index)
        if name == "nullable":
            frame["ratio"] = pd.Series(pd.arrays.ArrowExtensionArray(pa.array(columns["ratio"], type=pa.float64())))
        if name == "decimal":
            frame["d"] = pd.Series(columns["d"], dtype=pd.ArrowDtype(pa.decimal128(10, 2)))
        return frame

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
