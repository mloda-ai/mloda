import logging
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from mloda._testing.compute_framework.dtype_suite import DataTypeValidatorSuite, DtypeExtractionSuite
from mloda._testing.compute_framework.frame_suite import AppendColumnsSuite, EmptyResultSuite
from mloda._testing.framework_adapter import PandasAdapter
from mloda.provider import Cause
from mloda.user import DataType, FeatureName, ParallelizationMode
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from tests.helpers.availability_test_helper import (
    assert_unavailable_when_import_blocked,
)

logger = logging.getLogger(__name__)

try:
    import pandas as pd
except ImportError:
    logger.warning("Pandas is not installed. Some tests will be skipped.")
    pd = None


class TestPandasDataFrameAvailability:
    def test_is_available_when_pandas_not_installed(self) -> None:
        """Test that is_available() returns False when pandas import fails."""
        assert_unavailable_when_import_blocked(PandasDataFrame, ["pandas"])


@pytest.mark.skipif(pd is None, reason="Pandas is not installed. Skipping this test.")
class TestPandasDataFrameComputeFramework:
    @pytest.fixture
    def pd_dataframe(self) -> PandasDataFrame:
        """Create a fresh PandasDataFrame instance for each test."""
        return PandasDataFrame(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

    @pytest.fixture
    def expected_data(self, dict_data: dict[str, list[int]]) -> Any:
        """Create fresh expected DataFrame for each test."""
        return PandasDataFrame.pd_dataframe().from_dict(dict_data)

    def test_expected_data_framework(self, pd_dataframe: PandasDataFrame) -> None:
        assert pd_dataframe.expected_data_framework() == pd.DataFrame

    def test_transform_dict_to_table(
        self, pd_dataframe: PandasDataFrame, dict_data: dict[str, list[int]], expected_data: Any
    ) -> None:
        assert all(pd_dataframe.transform(dict_data, []) == expected_data)

    def test_transform_arrays(self) -> None:
        data = PandasDataFrame.pd_series()([1, 2, 3])
        _pdDf = PandasDataFrame(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        _pdDf.set_data(PandasDataFrame.pd_dataframe().from_dict({"existing_column": [4, 5, 6]}))

        data = _pdDf.transform(data=data, feature_names=["new_column"])
        assert data.equals(
            PandasDataFrame.pd_dataframe().from_dict({"existing_column": [4, 5, 6], "new_column": [1, 2, 3]})
        )

    def test_transform_invalid_data(self, pd_dataframe: PandasDataFrame) -> None:
        with pytest.raises(ValueError):
            pd_dataframe.transform(data=["a"], feature_names=[])

    def test_select_data_by_column_names(self, pd_dataframe: PandasDataFrame, expected_data: Any) -> None:
        data = pd_dataframe.select_data_by_column_names(expected_data, [FeatureName("column1")])
        assert data.columns == ["column1"]

    def test_set_column_names(self, pd_dataframe: PandasDataFrame, expected_data: Any) -> None:
        pd_dataframe.data = expected_data
        pd_dataframe.set_column_names()
        assert pd_dataframe.column_names == {"column1", "column2"}


@pytest.mark.skipif(pd is None, reason="Pandas is not installed. Skipping this test.")
class TestPandasTransformList:
    def test_transform_list_of_dicts(self) -> None:
        """PandasDataFrame should handle list of dicts (document reader output)."""
        pdf = PandasDataFrame(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        data = [{"content": "hello world", "source": "/path/to/file.txt", "file_type": "text"}]

        result = pdf.transform(data, ["content"])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert "content" in result.columns
        assert result["content"].iloc[0] == "hello world"
        assert result["source"].iloc[0] == "/path/to/file.txt"

    def test_transform_list_of_multiple_dicts(self) -> None:
        """PandasDataFrame should handle multiple dicts in list."""
        pdf = PandasDataFrame(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        data = [
            {"content": "first", "source": "/path/a.json", "file_type": "json"},
            {"content": "second", "source": "/path/b.json", "file_type": "json"},
        ]

        result = pdf.transform(data, ["content"])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert result["content"].iloc[0] == "first"
        assert result["content"].iloc[1] == "second"


class TestPandasDtypeExtractionSuite(DtypeExtractionSuite):
    adapter_class = PandasAdapter


@pytest.mark.skipif(pd is None, reason="Pandas is not installed. Skipping this test.")
class TestPandasDtypeExtraction:
    @pytest.fixture
    def framework_instance(self) -> Any:
        return PandasDataFrame(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

    @pytest.mark.parametrize(
        "values,expected",
        [
            (["a", "b"], DataType.STRING),
            (["a", None, "b"], DataType.STRING),
            (["a", np.nan, "b"], DataType.STRING),
            (["a", 1], None),
            ([None, None], None),
            ([], None),
            ([Decimal("12.34"), Decimal("5.50"), Decimal("99.99"), None], None),
            ([b"a", b"b"], None),
        ],
        ids=["strings", "strings_none", "strings_nan", "mixed", "all_none", "empty", "decimal", "bytes"],
    )
    def test_extract_object_column_data_type(
        self, framework_instance: Any, values: list[Any], expected: DataType | None
    ) -> None:
        data = pd.DataFrame({"d": pd.Series(values, dtype=object)})
        assert framework_instance._extract_column_data_type(data, "d") == expected

    def test_extract_duplicate_names_reads_first_object_string_column(self, framework_instance: Any) -> None:
        data = pd.DataFrame({"x": pd.Series(["a", "b"], dtype=object), "y": pd.Series([1, 2], dtype="int64")})
        data.columns = ["a", "a"]
        assert framework_instance._extract_column_data_type(data, "a") == DataType.STRING


_NO_LIST_ELEMENT_TYPE = (Cause.BACKEND, "to_pandas yields object columns for lists, which carry no element type")


class TestPandasDataTypeValidatorSuite(DataTypeValidatorSuite):
    adapter_class = PandasAdapter
    unsupported_cases = {
        "list_str_col": _NO_LIST_ELEMENT_TYPE,
        "list_struct_col": _NO_LIST_ELEMENT_TYPE,
        "list_fixed_col": _NO_LIST_ELEMENT_TYPE,
    }


@pytest.mark.skipif(pd is None, reason="Pandas is not installed. Skipping this test.")
class TestPandasDataTypeValidator:
    @pytest.fixture
    def framework_instance(self) -> Any:
        return PandasDataFrame(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

    @pytest.mark.parametrize(
        "values,arrow_type,expected",
        [
            ([["a"], ["b", "c"]], pa.list_(pa.string()), DataType.LIST),
            ([datetime(2024, 1, 1, tzinfo=UTC)], pa.timestamp("us", tz="UTC"), DataType.TIMESTAMP_MICROS_TZ),
            ([date(2024, 1, 1)], pa.date32(), DataType.DATE),
        ],
        ids=["list", "timestamp_tz", "date32"],
    )
    def test_arrow_dtype_column_resolves_through_arrow_mapping(
        self, framework_instance: Any, values: list[Any], arrow_type: pa.DataType, expected: DataType
    ) -> None:
        df = pd.DataFrame({"c": pd.Series(values, dtype=pd.ArrowDtype(arrow_type))})
        assert framework_instance._extract_column_data_type(df, "c") == expected


class TestPandasEmptyResultSuite(EmptyResultSuite):
    adapter_class = PandasAdapter


class TestPandasAppendColumnsSuite(AppendColumnsSuite):
    adapter_class = PandasAdapter

    def single_column(self, values: list[Any]) -> Any | None:
        return pd.Series(values)


@pytest.mark.skipif(pd is None, reason="Pandas is not installed. Skipping this test.")
class TestPandasIsMissingColumnError:
    def test_local_variable_undefined_is_not_a_missing_column(self) -> None:
        from pandas.errors import UndefinedVariableError

        error = UndefinedVariableError("@x", is_local=True)
        assert str(error).startswith("local variable")
        assert PandasDataFrame.is_missing_column_error(error) is False

    def test_attribute_read_of_an_unknown_name_is_a_missing_column(self) -> None:
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(AttributeError) as excinfo:
            df.x  # noqa: B018
        assert PandasDataFrame.is_missing_column_error(excinfo.value) is True

    def test_misspelled_method_is_not_a_missing_column(self) -> None:
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(AttributeError) as excinfo:
            df.fliter  # noqa: B018
        assert PandasDataFrame.is_missing_column_error(excinfo.value) is False

    @pytest.mark.parametrize("name", ["value", "date"])
    def test_short_name_near_a_method_is_a_missing_column(self, name: str) -> None:
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(AttributeError) as excinfo:
            getattr(df, name)
        assert PandasDataFrame.is_missing_column_error(excinfo.value) is True

    def test_private_attribute_is_not_a_missing_column(self) -> None:
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(AttributeError) as excinfo:
            df._private  # noqa: B018
        assert PandasDataFrame.is_missing_column_error(excinfo.value) is False

    def test_attribute_error_on_a_series_is_not_a_missing_column(self) -> None:
        series = pd.Series([1])
        with pytest.raises(AttributeError) as excinfo:
            series.x  # noqa: B018
        assert PandasDataFrame.is_missing_column_error(excinfo.value) is False

    def test_bare_attribute_error_is_not_a_missing_column(self) -> None:
        assert PandasDataFrame.is_missing_column_error(AttributeError("x")) is False


def _pandas_framework() -> PandasDataFrame:
    return PandasDataFrame(mode=ParallelizationMode.SYNC, children_if_root=frozenset())


_PANDAS_LENIENT = [
    ("Int16", "INT32"),
    ("float16", "FLOAT"),
]


@pytest.mark.parametrize("dtype, expected", _PANDAS_LENIENT)
def test_extract_column_data_type_folds_inexact_numpy_types(dtype: str, expected: str) -> None:
    df = pd.DataFrame({"c": pd.Series([1, 2], dtype=dtype)})
    assert _pandas_framework()._extract_column_data_type(df, "c") == DataType[expected]


_WIDE_LONGDOUBLE = np.dtype(np.longdouble).itemsize > 8


@pytest.mark.parametrize(
    "series, expected",
    [
        pytest.param(
            pd.Series(np.array([1.5], dtype=np.longdouble)),
            DataType.DOUBLE,
            marks=pytest.mark.skipif(not _WIDE_LONGDOUBLE, reason="longdouble is 8 bytes on this platform"),
            id="longdouble",
        ),
        pytest.param(pd.Series(pd.Categorical([True, False])), DataType.BOOLEAN, id="categorical_bool"),
        pytest.param(pd.Series(pd.arrays.SparseArray([True, False])), DataType.BOOLEAN, id="sparse_bool"),
    ],
)
def test_extract_column_data_type_keeps_pandas_predicate_folds(series: Any, expected: DataType) -> None:
    assert _pandas_framework()._extract_column_data_type(pd.DataFrame({"c": series}), "c") == expected


@pytest.mark.parametrize(
    "arrow_type, expected",
    [
        (pa.large_string(), DataType.STRING),
    ],
    ids=["large_string"],
)
def test_extract_arrow_dtype_column_folds_inexact_types(arrow_type: pa.DataType, expected: DataType) -> None:
    df = pd.DataFrame({"c": pd.Series(pa.array([], type=arrow_type), dtype=pd.ArrowDtype(arrow_type))})
    assert _pandas_framework()._extract_column_data_type(df, "c") == expected


_PANDAS_EXACT: list[tuple[Any, DataType | None]] = [
    (pd.Series([1], dtype="int16"), None),
    (pd.Series([1], dtype="uint32"), None),
    (pd.Series([1], dtype="float16"), None),
    (pd.Series([1], dtype="int32"), DataType.INT32),
    (pd.Series([1], dtype="Int64"), DataType.INT64),
    (pd.Series([1.5], dtype="float32"), DataType.FLOAT),
    (pd.Series([True], dtype="bool"), DataType.BOOLEAN),
    (pd.Series(["a"], dtype="string"), DataType.STRING),
    (pd.Series([Decimal("1.5")]), None),
    (pd.Series([datetime(2024, 1, 1)], dtype="datetime64[ms]"), DataType.TIMESTAMP_MILLIS),
    (pd.Series([datetime(2024, 1, 1, tzinfo=UTC)]), DataType.TIMESTAMP_MICROS_TZ),
    (pd.Series([1], dtype=pd.ArrowDtype(pa.int16())), None),
    (pd.Series(["a"], dtype=pd.ArrowDtype(pa.large_string())), None),
    (pd.Series([1], dtype=pd.ArrowDtype(pa.int64())), DataType.INT64),
    (pd.Series([[1]], dtype=pd.ArrowDtype(pa.list_(pa.int64()))), DataType.LIST),
]


@pytest.mark.parametrize("series, expected", _PANDAS_EXACT)
def test_exact_column_data_type(series: Any, expected: DataType | None) -> None:
    assert _pandas_framework()._exact_column_data_type(pd.DataFrame({"c": series}), "c") == expected


def test_exact_column_data_type_missing_column_is_none() -> None:
    assert _pandas_framework()._exact_column_data_type(pd.DataFrame({"c": [1]}), "nope") is None


def test_exact_column_data_type_sparse_column_does_not_raise() -> None:
    df = pd.DataFrame({"c": pd.arrays.SparseArray([0, 0, 1])})
    assert _pandas_framework()._exact_column_data_type(df, "c") is None
