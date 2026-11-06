import os
from typing import Any
import pytest
from mloda._testing.compute_framework.dtype_suite import DataTypeValidatorSuite, DtypeExtractionSuite
from mloda._testing.compute_framework.frame_suite import AppendColumnsSuite, EmptyResultSuite
from mloda._testing.declarations import Cause
from mloda._testing.framework_adapter import FrameworkAdapter, PolarsAdapter
from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame
from mloda.user import FeatureName
from mloda.user import ParallelizationMode
from tests.test_plugins.compute_framework.test_tooling.availability_test_helper import (
    assert_unavailable_when_import_blocked,
)

import logging

logger = logging.getLogger(__name__)

try:
    import polars as pl
except ImportError:
    logger.warning("Polars is not installed. Some tests will be skipped.")
    pl = None  # type: ignore


class TestPolarsDataFrameAvailability:
    def test_is_available_when_polars_not_installed(self) -> None:
        """Test that is_available() returns False when polars import fails."""
        assert_unavailable_when_import_blocked(PolarsDataFrame, ["polars"])


class TestPolarsInstallation:
    @pytest.mark.skipif(
        os.getenv("SKIP_POLARS_INSTALLATION_TEST", "false").lower() == "true",
        reason="Polars installation test is disabled by environment variable",
    )
    def test_polars_is_installed(self) -> None:
        """Test that Polars is properly installed and can be imported."""
        try:
            import polars as pl

            # Test basic functionality
            df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
            assert len(df) == 3
            assert df.columns == ["a", "b"]
        except ImportError:
            pytest.fail("Polars is not installed but is required for this test environment")


@pytest.mark.skipif(pl is None, reason="Polars is not installed. Skipping this test.")
class TestPolarsDataFrameComputeFramework:
    @pytest.fixture
    def pl_dataframe(self) -> PolarsDataFrame:
        """Create a fresh PolarsDataFrame instance for each test."""
        return PolarsDataFrame(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

    @pytest.fixture
    def expected_data(self, dict_data: dict[str, list[int]]) -> Any:
        """Create fresh expected DataFrame for each test."""
        return PolarsDataFrame.pl_dataframe()(dict_data)

    def test_expected_data_framework(self, pl_dataframe: PolarsDataFrame) -> None:
        assert pl_dataframe.expected_data_framework() == pl.DataFrame

    def test_transform_dict_to_table(
        self, pl_dataframe: PolarsDataFrame, dict_data: dict[str, list[int]], expected_data: Any
    ) -> None:
        result = pl_dataframe.transform(dict_data, [])
        assert result.equals(expected_data)

    def test_transform_arrays(self) -> None:
        data = PolarsDataFrame.pl_series()([1, 2, 3])
        _plDf = PolarsDataFrame(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        _plDf.set_data(PolarsDataFrame.pl_dataframe()({"existing_column": [4, 5, 6]}))

        result = _plDf.transform(data=data, feature_names=["new_column"])
        expected = PolarsDataFrame.pl_dataframe()({"existing_column": [4, 5, 6], "new_column": [1, 2, 3]})
        assert result.equals(expected)

    def test_transform_invalid_data(self, pl_dataframe: PolarsDataFrame) -> None:
        with pytest.raises(ValueError):
            pl_dataframe.transform(data=["a"], feature_names=[])

    def test_select_data_by_column_names(self, pl_dataframe: PolarsDataFrame, expected_data: Any) -> None:
        data = pl_dataframe.select_data_by_column_names(expected_data, [FeatureName("column1")])
        assert data.columns == ["column1"]

    def test_set_column_names(self, pl_dataframe: PolarsDataFrame, expected_data: Any) -> None:
        pl_dataframe.data = expected_data
        pl_dataframe.set_column_names()
        assert pl_dataframe.column_names == {"column1", "column2"}


class TestPolarsDtypeExtractionSuite(DtypeExtractionSuite):
    adapter_class = PolarsAdapter
    switched_off = {
        "test_extract_duplicate_column_dtype_returns_first_occurrence": Cause.BACKEND,
        "test_extract_duplicate_column_data_type_returns_first_occurrence": Cause.BACKEND,
        "test_extract_duplicate_column_dtype_reversed_order_returns_first_occurrence": Cause.BACKEND,
        "test_extract_duplicate_column_data_type_reversed_order_returns_first_occurrence": Cause.BACKEND,
        "test_extract_column_dtype_consistent_with_output_schema": Cause.BACKEND,
    }

    @pytest.mark.skip(reason="Polars frames cannot hold duplicate column names")
    def test_extract_duplicate_column_dtype_returns_first_occurrence(
        self, adapter: FrameworkAdapter, framework_instance: Any
    ) -> None: ...

    @pytest.mark.skip(reason="Polars frames cannot hold duplicate column names")
    def test_extract_duplicate_column_data_type_returns_first_occurrence(
        self, adapter: FrameworkAdapter, framework_instance: Any
    ) -> None: ...

    @pytest.mark.skip(reason="Polars frames cannot hold duplicate column names")
    def test_extract_duplicate_column_dtype_reversed_order_returns_first_occurrence(
        self, adapter: FrameworkAdapter, framework_instance: Any
    ) -> None: ...

    @pytest.mark.skip(reason="Polars frames cannot hold duplicate column names")
    def test_extract_duplicate_column_data_type_reversed_order_returns_first_occurrence(
        self, adapter: FrameworkAdapter, framework_instance: Any
    ) -> None: ...

    @pytest.mark.skip(reason="Polars frames cannot hold duplicate column names")
    def test_extract_column_dtype_consistent_with_output_schema(
        self, adapter: FrameworkAdapter, framework_instance: Any
    ) -> None: ...


class TestPolarsDataTypeValidatorSuite(DataTypeValidatorSuite):
    adapter_class = PolarsAdapter
    unsupported_cases = {
        "timestamp_s_col": (
            Cause.BACKEND,
            "Polars has no second-precision datetime; it widens to ms (TIMESTAMP_MILLIS)",
        ),
    }


class TestPolarsEmptyResultSuite(EmptyResultSuite):
    adapter_class = PolarsAdapter


class TestPolarsAppendColumnsSuite(AppendColumnsSuite):
    adapter_class = PolarsAdapter

    def single_column(self, values: list[Any]) -> Any | None:
        return pl.Series(values)


@pytest.mark.skipif(pl is None, reason="Polars is not installed. Skipping this test.")
class TestPolarsIsMissingColumnError:
    @staticmethod
    def _attribute_error(obj: Any, name: str) -> AttributeError:
        with pytest.raises(AttributeError) as excinfo:
            getattr(obj, name)
        return excinfo.value

    def test_attribute_read_on_eager_frame_is_a_missing_column(self) -> None:
        error = self._attribute_error(pl.DataFrame({"a": [1]}), "x")
        assert PolarsDataFrame.is_missing_column_error(error) is True

    def test_attribute_read_on_lazy_frame_is_a_missing_column(self) -> None:
        error = self._attribute_error(pl.LazyFrame({"a": [1]}), "x")
        assert PolarsDataFrame.is_missing_column_error(error) is True

    def test_misspelled_method_is_not_a_missing_column(self) -> None:
        error = self._attribute_error(pl.DataFrame({"a": [1]}), "fliter")
        assert PolarsDataFrame.is_missing_column_error(error) is False

    def test_declared_column_on_eager_frame_is_not_a_missing_column(self) -> None:
        error = self._attribute_error(pl.DataFrame({"declared": [1]}), "declared")
        assert PolarsDataFrame.is_missing_column_error(error) is False

    def test_declared_column_on_lazy_frame_is_matched(self) -> None:
        """A LazyFrame cannot be checked for columns inside an error hook, so the hedged hint applies."""
        error = self._attribute_error(pl.LazyFrame({"declared": [1]}), "declared")
        assert PolarsDataFrame.is_missing_column_error(error) is True

    def test_plain_attribute_error_with_a_custom_message_is_not_a_missing_column(self) -> None:
        class _Frame(pl.DataFrame):
            def __getattr__(self, name: str) -> Any:
                raise AttributeError(f"`{name}` was removed; use `other` instead", name=name, obj=self)

        error = self._attribute_error(_Frame({"a": [1]}), "melt")
        assert type(error) is AttributeError
        assert PolarsDataFrame.is_missing_column_error(error) is False

    def test_removed_api_error_is_not_a_missing_column(self) -> None:
        error = self._attribute_error(pl.DataFrame({"a": [1]}), "melt")
        assert type(error) is not AttributeError
        assert PolarsDataFrame.is_missing_column_error(error) is False

    def test_attribute_error_on_an_expression_is_not_a_missing_column(self) -> None:
        error = self._attribute_error(pl.col("a"), "x")
        assert PolarsDataFrame.is_missing_column_error(error) is False
