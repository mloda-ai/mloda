import os
from typing import Any, Optional, Type
import pytest
from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame
from mloda.user import FeatureName
from mloda.user import ParallelizationMode
from tests.test_plugins.compute_framework.test_tooling.dataframe_test_base import DataFrameTestBase
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
        result = pl_dataframe.transform(dict_data, set())
        assert result.equals(expected_data)

    def test_transform_arrays(self) -> None:
        data = PolarsDataFrame.pl_series()([1, 2, 3])
        _plDf = PolarsDataFrame(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        _plDf.set_data(PolarsDataFrame.pl_dataframe()({"existing_column": [4, 5, 6]}))

        result = _plDf.transform(data=data, feature_names={"new_column"})
        expected = PolarsDataFrame.pl_dataframe()({"existing_column": [4, 5, 6], "new_column": [1, 2, 3]})
        assert result.equals(expected)

    def test_transform_invalid_data(self, pl_dataframe: PolarsDataFrame) -> None:
        with pytest.raises(ValueError):
            pl_dataframe.transform(data=["a"], feature_names=set())

    def test_select_data_by_column_names(self, pl_dataframe: PolarsDataFrame, expected_data: Any) -> None:
        data = pl_dataframe.select_data_by_column_names(expected_data, {FeatureName("column1")})
        assert data.columns == ["column1"]

    def test_set_column_names(self, pl_dataframe: PolarsDataFrame, expected_data: Any) -> None:
        pl_dataframe.data = expected_data
        pl_dataframe.set_column_names()
        assert pl_dataframe.column_names == {"column1", "column2"}


@pytest.mark.skipif(pl is None, reason="Polars is not installed. Skipping this test.")
class TestPolarsDataFrameMerge(DataFrameTestBase):
    """Test PolarsDataFrame merge operations using the base test class."""

    @classmethod
    def framework_class(cls) -> Type[Any]:
        """Return the PolarsDataFrame class."""
        return PolarsDataFrame

    def create_dataframe(self, data: dict[str, Any]) -> Any:
        """Create a polars DataFrame from a dictionary."""
        return pl.DataFrame(data)

    def get_connection(self) -> Optional[Any]:
        """Return connection object (None for polars)."""
        return None
