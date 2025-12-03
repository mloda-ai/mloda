import os
from typing import Any, Optional, Type
import pytest
from unittest.mock import patch
from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.parallelization_modes import ParallelizationModes
from mloda_core.abstract_plugins.components.index.index import Index
from tests.test_plugins.compute_framework.test_tooling.dataframe_test_base import DataFrameTestBase

import logging

logger = logging.getLogger(__name__)

try:
    import polars as pl
except ImportError:
    logger.warning("Polars is not installed. Some tests will be skipped.")
    pl = None  # type: ignore


class TestPolarsDataFrameAvailability:
    @patch("builtins.__import__")
    def test_is_available_when_polars_not_installed(self, mock_import: Any) -> None:
        """Test that is_available() returns False when polars import fails."""

        def side_effect(name: Any, *args: Any, **kwargs: Any) -> Any:
            if name == "polars":
                raise ImportError("No module named 'polars'")
            return __import__(name, *args, **kwargs)

        mock_import.side_effect = side_effect
        assert PolarsDataFrame.is_available() is False


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
    if pl:
        pl_dataframe = PolarsDataFrame(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        dict_data = {"column1": [1, 2, 3], "column2": [4, 5, 6]}
        expected_data = PolarsDataFrame.pl_dataframe()(dict_data)
        left_data = PolarsDataFrame.pl_dataframe()({"idx": [1, 3], "col1": ["a", "b"]})
        right_data = PolarsDataFrame.pl_dataframe()({"idx": [1, 2], "col2": ["x", "z"]})
        idx = Index(("idx",))

    def test_expected_data_framework(self) -> None:
        assert self.pl_dataframe.expected_data_framework() == pl.DataFrame

    def test_transform_dict_to_table(self) -> None:
        result = self.pl_dataframe.transform(self.dict_data, set())
        assert result.equals(self.expected_data)

    def test_transform_arrays(self) -> None:
        data = PolarsDataFrame.pl_series()([1, 2, 3])
        _plDf = PolarsDataFrame(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        _plDf.set_data(PolarsDataFrame.pl_dataframe()({"existing_column": [4, 5, 6]}))

        result = _plDf.transform(data=data, feature_names={"new_column"})
        expected = PolarsDataFrame.pl_dataframe()({"existing_column": [4, 5, 6], "new_column": [1, 2, 3]})
        assert result.equals(expected)

    def test_transform_invalid_data(self) -> None:
        with pytest.raises(ValueError):
            self.pl_dataframe.transform(data=["a"], feature_names=set())

    def test_select_data_by_column_names(self) -> None:
        data = self.pl_dataframe.select_data_by_column_names(self.expected_data, {FeatureName("column1")})
        assert data.columns == ["column1"]

    def test_set_column_names(self) -> None:
        self.pl_dataframe.data = self.expected_data
        self.pl_dataframe.set_column_names()
        assert self.pl_dataframe.column_names == {"column1", "column2"}


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
