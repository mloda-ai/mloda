import pytest
from typing import Any, Optional, Type
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda.user import FeatureName
from mloda.user import ParallelizationMode
from tests.test_plugins.compute_framework.test_tooling.dataframe_test_base import DataFrameTestBase

import logging

logger = logging.getLogger(__name__)

try:
    import pandas as pd
except ImportError:
    logger.warning("Pandas is not installed. Some tests will be skipped.")
    pd = None


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
        assert all(pd_dataframe.transform(dict_data, set()) == expected_data)

    def test_transform_arrays(self) -> None:
        data = PandasDataFrame.pd_series()([1, 2, 3])
        _pdDf = PandasDataFrame(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        _pdDf.set_data(PandasDataFrame.pd_dataframe().from_dict({"existing_column": [4, 5, 6]}))

        data = _pdDf.transform(data=data, feature_names={"new_column"})
        assert data.equals(
            PandasDataFrame.pd_dataframe().from_dict({"existing_column": [4, 5, 6], "new_column": [1, 2, 3]})
        )

    def test_transform_invalid_data(self, pd_dataframe: PandasDataFrame) -> None:
        with pytest.raises(ValueError):
            pd_dataframe.transform(data=["a"], feature_names=set())

    def test_select_data_by_column_names(self, pd_dataframe: PandasDataFrame, expected_data: Any) -> None:
        data = pd_dataframe.select_data_by_column_names(expected_data, {FeatureName("column1")})
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

        result = pdf.transform(data, {"content"})

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

        result = pdf.transform(data, {"content"})

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert result["content"].iloc[0] == "first"
        assert result["content"].iloc[1] == "second"


@pytest.mark.skipif(pd is None, reason="Pandas is not installed. Skipping this test.")
class TestPandasDataFrameMerge(DataFrameTestBase):
    """Test PandasDataFrame merge operations using the base test class."""

    @classmethod
    def framework_class(cls) -> Type[Any]:
        """Return the PandasDataFrame class."""
        return PandasDataFrame

    def create_dataframe(self, data: dict[str, Any]) -> Any:
        """Create a pandas DataFrame from a dictionary."""
        return pd.DataFrame.from_dict(data)

    def get_connection(self) -> Optional[Any]:
        """Return connection object (None for pandas)."""
        return None
