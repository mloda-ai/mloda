import pytest
from typing import Any, Optional, Type
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.parallelization_modes import ParallelizationModes
from mloda_core.abstract_plugins.components.index.index import Index
from tests.test_plugins.compute_framework.test_tooling.dataframe_test_base import DataFrameTestBase

import logging

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    from pandas.testing import assert_series_equal
except ImportError:
    logger.warning("Pandas is not installed. Some tests will be skipped.")
    pd = None
    assert_series_equal = None


@pytest.mark.skipif(pd is None, reason="Pandas is not installed. Skipping this test.")
class TestPandasDataframeComputeFramework:
    pd_dataframe = PandasDataframe(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
    dict_data = {"column1": [1, 2, 3], "column2": [4, 5, 6]}
    expected_data = PandasDataframe.pd_dataframe().from_dict(dict_data)
    left_data = PandasDataframe.pd_dataframe().from_dict({"idx": [1, 3], "col1": ["a", "b"]}).set_index("idx")
    right_data = PandasDataframe.pd_dataframe().from_dict({"idx": [1, 2], "col2": ["x", "z"]}).set_index("idx")
    idx = Index(("idx",))

    def test_expected_data_framework(self) -> None:
        assert self.pd_dataframe.expected_data_framework() == pd.DataFrame

    def test_transform_dict_to_table(self) -> None:
        assert all(self.pd_dataframe.transform(self.dict_data, set()) == self.expected_data)

    def test_transform_arrays(self) -> None:
        data = PandasDataframe.pd_series()([1, 2, 3])
        _pdDf = PandasDataframe(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        _pdDf.set_data(PandasDataframe.pd_dataframe().from_dict({"existing_column": [4, 5, 6]}))

        data = _pdDf.transform(data=data, feature_names={"new_column"})
        assert data.equals(
            PandasDataframe.pd_dataframe().from_dict({"existing_column": [4, 5, 6], "new_column": [1, 2, 3]})
        )

    def test_transform_invalid_data(self) -> None:
        with pytest.raises(ValueError):
            self.pd_dataframe.transform(data=["a"], feature_names=set())

    def test_select_data_by_column_names(self) -> None:
        data = self.pd_dataframe.select_data_by_column_names(self.expected_data, {FeatureName("column1")})
        assert data.columns == ["column1"]

    def test_set_column_names(self) -> None:
        self.pd_dataframe.data = self.expected_data
        self.pd_dataframe.set_column_names()
        assert self.pd_dataframe.column_names == {"column1", "column2"}


@pytest.mark.skipif(pd is None, reason="Pandas is not installed. Skipping this test.")
class TestPandasDataframeMerge(DataFrameTestBase):
    """Test PandasDataframe merge operations using the base test class."""

    @classmethod
    def framework_class(cls) -> Type[Any]:
        """Return the PandasDataframe class."""
        return PandasDataframe

    def create_dataframe(self, data: dict[str, Any]) -> Any:
        """Create a pandas DataFrame from a dictionary."""
        return pd.DataFrame.from_dict(data)

    def get_connection(self) -> Optional[Any]:
        """Return connection object (None for pandas)."""
        return None
