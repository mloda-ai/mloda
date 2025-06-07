from mloda_core.abstract_plugins.components.link import JoinType
import pytest
from unittest.mock import patch
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.parallelization_modes import ParallelizationModes
from mloda_core.abstract_plugins.components.index.index import Index

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

    def test_merge_inner(self) -> None:
        _pdDf = PandasDataframe(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        _pdDf.data = self.left_data
        merge_engine = _pdDf.merge_engine()
        result = merge_engine().merge(_pdDf.data, self.right_data, JoinType.INNER, self.idx, self.idx)
        assert len(result) == 1
        expected = PandasDataframe.pd_merge()(self.left_data, self.right_data, on="idx", how="inner")
        assert result.equals(expected)

    def test_merge_left(self) -> None:
        _pdDf = PandasDataframe(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        _pdDf.data = self.left_data
        merge_engine = _pdDf.merge_engine()
        result = merge_engine().merge(_pdDf.data, self.right_data, JoinType.LEFT, self.idx, self.idx)
        expected = PandasDataframe.pd_merge()(self.left_data, self.right_data, on="idx", how="left")
        assert result.equals(expected)

    def test_merge_right(self) -> None:
        _pdDf = PandasDataframe(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        _pdDf.data = self.left_data
        merge_engine = _pdDf.merge_engine()
        result = merge_engine().merge(_pdDf.data, self.right_data, JoinType.RIGHT, self.idx, self.idx)
        expected = PandasDataframe.pd_merge()(self.left_data, self.right_data, on="idx", how="right")
        assert all(expected["col2"] == result["col2"])
        assert_series_equal(expected["col1"], result["col1"])
        assert result.equals(expected)

    def test_merge_full_outer(self) -> None:
        _pdDf = PandasDataframe(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        _pdDf.data = self.left_data
        merge_engine = _pdDf.merge_engine()
        result = merge_engine().merge(_pdDf.data, self.right_data, JoinType.OUTER, self.idx, self.idx)
        expected = PandasDataframe.pd_merge()(self.left_data, self.right_data, on="idx", how="outer")
        assert result.equals(expected)

    def test_merge_append(self) -> None:
        _pdDf = PandasDataframe(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        _pdDf.data = self.left_data
        merge_engine = _pdDf.merge_engine()
        result = merge_engine().merge(_pdDf.data, self.right_data, JoinType.APPEND, self.idx, self.idx)
        expected = pd.concat([self.left_data, self.right_data], ignore_index=True)
        assert result.equals(expected)

    def test_merge_union(self) -> None:
        _pdDf = PandasDataframe(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        _pdDf.data = self.left_data
        merge_engine = _pdDf.merge_engine()
        result = merge_engine().merge(_pdDf.data, self.right_data, JoinType.UNION, self.idx, self.idx)
        expected = pd.concat([self.left_data, self.right_data], ignore_index=True).drop_duplicates()
        assert result.equals(expected)
