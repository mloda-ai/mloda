from typing import Any
from mloda_core.abstract_plugins.components.link import JoinType
import pytest
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.parallelization_modes import ParallelizationModes
from mloda_core.abstract_plugins.components.index.index import Index
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


class TestPolarsLazyDataFrameAvailability:
    def test_is_available_when_polars_not_installed(self) -> None:
        """Test that is_available() returns False when polars import fails."""
        assert_unavailable_when_import_blocked(PolarsLazyDataFrame, ["polars"])


@pytest.mark.skipif(pl is None, reason="Polars is not installed. Skipping this test.")
class TestPolarsLazyDataFrameComputeFramework:
    if pl:
        lazy_df = PolarsLazyDataFrame(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        dict_data = {"column1": [1, 2, 3], "column2": [4, 5, 6]}
        expected_lazy_data = PolarsLazyDataFrame.pl_lazy_frame()(dict_data)
        left_data = PolarsLazyDataFrame.pl_lazy_frame()({"idx": [1, 3], "col1": ["a", "b"]})
        right_data = PolarsLazyDataFrame.pl_lazy_frame()({"idx": [1, 2], "col2": ["x", "z"]})
        idx = Index(("idx",))

    def test_expected_data_framework(self) -> None:
        assert self.lazy_df.expected_data_framework() == pl.LazyFrame

    def test_transform_dict_to_lazy_frame(self) -> None:
        result = self.lazy_df.transform(self.dict_data, set())
        assert isinstance(result, pl.LazyFrame)
        # Verify it's actually lazy (no execution yet) by checking schema
        assert set(result.collect_schema().names()) == {"column1", "column2"}
        # Verify results are correct when collected
        collected = result.collect()
        expected = pl.DataFrame(self.dict_data)
        assert collected.equals(expected)

    def test_transform_arrays(self) -> None:
        data = PolarsLazyDataFrame.pl_series()([1, 2, 3])
        _lazy_df = PolarsLazyDataFrame(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        _lazy_df.set_data(PolarsLazyDataFrame.pl_lazy_frame()({"existing_column": [4, 5, 6]}))

        result = _lazy_df.transform(data=data, feature_names={"new_column"})
        assert isinstance(result, pl.LazyFrame)

        # Verify schema without executing
        schema_names = set(result.collect_schema().names())
        assert schema_names == {"existing_column", "new_column"}

        # Verify results when collected
        collected = result.collect()
        expected = pl.DataFrame({"existing_column": [4, 5, 6], "new_column": [1, 2, 3]})
        assert collected.equals(expected)

    def test_transform_dataframe_to_lazy(self) -> None:
        """Test converting DataFrame to LazyFrame"""
        df = pl.DataFrame({"col": [1, 2, 3]})
        result = self.lazy_df.transform(df, set())
        assert isinstance(result, pl.LazyFrame)
        # Verify schema
        assert set(result.collect_schema().names()) == {"col"}

    def test_transform_invalid_data(self) -> None:
        with pytest.raises(ValueError):
            self.lazy_df.transform(data=["a"], feature_names=set())

    def test_set_column_names(self) -> None:
        self.lazy_df.data = self.expected_lazy_data
        self.lazy_df.set_column_names()
        assert self.lazy_df.column_names == {"column1", "column2"}

    def test_set_column_names_invalid_data(self) -> None:
        """Test error when data doesn't have collect_schema method"""
        self.lazy_df.data = "invalid_data"
        with pytest.raises(ValueError, match="Data does not have a collect_schema method"):
            self.lazy_df.set_column_names()

    def test_merge_inner(self) -> None:
        _lazy_df = PolarsLazyDataFrame(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        _lazy_df.data = self.left_data
        merge_engine = _lazy_df.merge_engine()
        result = merge_engine().merge(_lazy_df.data, self.right_data, JoinType.INNER, self.idx, self.idx)
        assert isinstance(result, pl.LazyFrame)

        # Verify results when collected
        collected = result.collect()
        expected = self.left_data.join(self.right_data, on="idx", how="inner").collect()
        assert len(collected) == 1
        assert collected.equals(expected)

    def test_merge_left(self) -> None:
        _lazy_df = PolarsLazyDataFrame(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        _lazy_df.data = self.left_data
        merge_engine = _lazy_df.merge_engine()
        result = merge_engine().merge(_lazy_df.data, self.right_data, JoinType.LEFT, self.idx, self.idx)
        assert isinstance(result, pl.LazyFrame)

        # Verify results when collected
        collected = result.collect()
        expected = self.left_data.join(self.right_data, on="idx", how="left").collect()
        assert collected.equals(expected)

    def test_merge_append(self) -> None:
        _lazy_df = PolarsLazyDataFrame(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        _lazy_df.data = self.left_data
        merge_engine = _lazy_df.merge_engine()
        result = merge_engine().merge(_lazy_df.data, self.right_data, JoinType.APPEND, self.idx, self.idx)
        assert isinstance(result, pl.LazyFrame)

        # Verify results when collected
        collected = result.collect()
        expected = pl.concat([self.left_data, self.right_data], how="diagonal").collect()
        assert collected.equals(expected)


@pytest.mark.skipif(pl is None, reason="Polars is not installed. Skipping this test.")
class TestLazyEagerEquivalence:
    """Test that lazy and eager frameworks produce identical results"""

    if pl:
        dict_data = {"column1": [1, 2, 3], "column2": [4, 5, 6]}

    def test_transform_dict_equivalence(self) -> None:
        """Test that lazy and eager produce same results for dict transformation"""
        from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame

        # Eager transformation
        eager_df = PolarsDataFrame(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        eager_result = eager_df.transform(self.dict_data, set())

        # Lazy transformation
        lazy_df = PolarsLazyDataFrame(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        lazy_result = lazy_df.transform(self.dict_data, set())

        # Compare collected lazy result with eager result
        assert lazy_result.collect().equals(eager_result)

    def test_select_columns_equivalence(self) -> None:
        """Test that lazy and eager produce same results for column selection"""
        from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame

        # Setup data
        df_data = pl.DataFrame(self.dict_data)
        lazy_data = df_data.lazy()
        feature_names = {FeatureName("column1")}

        # Eager selection
        eager_df = PolarsDataFrame(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        eager_result = eager_df.select_data_by_column_names(df_data, feature_names)

        # Lazy selection
        lazy_df = PolarsLazyDataFrame(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        lazy_result = lazy_df.select_data_by_column_names(lazy_data, feature_names)

        # Compare results
        assert lazy_result.equals(eager_result)


@pytest.mark.skipif(pl is None, reason="Polars is not installed. Skipping this test.")
class TestLazyExecutionTiming:
    """Test that operations are actually deferred in lazy mode"""

    if pl:
        dict_data = {"column1": [1, 2, 3], "column2": [4, 5, 6]}

    def test_schema_access_without_execution(self) -> None:
        """Test that we can access schema without triggering execution"""
        lazy_df = PolarsLazyDataFrame(mode=ParallelizationModes.SYNC, children_if_root=frozenset())

        # Create lazy frame
        result = lazy_df.transform(self.dict_data, set())

        # Should be able to get schema without execution
        schema = result.collect_schema()
        assert set(schema.names()) == {"column1", "column2"}

        # Result should still be lazy
        assert isinstance(result, pl.LazyFrame)
