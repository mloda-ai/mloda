import os
from mloda_core.abstract_plugins.components.link import JoinType
import pytest
from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataframe
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.parallelization_modes import ParallelizationModes
from mloda_core.abstract_plugins.components.index.index import Index

import logging

logger = logging.getLogger(__name__)

try:
    import polars as pl
except ImportError:
    logger.warning("Polars is not installed. Some tests will be skipped.")
    pl = None  # type: ignore


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
class TestPolarsDataframeComputeFramework:
    pl_dataframe = PolarsDataframe(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
    dict_data = {"column1": [1, 2, 3], "column2": [4, 5, 6]}
    expected_data = PolarsDataframe.pl_dataframe()(dict_data)
    left_data = PolarsDataframe.pl_dataframe()({"idx": [1, 3], "col1": ["a", "b"]})
    right_data = PolarsDataframe.pl_dataframe()({"idx": [1, 2], "col2": ["x", "z"]})
    idx = Index(("idx",))

    def test_expected_data_framework(self) -> None:
        assert self.pl_dataframe.expected_data_framework() == pl.DataFrame

    def test_transform_dict_to_table(self) -> None:
        result = self.pl_dataframe.transform(self.dict_data, set())
        assert result.equals(self.expected_data)

    def test_transform_arrays(self) -> None:
        data = PolarsDataframe.pl_series()([1, 2, 3])
        _plDf = PolarsDataframe(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        _plDf.set_data(PolarsDataframe.pl_dataframe()({"existing_column": [4, 5, 6]}))

        result = _plDf.transform(data=data, feature_names={"new_column"})
        expected = PolarsDataframe.pl_dataframe()({"existing_column": [4, 5, 6], "new_column": [1, 2, 3]})
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

    def test_merge_inner(self) -> None:
        _plDf = PolarsDataframe(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        _plDf.data = self.left_data
        merge_engine = _plDf.merge_engine()
        result = merge_engine().merge(_plDf.data, self.right_data, JoinType.INNER, self.idx, self.idx)
        assert len(result) == 1
        expected = self.left_data.join(self.right_data, on="idx", how="inner")
        assert result.equals(expected)

    def test_merge_left(self) -> None:
        _plDf = PolarsDataframe(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        _plDf.data = self.left_data
        merge_engine = _plDf.merge_engine()
        result = merge_engine().merge(_plDf.data, self.right_data, JoinType.LEFT, self.idx, self.idx)
        expected = self.left_data.join(self.right_data, on="idx", how="left")
        assert result.equals(expected)

    def test_merge_right(self) -> None:
        _plDf = PolarsDataframe(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        _plDf.data = self.left_data
        merge_engine = _plDf.merge_engine()
        result = merge_engine().merge(_plDf.data, self.right_data, JoinType.RIGHT, self.idx, self.idx)
        expected = self.left_data.join(self.right_data, on="idx", how="right")

        # Reorder columns to match and sort by all columns to ensure consistent ordering for comparison
        column_order = ["idx", "col1", "col2"]
        result_reordered = result.select(column_order).sort(column_order)
        expected_reordered = expected.select(column_order).sort(column_order)
        assert result_reordered.equals(expected_reordered)

    def test_merge_full_outer(self) -> None:
        _plDf = PolarsDataframe(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        _plDf.data = self.left_data
        merge_engine = _plDf.merge_engine()
        result = merge_engine().merge(_plDf.data, self.right_data, JoinType.OUTER, self.idx, self.idx)

        # Get the native result and properly coalesce it like our implementation does
        native_raw = self.left_data.join(self.right_data, on="idx", how="full")
        expected = native_raw.with_columns(
            pl.when(pl.col("idx").is_null()).then(pl.col("idx_right")).otherwise(pl.col("idx")).alias("idx")
        ).drop("idx_right")

        # Reorder columns to match and sort by all columns to ensure consistent ordering for comparison
        column_order = ["idx", "col1", "col2"]
        result_reordered = result.select(column_order).sort(column_order)
        expected_reordered = expected.select(column_order).sort(column_order)
        assert result_reordered.equals(expected_reordered)

    def test_merge_append(self) -> None:
        _plDf = PolarsDataframe(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        _plDf.data = self.left_data
        merge_engine = _plDf.merge_engine()
        result = merge_engine().merge(_plDf.data, self.right_data, JoinType.APPEND, self.idx, self.idx)
        expected = pl.concat([self.left_data, self.right_data], how="diagonal")
        assert result.equals(expected)

    def test_merge_union(self) -> None:
        _plDf = PolarsDataframe(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        _plDf.data = self.left_data
        merge_engine = _plDf.merge_engine()
        result = merge_engine().merge(_plDf.data, self.right_data, JoinType.UNION, self.idx, self.idx)
        expected = pl.concat([self.left_data, self.right_data], how="diagonal").unique()
        # Sort both by all columns to ensure consistent ordering for comparison
        result_sorted = result.sort(["idx", "col1", "col2"])
        expected_sorted = expected.sort(["idx", "col1", "col2"])
        assert result_sorted.equals(expected_sorted)
