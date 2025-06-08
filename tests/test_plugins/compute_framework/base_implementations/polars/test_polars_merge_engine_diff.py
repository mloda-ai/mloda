import pytest
from mloda_core.abstract_plugins.components.link import JoinType
from mloda_plugins.compute_framework.base_implementations.polars.polars_merge_engine import PolarsMergeEngine
from mloda_plugins.compute_framework.base_implementations.polars.polars_lazy_merge_engine import PolarsLazyMergeEngine
from mloda_core.abstract_plugins.components.index.index import Index

import logging

logger = logging.getLogger(__name__)

try:
    import polars as pl
except ImportError:
    logger.warning("Polars is not installed. Some tests will be skipped.")
    pl = None  # type: ignore


@pytest.mark.skipif(pl is None, reason="Polars is not installed. Skipping this test.")
class TestPolarsMergeEngineDifferences:
    """Test the specific differences between eager and lazy merge engines."""

    if pl:
        test_data_left = {"idx": [1, 3], "col1": ["a", "b"]}
        test_data_right = {"idx": [1, 2], "col2": ["x", "z"]}
        idx = Index(("idx",))

    def test_get_column_names_difference(self) -> None:
        """Test that get_column_names works differently for DataFrame vs LazyFrame."""
        eager_engine = PolarsMergeEngine()
        lazy_engine = PolarsLazyMergeEngine()

        # Create DataFrame and LazyFrame with same data
        df = pl.DataFrame(self.test_data_left)
        lazy_df = df.lazy()

        # Both should return the same column names but use different methods
        eager_cols = eager_engine.get_column_names(df)
        lazy_cols = lazy_engine.get_column_names(lazy_df)

        assert eager_cols == lazy_cols == ["idx", "col1"]

    def test_is_empty_data_difference(self) -> None:
        """Test that is_empty_data behaves differently for DataFrame vs LazyFrame."""
        eager_engine = PolarsMergeEngine()
        lazy_engine = PolarsLazyMergeEngine()

        # Create empty DataFrame and LazyFrame
        empty_df = pl.DataFrame({"col": []})
        empty_lazy_df = empty_df.lazy()

        # Eager engine should detect empty data
        assert eager_engine.is_empty_data(empty_df) is True

        # Lazy engine should always return False (skip empty checking)
        assert lazy_engine.is_empty_data(empty_lazy_df) is False

    def test_column_exists_in_result_difference(self) -> None:
        """Test that column_exists_in_result uses different methods."""
        eager_engine = PolarsMergeEngine()
        lazy_engine = PolarsLazyMergeEngine()

        # Create DataFrame and LazyFrame with same data
        df = pl.DataFrame(self.test_data_left)
        lazy_df = df.lazy()

        # Both should detect existing columns but use different methods
        assert eager_engine.column_exists_in_result(df, "idx") is True
        assert eager_engine.column_exists_in_result(df, "nonexistent") is False

        assert lazy_engine.column_exists_in_result(lazy_df, "idx") is True
        assert lazy_engine.column_exists_in_result(lazy_df, "nonexistent") is False

    def test_handle_empty_data_difference(self) -> None:
        """Test that handle_empty_data behaves differently."""
        eager_engine = PolarsMergeEngine()
        lazy_engine = PolarsLazyMergeEngine()

        # Create empty DataFrame and LazyFrame
        empty_df = pl.DataFrame({"idx": []})
        non_empty_df = pl.DataFrame(self.test_data_left)
        empty_lazy_df = empty_df.lazy()
        non_empty_lazy_df = non_empty_df.lazy()

        # Eager engine should handle empty data
        eager_result = eager_engine.handle_empty_data(empty_df, non_empty_df, "idx", "idx")
        assert eager_result is not None  # Should return a DataFrame

        # Lazy engine should skip empty data handling
        lazy_result = lazy_engine.handle_empty_data(empty_lazy_df, non_empty_lazy_df, "idx", "idx")
        assert lazy_result is None  # Should return None (skip handling)

    def test_join_logic_equivalence(self) -> None:
        """Test that join_logic produces equivalent results despite different implementations."""
        eager_engine = PolarsMergeEngine()
        lazy_engine = PolarsLazyMergeEngine()

        # Create equivalent data
        left_df = pl.DataFrame(self.test_data_left)
        right_df = pl.DataFrame(self.test_data_right)
        left_lazy = left_df.lazy()
        right_lazy = right_df.lazy()

        # Test inner join
        eager_result = eager_engine.join_logic("inner", left_df, right_df, self.idx, self.idx, JoinType.INNER)
        lazy_result = lazy_engine.join_logic("inner", left_lazy, right_lazy, self.idx, self.idx, JoinType.INNER)

        # Results should be equivalent when lazy is collected
        assert lazy_result.collect().equals(eager_result)

    def test_merge_methods_equivalence(self) -> None:
        """Test that all merge methods produce equivalent results."""
        eager_engine = PolarsMergeEngine()
        lazy_engine = PolarsLazyMergeEngine()

        # Create equivalent data
        left_df = pl.DataFrame(self.test_data_left)
        right_df = pl.DataFrame(self.test_data_right)
        left_lazy = left_df.lazy()
        right_lazy = right_df.lazy()

        # Test different join types
        join_methods = [
            ("merge_inner", JoinType.INNER),
            ("merge_left", JoinType.LEFT),
            ("merge_append", JoinType.APPEND),
        ]

        for method_name, join_type in join_methods:
            eager_method = getattr(eager_engine, method_name)
            lazy_method = getattr(lazy_engine, method_name)

            eager_result = eager_method(left_df, right_df, self.idx, self.idx)
            lazy_result = lazy_method(left_lazy, right_lazy, self.idx, self.idx)

            # Results should be equivalent when lazy is collected
            if hasattr(lazy_result, "collect"):
                lazy_collected = lazy_result.collect()
            else:
                lazy_collected = lazy_result

            assert lazy_collected.equals(eager_result), f"Results differ for {method_name}"


@pytest.mark.skipif(pl is None, reason="Polars is not installed. Skipping this test.")
class TestRefactoredMergeEngineIntegration:
    """Test that the refactored merge engines work correctly with the compute frameworks."""

    if pl:
        test_data = {"column1": [1, 2, 3], "column2": [4, 5, 6]}

    def test_eager_framework_uses_eager_merge_engine(self) -> None:
        """Test that PolarsDataframe uses PolarsMergeEngine."""
        from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataframe
        from mloda_core.abstract_plugins.components.parallelization_modes import ParallelizationModes

        df = PolarsDataframe(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        merge_engine_class = df.merge_engine()

        assert merge_engine_class == PolarsMergeEngine

    def test_lazy_framework_uses_lazy_merge_engine(self) -> None:
        """Test that PolarsLazyDataframe uses PolarsLazyMergeEngine."""
        from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataframe
        from mloda_core.abstract_plugins.components.parallelization_modes import ParallelizationModes

        lazy_df = PolarsLazyDataframe(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        merge_engine_class = lazy_df.merge_engine()

        assert merge_engine_class == PolarsLazyMergeEngine

    def test_inheritance_chain(self) -> None:
        """Test that PolarsLazyMergeEngine properly inherits from PolarsMergeEngine."""
        assert issubclass(PolarsLazyMergeEngine, PolarsMergeEngine)

        # Test that lazy engine has access to parent methods
        lazy_engine = PolarsLazyMergeEngine()
        assert hasattr(lazy_engine, "merge_inner")
        assert hasattr(lazy_engine, "merge_left")
        assert hasattr(lazy_engine, "join_logic")

        # Test that lazy engine overrides specific methods
        assert lazy_engine.get_column_names != PolarsMergeEngine.get_column_names  # type: ignore
        assert lazy_engine.is_empty_data != PolarsMergeEngine.is_empty_data  # type: ignore
