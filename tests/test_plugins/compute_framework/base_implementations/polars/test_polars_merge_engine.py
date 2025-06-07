from typing import Any
import pytest

from mloda_core.abstract_plugins.components.link import JoinType
from mloda_core.abstract_plugins.components.index.index import Index
from mloda_plugins.compute_framework.base_implementations.polars.polars_merge_engine import PolarsMergeEngine

import logging

logger = logging.getLogger(__name__)

try:
    import polars as pl
except ImportError:
    logger.warning("Polars is not installed. Some tests will be skipped.")
    pl = None  # type: ignore


@pytest.mark.skipif(pl is None, reason="Polars is not installed. Skipping this test.")
class TestPolarsMergeEngine:
    """Unit tests for the PolarsMergeEngine class."""

    @pytest.fixture
    def left_data(self) -> Any:
        """Create sample left dataset."""
        return pl.DataFrame({"idx": [1, 3], "col1": ["a", "b"]})

    @pytest.fixture
    def right_data(self) -> Any:
        """Create sample right dataset."""
        return pl.DataFrame({"idx": [1, 2], "col2": ["x", "z"]})

    @pytest.fixture
    def index_obj(self) -> Any:
        """Create index object for joins."""
        return Index(("idx",))

    def test_check_import(self) -> None:
        """Test that check_import passes without errors."""
        engine = PolarsMergeEngine()
        engine.check_import()  # Should not raise any exception

    def test_merge_inner(self, left_data: Any, right_data: Any, index_obj: Any) -> None:
        """Test inner join."""
        engine = PolarsMergeEngine()
        result = engine.merge_inner(left_data, right_data, index_obj, index_obj)

        expected = pl.DataFrame({"idx": [1], "col1": ["a"], "col2": ["x"]})
        assert result.equals(expected)

    def test_merge_left(self, left_data: Any, right_data: Any, index_obj: Any) -> None:
        """Test left join."""
        engine = PolarsMergeEngine()
        result = engine.merge_left(left_data, right_data, index_obj, index_obj)

        expected = pl.DataFrame({"idx": [1, 3], "col1": ["a", "b"], "col2": ["x", None]})
        assert result.equals(expected)

    def test_merge_right(self, left_data: Any, right_data: Any, index_obj: Any) -> None:
        """Test right join."""
        engine = PolarsMergeEngine()
        result = engine.merge_right(left_data, right_data, index_obj, index_obj)

        expected = pl.DataFrame({"idx": [1, 2], "col1": ["a", None], "col2": ["x", "z"]})
        assert result.equals(expected)

    def test_merge_full_outer(self, left_data: Any, right_data: Any, index_obj: Any) -> None:
        """Test outer join."""
        engine = PolarsMergeEngine()
        result = engine.merge_full_outer(left_data, right_data, index_obj, index_obj)

        # Sort by idx for consistent comparison
        result_sorted = result.sort("idx")
        expected = pl.DataFrame({"idx": [1, 2, 3], "col1": ["a", None, "b"], "col2": ["x", "z", None]})
        assert result_sorted.equals(expected)

    def test_merge_append(self, left_data: Any, right_data: Any, index_obj: Any) -> None:
        """Test append operation."""
        engine = PolarsMergeEngine()
        result = engine.merge_append(left_data, right_data, index_obj, index_obj)

        expected = pl.concat([left_data, right_data], how="diagonal")
        assert result.equals(expected)

    def test_merge_union(self, left_data: Any, right_data: Any, index_obj: Any) -> None:
        """Test union operation (removes duplicates)."""
        engine = PolarsMergeEngine()
        result = engine.merge_union(left_data, right_data, index_obj, index_obj)

        expected = pl.concat([left_data, right_data], how="diagonal").unique()
        # Sort both for consistent comparison
        result_sorted = result.sort(["idx", "col1", "col2"])
        expected_sorted = expected.sort(["idx", "col1", "col2"])
        assert result_sorted.equals(expected_sorted)

    def test_merge_with_different_join_columns(self) -> None:
        """Test merge with different column names for join."""
        left_data = pl.DataFrame({"left_id": [1, 2], "col1": ["a", "b"]})
        right_data = pl.DataFrame({"right_id": [1, 3], "col2": ["x", "z"]})

        left_index = Index(("left_id",))
        right_index = Index(("right_id",))

        engine = PolarsMergeEngine()
        result = engine.merge_inner(left_data, right_data, left_index, right_index)

        expected = pl.DataFrame({"left_id": [1], "col1": ["a"], "right_id": [1], "col2": ["x"]})
        assert result.equals(expected)

    def test_merge_with_empty_datasets(self, index_obj: Any) -> None:
        """Test merge operations with empty datasets."""
        engine = PolarsMergeEngine()
        empty_df = pl.DataFrame({"idx": [], "col": []})
        non_empty_df = pl.DataFrame({"idx": [1], "col": ["a"]})

        # Empty left
        result = engine.merge_inner(empty_df, non_empty_df, index_obj, index_obj)
        assert len(result) == 0

        # Empty right
        result = engine.merge_inner(non_empty_df, empty_df, index_obj, index_obj)
        assert len(result) == 0

        # Both empty
        result = engine.merge_inner(empty_df, empty_df, index_obj, index_obj)
        assert len(result) == 0

    def test_merge_with_multi_index_error(self, left_data: Any, right_data: Any) -> None:
        """Test that multi-index raises an error."""
        multi_index = Index(("col1", "col2"))
        engine = PolarsMergeEngine()

        with pytest.raises(ValueError, match="MultiIndex is not yet implemented"):
            engine.merge_inner(left_data, right_data, multi_index, multi_index)

    def test_merge_with_complex_data(self) -> None:
        """Test merge with more complex data structures."""
        left_data = pl.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]})

        right_data = pl.DataFrame(
            {"id": [1, 2, 4], "city": ["New York", "London", "Tokyo"], "country": ["USA", "UK", "Japan"]}
        )

        index_obj = Index(("id",))
        engine = PolarsMergeEngine()

        # Test inner join
        result = engine.merge_inner(left_data, right_data, index_obj, index_obj)
        expected = pl.DataFrame(
            {
                "id": [1, 2],
                "name": ["Alice", "Bob"],
                "age": [25, 30],
                "city": ["New York", "London"],
                "country": ["USA", "UK"],
            }
        )
        assert result.equals(expected)

    def test_merge_with_null_values(self) -> None:
        """Test merge operations with null values in join columns."""
        left_data = pl.DataFrame({"id": [1, None], "col1": ["a", "b"]})
        right_data = pl.DataFrame({"id": [1, None], "col2": ["x", "y"]})

        index_obj = Index(("id",))
        engine = PolarsMergeEngine()

        result = engine.merge_inner(left_data, right_data, index_obj, index_obj)
        expected = pl.DataFrame({"id": [1, None], "col1": ["a", "b"], "col2": ["x", "y"]})
        assert result.equals(expected)

    def test_join_logic_integration(self, left_data: Any, right_data: Any, index_obj: Any) -> None:
        """Test that join_logic method works correctly."""
        engine = PolarsMergeEngine()

        # Test inner join through join_logic
        result = engine.join_logic("inner", left_data, right_data, index_obj, index_obj, JoinType.INNER)
        expected = pl.DataFrame({"idx": [1], "col1": ["a"], "col2": ["x"]})
        assert result.equals(expected)

    def test_merge_method_integration(self, left_data: Any, right_data: Any, index_obj: Any) -> None:
        """Test the main merge method that dispatches to specific join types."""
        engine = PolarsMergeEngine()

        # Test all join types through the main merge method
        result = engine.merge(left_data, right_data, JoinType.INNER, index_obj, index_obj)
        expected = pl.DataFrame({"idx": [1], "col1": ["a"], "col2": ["x"]})
        assert result.equals(expected)

        result = engine.merge(left_data, right_data, JoinType.LEFT, index_obj, index_obj)
        expected = pl.DataFrame({"idx": [1, 3], "col1": ["a", "b"], "col2": ["x", None]})
        assert result.equals(expected)

    def test_pl_concat_static_method(self) -> None:
        """Test the static pl_concat method."""
        concat_func = PolarsMergeEngine.pl_concat()

        df1 = pl.DataFrame({"a": [1, 2]})
        df2 = pl.DataFrame({"a": [3, 4]})

        result = concat_func([df1, df2])
        expected = pl.DataFrame({"a": [1, 2, 3, 4]})
        assert result.equals(expected)
