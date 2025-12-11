from typing import Any, Optional, Type
import pytest

from mloda_core.abstract_plugins.components.link import JoinType
from mloda_core.abstract_plugins.components.index.index import Index
from mloda_core.abstract_plugins.components.merge.base_merge_engine import BaseMergeEngine
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_merge_engine import (
    PythonDictMergeEngine,
)
from tests.test_plugins.compute_framework.test_tooling.multi_index.multi_index_test_base import (
    MultiIndexMergeEngineTestBase,
)


class TestPythonDictMergeEngine:
    """Unit tests for the PythonDictMergeEngine class."""

    @pytest.fixture
    def left_data(self) -> Any:
        """Create sample left dataset."""
        return [{"idx": 1, "col1": "a"}, {"idx": 3, "col1": "b"}]

    @pytest.fixture
    def right_data(self) -> Any:
        """Create sample right dataset."""
        return [{"idx": 1, "col2": "x"}, {"idx": 2, "col2": "z"}]

    def test_check_import(self) -> None:
        """Test that check_import passes without errors."""
        engine = PythonDictMergeEngine()
        engine.check_import()  # Should not raise any exception

    def test_merge_inner(self, left_data: Any, right_data: Any, index_obj: Any) -> None:
        """Test inner join."""
        engine = PythonDictMergeEngine()
        result = engine.merge_inner(left_data, right_data, index_obj, index_obj)

        expected = [{"idx": 1, "col1": "a", "col2": "x"}]
        assert result == expected

    def test_merge_left(self, left_data: Any, right_data: Any, index_obj: Any) -> None:
        """Test left join."""
        engine = PythonDictMergeEngine()
        result = engine.merge_left(left_data, right_data, index_obj, index_obj)

        expected = [{"idx": 1, "col1": "a", "col2": "x"}, {"idx": 3, "col1": "b", "col2": None}]
        assert result == expected

    def test_merge_right(self, left_data: Any, right_data: Any, index_obj: Any) -> None:
        """Test right join."""
        engine = PythonDictMergeEngine()
        result = engine.merge_right(left_data, right_data, index_obj, index_obj)

        expected = [{"idx": 1, "col1": "a", "col2": "x"}, {"idx": 2, "col1": None, "col2": "z"}]
        assert result == expected

    def test_merge_full_outer(self, left_data: Any, right_data: Any, index_obj: Any) -> None:
        """Test outer join."""
        engine = PythonDictMergeEngine()
        result = engine.merge_full_outer(left_data, right_data, index_obj, index_obj)

        # Sort by idx for consistent comparison
        result_sorted = sorted(result, key=lambda x: x["idx"])
        expected = [
            {"idx": 1, "col1": "a", "col2": "x"},
            {"idx": 2, "col1": None, "col2": "z"},
            {"idx": 3, "col1": "b", "col2": None},
        ]
        assert result_sorted == expected

    def test_merge_append(self, left_data: Any, right_data: Any, index_obj: Any) -> None:
        """Test append operation."""
        engine = PythonDictMergeEngine()
        result = engine.merge_append(left_data, right_data, index_obj, index_obj)

        expected = left_data + right_data
        assert result == expected

    def test_merge_union(self, left_data: Any, right_data: Any, index_obj: Any) -> None:
        """Test union operation (removes duplicates)."""
        engine = PythonDictMergeEngine()
        result = engine.merge_union(left_data, right_data, index_obj, index_obj)

        expected = [{"idx": 1, "col1": "a"}, {"idx": 3, "col1": "b"}, {"idx": 2, "col2": "z"}]
        assert result == expected

    def test_merge_with_different_join_columns(self) -> None:
        """Test merge with different column names for join."""
        left_data = [{"left_id": 1, "col1": "a"}, {"left_id": 2, "col1": "b"}]
        right_data = [{"right_id": 1, "col2": "x"}, {"right_id": 3, "col2": "z"}]

        left_index = Index(("left_id",))
        right_index = Index(("right_id",))

        engine = PythonDictMergeEngine()
        result = engine.merge_inner(left_data, right_data, left_index, right_index)

        expected = [{"left_id": 1, "col1": "a", "right_id": 1, "col2": "x"}]
        assert result == expected

    def test_merge_with_empty_datasets(self, index_obj: Any) -> None:
        """Test merge operations with empty datasets."""
        engine = PythonDictMergeEngine()

        # Empty left
        result = engine.merge_inner([], [{"idx": 1, "col": "a"}], index_obj, index_obj)
        assert result == []

        # Empty right
        result = engine.merge_inner([{"idx": 1, "col": "a"}], [], index_obj, index_obj)
        assert result == []

        # Both empty
        result = engine.merge_inner([], [], index_obj, index_obj)
        assert result == []

    def test_merge_with_complex_data(self) -> None:
        """Test merge with more complex data structures."""
        left_data = [
            {"id": 1, "name": "Alice", "age": 25},
            {"id": 2, "name": "Bob", "age": 30},
            {"id": 3, "name": "Charlie", "age": 35},
        ]

        right_data = [
            {"id": 1, "city": "New York", "country": "USA"},
            {"id": 2, "city": "London", "country": "UK"},
            {"id": 4, "city": "Tokyo", "country": "Japan"},
        ]

        index_obj = Index(("id",))
        engine = PythonDictMergeEngine()

        # Test inner join
        result = engine.merge_inner(left_data, right_data, index_obj, index_obj)
        expected = [
            {"id": 1, "name": "Alice", "age": 25, "city": "New York", "country": "USA"},
            {"id": 2, "name": "Bob", "age": 30, "city": "London", "country": "UK"},
        ]
        assert result == expected

    def test_merge_with_none_values(self) -> None:
        """Test merge operations with None values in join columns."""
        left_data = [{"id": 1, "col1": "a"}, {"id": None, "col1": "b"}]
        right_data = [{"id": 1, "col2": "x"}, {"id": None, "col2": "y"}]

        index_obj = Index(("id",))
        engine = PythonDictMergeEngine()

        result = engine.merge_inner(left_data, right_data, index_obj, index_obj)
        expected = [{"id": 1, "col1": "a", "col2": "x"}, {"id": None, "col1": "b", "col2": "y"}]
        assert result == expected

    def test_join_logic_integration(self, left_data: Any, right_data: Any, index_obj: Any) -> None:
        """Test that join_logic method works correctly."""
        engine = PythonDictMergeEngine()

        # Test inner join through join_logic
        result = engine.join_logic("inner", left_data, right_data, index_obj, index_obj, JoinType.INNER)
        expected = [{"idx": 1, "col1": "a", "col2": "x"}]
        assert result == expected

        # Test unsupported join type
        with pytest.raises(ValueError, match="Join type unsupported is not supported"):
            engine.join_logic("unsupported", left_data, right_data, index_obj, index_obj, JoinType.INNER)

    def test_merge_method_integration(self, left_data: Any, right_data: Any, index_obj: Any) -> None:
        """Test the main merge method that dispatches to specific join types."""
        engine = PythonDictMergeEngine()

        # Test all join types through the main merge method
        result = engine.merge(left_data, right_data, JoinType.INNER, index_obj, index_obj)
        expected = [{"idx": 1, "col1": "a", "col2": "x"}]
        assert result == expected

        result = engine.merge(left_data, right_data, JoinType.LEFT, index_obj, index_obj)
        expected = [{"idx": 1, "col1": "a", "col2": "x"}, {"idx": 3, "col1": "b", "col2": None}]
        assert result == expected


class TestPythonDictMergeEngineMultiIndex(MultiIndexMergeEngineTestBase):
    """Test PythonDictMergeEngine multi-index support using shared test scenarios."""

    @classmethod
    def merge_engine_class(cls) -> Type[BaseMergeEngine]:
        """Return the PythonDictMergeEngine class."""
        return PythonDictMergeEngine

    @classmethod
    def framework_type(cls) -> Type[Any]:
        """Return list type (PythonDict uses List[Dict])."""
        return list

    def get_connection(self) -> Optional[Any]:
        """PythonDict does not require a connection object."""
        return None
