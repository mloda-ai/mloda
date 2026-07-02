from datetime import datetime, timezone
from typing import Any, Optional
import pytest

from mloda.user import JoinType
from mloda.user import Index
from mloda.provider import BaseMergeEngine
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_merge_engine import (
    PythonDictMergeEngine,
)
from tests.test_plugins.compute_framework.test_tooling.merge_link import make_merge_link
from tests.test_plugins.compute_framework.test_tooling.multi_index.multi_index_test_base import (
    MultiIndexMergeEngineTestBase,
)


class TestPythonDictMergeEngine:
    """Unit tests for the PythonDictMergeEngine class."""

    @pytest.fixture
    def left_data(self) -> Any:
        """Create sample columnar left dataset."""
        return {"idx": [1, 3], "col1": ["a", "b"]}

    @pytest.fixture
    def right_data(self) -> Any:
        """Create sample columnar right dataset."""
        return {"idx": [1, 2], "col2": ["x", "z"]}

    def test_check_import(self) -> None:
        """Test that check_import passes without errors."""
        engine = PythonDictMergeEngine()
        engine.check_import()  # Should not raise any exception

    def test_merge_inner(self, left_data: Any, right_data: Any, index_obj: Any) -> None:
        """Test inner join."""
        engine = PythonDictMergeEngine()
        result = engine.merge_inner(left_data, right_data, index_obj, index_obj)

        assert result == {"idx": [1], "col1": ["a"], "col2": ["x"]}

    def test_merge_left(self, left_data: Any, right_data: Any, index_obj: Any) -> None:
        """Test left join."""
        engine = PythonDictMergeEngine()
        result = engine.merge_left(left_data, right_data, index_obj, index_obj)

        assert result == {"idx": [1, 3], "col1": ["a", "b"], "col2": ["x", None]}

    def test_merge_right(self, left_data: Any, right_data: Any, index_obj: Any) -> None:
        """Test right join."""
        engine = PythonDictMergeEngine()
        result = engine.merge_right(left_data, right_data, index_obj, index_obj)

        assert result == {"idx": [1, 2], "col2": ["x", "z"], "col1": ["a", None]}

    def test_merge_full_outer(self, left_data: Any, right_data: Any, index_obj: Any) -> None:
        """Test outer join."""
        engine = PythonDictMergeEngine()
        result = engine.merge_full_outer(left_data, right_data, index_obj, index_obj)

        assert result == {"idx": [1, 3, 2], "col1": ["a", "b", None], "col2": ["x", None, "z"]}

    def test_merge_append(self, left_data: Any, right_data: Any, index_obj: Any) -> None:
        """Test append operation (column-wise concatenation with the union of columns)."""
        engine = PythonDictMergeEngine()
        result = engine.merge_append(left_data, right_data, index_obj, index_obj)

        assert result == {
            "idx": [1, 3, 1, 2],
            "col1": ["a", "b", None, None],
            "col2": [None, None, "x", "z"],
        }

    def test_merge_union(self, left_data: Any, right_data: Any, index_obj: Any) -> None:
        """Test union operation (removes duplicates)."""
        engine = PythonDictMergeEngine()
        result = engine.merge_union(left_data, right_data, index_obj, index_obj)

        assert result == {"idx": [1, 3, 2], "col1": ["a", "b", None], "col2": [None, None, "z"]}

    def test_merge_with_different_join_columns(self) -> None:
        """Test merge with different column names for join."""
        left_data = {"left_id": [1, 2], "col1": ["a", "b"]}
        right_data = {"right_id": [1, 3], "col2": ["x", "z"]}

        left_index = Index(("left_id",))
        right_index = Index(("right_id",))

        engine = PythonDictMergeEngine()
        result = engine.merge_inner(left_data, right_data, left_index, right_index)

        assert result == {"left_id": [1], "col1": ["a"], "right_id": [1], "col2": ["x"]}

    def test_merge_with_empty_datasets(self, index_obj: Any) -> None:
        """Test merge operations with empty datasets (schema-bearing zero-row frames)."""
        engine = PythonDictMergeEngine()

        # Empty left carries its schema; the right side contributes its columns.
        result = engine.merge_inner({"idx": [], "col1": []}, {"idx": [1], "col": ["a"]}, index_obj, index_obj)
        assert result == {"idx": [], "col1": [], "col": []}

        # Empty right.
        result = engine.merge_inner({"idx": [1], "col": ["a"]}, {"idx": [], "col2": []}, index_obj, index_obj)
        assert result == {"idx": [], "col": [], "col2": []}

        # Both empty.
        result = engine.merge_inner({"idx": []}, {"idx": []}, index_obj, index_obj)
        assert result == {"idx": []}

    def test_merge_with_complex_data(self) -> None:
        """Test merge with more complex data structures."""
        left_data = {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
        }

        right_data = {
            "id": [1, 2, 4],
            "city": ["New York", "London", "Tokyo"],
            "country": ["USA", "UK", "Japan"],
        }

        index_obj = Index(("id",))
        engine = PythonDictMergeEngine()

        result = engine.merge_inner(left_data, right_data, index_obj, index_obj)
        assert result == {
            "id": [1, 2],
            "name": ["Alice", "Bob"],
            "age": [25, 30],
            "city": ["New York", "London"],
            "country": ["USA", "UK"],
        }

    def test_merge_with_none_values(self) -> None:
        """Test merge operations with None values in join columns."""
        left_data = {"id": [1, None], "col1": ["a", "b"]}
        right_data = {"id": [1, None], "col2": ["x", "y"]}

        index_obj = Index(("id",))
        engine = PythonDictMergeEngine()

        result = engine.merge_inner(left_data, right_data, index_obj, index_obj)
        assert result == {"id": [1, None], "col1": ["a", "b"], "col2": ["x", "y"]}

    def test_join_logic_integration(self, left_data: Any, right_data: Any, index_obj: Any) -> None:
        """Test that join_logic method works correctly."""
        engine = PythonDictMergeEngine()

        result = engine.join_logic("inner", left_data, right_data, index_obj, index_obj, JoinType.INNER)
        assert result == {"idx": [1], "col1": ["a"], "col2": ["x"]}

        with pytest.raises(ValueError, match="Join type unsupported is not supported"):
            engine.join_logic("unsupported", left_data, right_data, index_obj, index_obj, JoinType.INNER)

    def test_merge_method_integration(self, left_data: Any, right_data: Any, index_obj: Any) -> None:
        """Test the main merge method that dispatches to specific join types."""
        engine = PythonDictMergeEngine()

        result = engine.merge(left_data, right_data, make_merge_link(JoinType.INNER, index_obj, index_obj))
        assert result == {"idx": [1], "col1": ["a"], "col2": ["x"]}

        result = engine.merge(left_data, right_data, make_merge_link(JoinType.LEFT, index_obj, index_obj))
        assert result == {"idx": [1, 3], "col1": ["a", "b"], "col2": ["x", None]}


class TestPythonDictEquiJoinTimezoneGuard:
    """Phase 3b: extend the temporal cross-side timezone guard to python_dict equi-joins.

    python_dict's Phase-1 column_semantics reads temporal/tz from the first non-null value's
    tzinfo, so once Green wires PythonDictMergeEngine._column_semantics to it, an INNER equi-join
    whose key is tz-aware on one side and tz-naive on the other must raise a ValueError naming the
    timezone mismatch. Today no override is wired, so the guard never fires.
    """

    def test_inner_equi_join_tz_aware_vs_naive_raises(self) -> None:
        """NEGATIVE: tz-aware datetime key on the left, tz-naive on the right must be rejected."""
        engine = PythonDictMergeEngine()
        idx = Index(("t",))
        left_data = [{"t": datetime(2021, 1, 1, 12, 0, tzinfo=timezone.utc), "lv": "a"}]
        right_data = [{"t": datetime(2021, 1, 1, 12, 0), "rv": "x"}]
        link = make_merge_link(JoinType.INNER, idx, idx)

        with pytest.raises(ValueError, match=r"(?i)time[ -]?zone"):
            engine.merge(left_data, right_data, link)

    def test_inner_equi_join_both_naive_succeeds(self) -> None:
        """POSITIVE: both-naive datetime keys are legal and join normally."""
        engine = PythonDictMergeEngine()
        idx = Index(("t",))
        left_data = [{"t": datetime(2021, 1, 1, 12, 0), "lv": "a"}]
        right_data = [{"t": datetime(2021, 1, 1, 12, 0), "rv": "x"}]
        link = make_merge_link(JoinType.INNER, idx, idx)

        result = engine.merge(left_data, right_data, link)
        assert result == [{"t": datetime(2021, 1, 1, 12, 0), "lv": "a", "rv": "x"}]

    def test_inner_equi_join_string_key_is_legal(self) -> None:
        """POSITIVE: a non-temporal (string) equi-join key stays unaffected by the guard."""
        engine = PythonDictMergeEngine()
        idx = Index(("k",))
        left_data = [{"k": "a", "lv": 1}]
        right_data = [{"k": "a", "rv": 2}]
        link = make_merge_link(JoinType.INNER, idx, idx)

        result = engine.merge(left_data, right_data, link)
        assert result == [{"k": "a", "lv": 1, "rv": 2}]


class TestPythonDictMergeEngineMultiIndex(MultiIndexMergeEngineTestBase):
    """Test PythonDictMergeEngine multi-index support using shared test scenarios."""

    @classmethod
    def merge_engine_class(cls) -> type[BaseMergeEngine]:
        """Return the PythonDictMergeEngine class."""
        return PythonDictMergeEngine

    @classmethod
    def framework_type(cls) -> type[Any]:
        """Return dict type (PythonDict uses a columnar dict)."""
        return dict

    def get_connection(self) -> Optional[Any]:
        """PythonDict does not require a connection object."""
        return None
