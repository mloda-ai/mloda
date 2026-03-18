"""
Shared test mixin for all Relation implementations.

This mixin provides common test methods that verify the relation contract.
Each framework-specific test class should inherit from this mixin and provide:
- sample_relation fixture: Returns a relation with standard test data (id, age, name, category)
- relation_class fixture: Returns the relation class (for factory tests)
- get_column_values method: Extracts column values as a list from results

The ``connection`` fixture is expected to come from conftest.py in the backend directory.
"""

from abc import abstractmethod
from typing import Any, List

import pyarrow as pa
import pytest


class RelationTestMixin:
    """Shared tests for all Relation implementations."""

    @pytest.fixture
    @abstractmethod
    def sample_relation(self, connection: Any) -> Any:
        """Return a relation with standard test data.

        Override in framework-specific test class.
        Data should contain columns: id, age, name, category
        with values:
            id: [1, 2, 3, 4, 5]
            age: [25, 30, 35, 40, 45]
            name: ["Alice", "Bob", "Charlie", "David", "Eve"]
            category: ["A", "B", "A", "C", "B"]
        """
        raise NotImplementedError

    @pytest.fixture
    @abstractmethod
    def relation_class(self) -> Any:
        """Return the relation class to test (for from_arrow/from_dict factory tests).

        Override in framework-specific test class.
        """
        raise NotImplementedError

    @abstractmethod
    def get_column_values(self, result: Any, column: str) -> List[Any]:
        """Extract column values as a list from the result.

        Override in framework-specific test class.
        """
        raise NotImplementedError

    # --- Basics ---

    def test_columns(self, sample_relation: Any) -> None:
        assert sample_relation.columns == ["id", "age", "name", "category"]

    def test_len(self, sample_relation: Any) -> None:
        assert len(sample_relation) == 5

    def test_to_arrow_table(self, sample_relation: Any) -> None:
        result = sample_relation.to_arrow_table()
        assert isinstance(result, pa.Table)
        assert result.num_rows == 5
        assert set(result.column_names) == {"id", "age", "name", "category"}

    def test_df(self, sample_relation: Any) -> None:
        df = sample_relation.df()
        assert len(df) == 5
        assert set(df.columns) == {"id", "age", "name", "category"}

    # --- Filter ---

    def test_filter_returns_filtered_rows(self, sample_relation: Any) -> None:
        filtered = sample_relation.filter("age >= ?", (35,))
        assert len(filtered) == 3

    def test_filter_does_not_mutate_original(self, sample_relation: Any) -> None:
        sample_relation.filter("age >= ?", (35,))
        assert len(sample_relation) == 5

    def test_filter_with_params(self, sample_relation: Any) -> None:
        filtered = sample_relation.filter("age >= ?", (30,))
        assert len(filtered) == 4

    def test_filter_sql_injection(self, sample_relation: Any) -> None:
        """A crafted value passed as a native param must not be interpreted as SQL."""
        crafted = "'; DROP TABLE users; --"
        filtered = sample_relation.filter("name = ?", (crafted,))
        assert len(filtered) == 0
        assert len(sample_relation) == 5

    # --- Select ---

    def test_select_columns(self, sample_relation: Any) -> None:
        selected = sample_relation.select("id", "name")
        assert selected.columns == ["id", "name"]
        assert len(selected) == 5

    def test_select_raw_sql_expression(self, sample_relation: Any) -> None:
        selected = sample_relation.select(_raw_sql="*, age * 2 AS doubled_age")
        assert "doubled_age" in selected.columns
        assert len(selected) == 5

    def test_select_raw_sql_window_function(self, sample_relation: Any) -> None:
        selected = sample_relation.select(_raw_sql="*, AVG(age) OVER () AS avg_age")
        assert "avg_age" in selected.columns
        assert len(selected) == 5

    # --- Alias ---

    def test_set_alias(self, sample_relation: Any) -> None:
        aliased = sample_relation.set_alias("my_alias")
        assert aliased.get_alias() == "my_alias"

    def test_alias_does_not_change_original(self, sample_relation: Any) -> None:
        sample_relation.set_alias("my_alias")
        assert sample_relation.get_alias() is None

    def test_aliased_data_accessible(self, sample_relation: Any) -> None:
        aliased = sample_relation.set_alias("my_alias")
        assert len(aliased) == 5

    # --- Limit ---

    def test_limit_restricts_rows(self, sample_relation: Any) -> None:
        limited = sample_relation.limit(2)
        assert len(limited) == 2

    def test_limit_returns_new_relation(self, sample_relation: Any) -> None:
        limited = sample_relation.limit(3)
        assert len(sample_relation) == 5
        assert len(limited) == 3

    # --- Join ---

    def test_inner_join(self, connection: Any, relation_class: Any) -> None:
        left = relation_class.from_dict(connection, {"idx": [1, 2, 3], "val": ["a", "b", "c"]})
        right = relation_class.from_dict(connection, {"idx": [2, 3, 4], "score": [10, 20, 30]})

        left_aliased = left.set_alias("left_rel")
        right_aliased = right.set_alias("right_rel")

        result = left_aliased.join(right_aliased, "left_rel.idx = right_rel.idx", how="inner")
        assert len(result) == 2

    def test_left_join(self, connection: Any, relation_class: Any) -> None:
        left = relation_class.from_dict(connection, {"idx": [1, 2, 3], "val": ["a", "b", "c"]})
        right = relation_class.from_dict(connection, {"idx": [2, 3, 4], "score": [10, 20, 30]})

        left_aliased = left.set_alias("left_rel")
        right_aliased = right.set_alias("right_rel")

        result = left_aliased.join(right_aliased, "left_rel.idx = right_rel.idx", how="left")
        assert len(result) == 3

    def test_unsupported_join_type_raises(self, connection: Any, relation_class: Any) -> None:
        left = relation_class.from_dict(connection, {"idx": [1]})
        right = relation_class.from_dict(connection, {"idx": [1]})

        with pytest.raises(ValueError, match="Unsupported join type"):
            left.join(right, "1=1", how="cross")

    # --- Factory: from_arrow ---

    def test_from_arrow_roundtrip(self, connection: Any, relation_class: Any) -> None:
        original = pa.Table.from_pydict({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        rel = relation_class.from_arrow(connection, original)
        result = rel.to_arrow_table()
        assert result.num_rows == 3
        assert set(result.column_names) == {"a", "b"}

    def test_from_arrow_empty_table(self, connection: Any, relation_class: Any) -> None:
        arrow = pa.Table.from_pydict({"id": pa.array([], type=pa.int64()), "name": pa.array([], type=pa.string())})
        rel = relation_class.from_arrow(connection, arrow)
        assert len(rel) == 0
        assert set(rel.columns) == {"id", "name"}

    # --- Factory: from_dict ---

    def test_from_dict_basic(self, connection: Any, relation_class: Any) -> None:
        data: dict[str, list[Any]] = {"x": [10, 20], "y": ["a", "b"]}
        rel = relation_class.from_dict(connection, data)
        assert len(rel) == 2
        assert rel.columns == ["x", "y"]

    def test_from_dict_empty_raises(self, connection: Any, relation_class: Any) -> None:
        empty: dict[str, list[Any]] = {}
        with pytest.raises(ValueError, match="Cannot create relation from empty dictionary"):
            relation_class.from_dict(connection, empty)

    # --- Join: reserved-word column name ---

    def test_join_bare_column_reserved_word(self, connection: Any, relation_class: Any) -> None:
        """Joining on a bare column name that is a SQL reserved word must work."""
        left = relation_class.from_dict(connection, {"order": [1, 2, 3], "val": ["a", "b", "c"]})
        right = relation_class.from_dict(connection, {"order": [2, 3, 4], "score": [10, 20, 30]})
        left_aliased = left.set_alias("left_rel")
        right_aliased = right.set_alias("right_rel")
        result = left_aliased.join(right_aliased, "order", how="inner")
        assert len(result) == 2
