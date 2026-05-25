"""
Shared test mixin for SQL Relation implementations exercising window helpers.

Backends inheriting this mixin must also inherit RelationTestMixin: the mixin relies
entirely on the existing ``sample_relation`` / ``relation_class`` fixtures and the
``get_column_values`` helper declared by ``RelationTestMixin``.
"""

from abc import abstractmethod
from typing import Any

import pytest

from mloda_plugins.compute_framework.base_implementations.sql.sql_window import (
    CurrentRow,
    Following,
    OrderBy,
    Preceding,
    Unbounded,
    WindowFrame,
)


class SqlRelationWindowTestMixin:
    """Shared window/with_row_number tests for SQL-backed Relation implementations."""

    @pytest.fixture
    @abstractmethod
    def sample_relation(self, connection: Any) -> Any:
        """Provided by the backend test class (also satisfied by RelationTestMixin)."""
        raise NotImplementedError

    @pytest.fixture
    @abstractmethod
    def relation_class(self) -> Any:
        """Provided by the backend test class (also satisfied by RelationTestMixin)."""
        raise NotImplementedError

    @abstractmethod
    def get_column_values(self, result: Any, column: str) -> list[Any]:
        """Provided by the backend test class (also satisfied by RelationTestMixin)."""
        raise NotImplementedError

    # --- with_row_number ---

    def test_with_row_number_bare_over(self, sample_relation: Any) -> None:
        """Empty partition_by and order_by produce a bare ROW_NUMBER() OVER () clause."""
        result = sample_relation.with_row_number("rn")
        assert "rn" in result.columns
        assert len(result) == 5
        rns = self.get_column_values(result, "rn")
        # DuckDB does not guarantee assignment order with bare OVER (), so test set equality
        assert sorted(rns) == [1, 2, 3, 4, 5]

    def test_with_row_number_partition_by_only(self, sample_relation: Any) -> None:
        """partition_by without order_by: each row's rn is within partition size; sum is deterministic."""
        result = sample_relation.with_row_number("rn", partition_by=("category",))
        assert "rn" in result.columns
        assert len(result) == 5
        rns = self.get_column_values(result, "rn")
        # Categories: A,A,B,B,C -> partition sizes 2,2,1 -> rns are permutations of [1,2],[1,2],[1]
        # Sum is 1+2+1+2+1 = 7
        assert sum(rns) == 7
        # Each rn must be between 1 and its partition size (max partition size is 2)
        for rn in rns:
            assert 1 <= rn <= 2

    def test_with_row_number_order_by_only(self, sample_relation: Any) -> None:
        """order_by without partition_by: rows ordered by age get rn 1..5."""
        result = sample_relation.with_row_number("rn", order_by=("age",))
        assert "rn" in result.columns
        # Order the result by age and verify rn == [1,2,3,4,5]
        ordered = result.order("age")
        rns = self.get_column_values(ordered, "rn")
        assert rns == [1, 2, 3, 4, 5]

    def test_with_row_number_partition_and_order(self, sample_relation: Any) -> None:
        """partition_by=('category',), order_by=('age',). Verify expected rn per row by id."""
        result = sample_relation.with_row_number("rn", partition_by=("category",), order_by=("age",))
        ordered = result.order("id")
        ids = self.get_column_values(ordered, "id")
        rns = self.get_column_values(ordered, "rn")
        assert ids == [1, 2, 3, 4, 5]
        # id=1 (A,25)->1, id=2 (B,30)->1, id=3 (A,35)->2, id=4 (C,40)->1, id=5 (B,45)->2
        assert rns == [1, 1, 2, 1, 2]

    def test_with_row_number_returns_new_relation(self, sample_relation: Any, relation_class: Any) -> None:
        """Original relation is not mutated; returned object is a relation_class instance."""
        original_columns = list(sample_relation.columns)
        result = sample_relation.with_row_number("rn", order_by=("age",))
        assert isinstance(result, relation_class)
        assert sample_relation.columns == original_columns
        assert "rn" not in sample_relation.columns

    def test_with_row_number_alias_with_special_chars(self, sample_relation: Any) -> None:
        """The method must quote the alias; an alias containing a double quote must round-trip verbatim."""
        weird = 'weird"name'
        result = sample_relation.with_row_number(weird, order_by=("age",))
        assert weird in result.columns
        assert len(result) == 5

    def test_with_row_number_partition_column_with_special_chars(self, connection: Any, relation_class: Any) -> None:
        """The method must quote partition_by columns; a column literally named 'odd col' must work."""
        rel = relation_class.from_dict(
            connection,
            {
                "odd col": ["x", "x", "y"],
                "v": [1, 2, 3],
            },
        )
        result = rel.with_row_number("rn", partition_by=("odd col",))
        assert "rn" in result.columns
        assert len(result) == 3

    def test_with_row_number_preserves_original_columns(self, sample_relation: Any) -> None:
        """Original columns appear first (in original order); the new column appears last."""
        original_columns = list(sample_relation.columns)
        result = sample_relation.with_row_number("rn", order_by=("age",))
        assert result.columns[: len(original_columns)] == original_columns
        assert result.columns[-1] == "rn"

    def test_with_row_number_partition_by_keyword_only(self, sample_relation: Any) -> None:
        """partition_by must be keyword-only to match the sibling window() API."""
        with pytest.raises(TypeError):
            sample_relation.with_row_number("rn", ("category",))  # type: ignore[misc]

    def test_with_row_number_order_by_keyword_only(self, sample_relation: Any) -> None:
        """order_by must be keyword-only to match the sibling window() API."""
        with pytest.raises(TypeError):
            sample_relation.with_row_number("rn", (), ("age",))  # type: ignore[misc]

    def test_with_row_number_raises_when_alias_already_exists(self, sample_relation: Any) -> None:
        """with_row_number must reject an alias colliding with an existing column instead of producing a duplicated name."""
        with pytest.raises(ValueError, match="category"):
            sample_relation.with_row_number("category")

    def test_with_row_number_raises_on_case_only_collision(self, connection: Any, relation_class: Any) -> None:
        """DuckDB identifiers are case-insensitive: alias 'category' must collide with existing 'Category'."""
        rel = relation_class.from_dict(
            connection,
            {
                "id": [1, 2, 3, 4, 5],
                "Category": ["A", "B", "A", "C", "B"],
            },
        )
        with pytest.raises(ValueError, match="category"):
            rel.with_row_number("category")

    def test_with_row_number_order_by_orderby_descending(self, sample_relation: Any) -> None:
        """with_row_number must honor OrderBy(descending=True) and produce the descending row numbers."""
        result = sample_relation.with_row_number("rn", order_by=(OrderBy("id", descending=True),))
        ordered = result.order("id")
        assert self.get_column_values(ordered, "rn") == [5, 4, 3, 2, 1]

    # --- window() ---

    def test_window_partition_only(self, sample_relation: Any) -> None:
        """SUM(age) partitioned by category, no order, no frame: each row gets its category total."""
        result = sample_relation.window("SUM(age)", "cat_sum", partition_by=("category",))
        assert "cat_sum" in result.columns
        ordered = result.order("id")
        sums = self.get_column_values(ordered, "cat_sum")
        # Categories per id (1..5): A,B,A,C,B; sums: A=25+35=60, B=30+45=75, C=40
        assert sums == [60, 75, 60, 40, 75]

    def test_window_partition_and_order_running_sum(self, sample_relation: Any) -> None:
        """SUM(age) partitioned by category, ordered by id, default RANGE UNBOUNDED PRECEDING..CURRENT ROW."""
        result = sample_relation.window("SUM(age)", "rs", partition_by=("category",), order_by=("id",))
        ordered = result.order("id")
        rs = self.get_column_values(ordered, "rs")
        assert rs == [25, 30, 60, 40, 75]

    def test_window_no_partition_no_order(self, sample_relation: Any) -> None:
        """COUNT(*) with no partition / order / frame returns total row count for every row."""
        result = sample_relation.window("COUNT(*)", "n")
        ns = self.get_column_values(result, "n")
        assert ns == [5, 5, 5, 5, 5]

    def test_window_returns_new_relation_and_preserves_columns(self, sample_relation: Any, relation_class: Any) -> None:
        """window() returns a new relation with original columns plus alias; original is untouched."""
        original_columns = list(sample_relation.columns)
        result = sample_relation.window("COUNT(*)", "n")
        assert isinstance(result, relation_class)
        for col in original_columns:
            assert col in result.columns
        assert "n" in result.columns
        # Original relation untouched
        assert sample_relation.columns == original_columns
        assert "n" not in sample_relation.columns

    def test_window_rows_frame_unbounded_to_current(self, sample_relation: Any) -> None:
        """ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW: cumulative sum ordered by id."""
        result = sample_relation.window(
            "SUM(age)",
            "cum",
            order_by=("id",),
            frame=WindowFrame(kind="rows", start=Unbounded(), end=CurrentRow()),
        )
        ordered = result.order("id")
        cum = self.get_column_values(ordered, "cum")
        assert cum == [25, 55, 90, 130, 175]

    def test_window_range_frame_unbounded_to_unbounded(self, sample_relation: Any) -> None:
        """RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING: grand total per row."""
        result = sample_relation.window(
            "SUM(age)",
            "total",
            order_by=("id",),
            frame=WindowFrame(kind="range", start=Unbounded(), end=Unbounded()),
        )
        ordered = result.order("id")
        totals = self.get_column_values(ordered, "total")
        assert totals == [175, 175, 175, 175, 175]

    def test_window_groups_frame_preceding_to_current(self, sample_relation: Any) -> None:
        """GROUPS BETWEEN 1 PRECEDING AND CURRENT ROW: just verify the frame variant compiles and produces a column."""
        result = sample_relation.window(
            "SUM(age)",
            "g",
            order_by=("category",),
            frame=WindowFrame(kind="groups", start=Preceding(1), end=CurrentRow()),
        )
        assert "g" in result.columns
        assert len(result) == 5

    def test_window_rows_frame_preceding_and_following(self, sample_relation: Any) -> None:
        """ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING ordered by id: sliding 3-row window sum."""
        result = sample_relation.window(
            "SUM(age)",
            "win",
            order_by=("id",),
            frame=WindowFrame(kind="rows", start=Preceding(1), end=Following(1)),
        )
        ordered = result.order("id")
        win = self.get_column_values(ordered, "win")
        # ages by id: 25, 30, 35, 40, 45
        # id=1: 25+30=55; id=2: 25+30+35=90; id=3: 30+35+40=105; id=4: 35+40+45=120; id=5: 40+45=85
        assert win == [55, 90, 105, 120, 85]

    def test_window_alias_quoted(self, sample_relation: Any) -> None:
        """alias must be quoted via quote_ident; an alias with a double quote must round-trip verbatim."""
        weird = 'weird"alias'
        result = sample_relation.window("COUNT(*)", weird)
        assert weird in result.columns
        assert len(result) == 5

    def test_window_partition_column_quoted(self, connection: Any, relation_class: Any) -> None:
        """partition_by column names must be quoted; a column literally named 'odd col' must work."""
        rel = relation_class.from_dict(
            connection,
            {
                "odd col": ["x", "x", "y"],
                "v": [1, 2, 3],
            },
        )
        result = rel.window("COUNT(*)", "n", partition_by=("odd col",))
        assert isinstance(result, relation_class)
        assert "n" in result.columns

    def test_window_func_passed_verbatim(self, sample_relation: Any) -> None:
        """func is a raw SQL fragment passed verbatim; LEAD(age, 1) must execute as a function call, not be quoted."""
        result = sample_relation.window("LEAD(age, 1)", "next_age", order_by=("id",))
        ordered = result.order("id")
        next_ages = ordered.to_arrow_table().column("next_age").to_pylist()
        assert next_ages == [30, 35, 40, 45, None]

    def test_window_raises_when_alias_already_exists(self, sample_relation: Any) -> None:
        """window must reject an alias colliding with an existing column instead of producing a duplicated name."""
        with pytest.raises(ValueError, match="age"):
            sample_relation.window("COUNT(*)", "age")

    def test_window_raises_on_case_only_collision(self, connection: Any, relation_class: Any) -> None:
        """DuckDB identifiers are case-insensitive: alias 'age' must collide with existing 'Age'."""
        rel = relation_class.from_dict(
            connection,
            {
                "id": [1, 2, 3, 4, 5],
                "Age": [25, 30, 35, 40, 45],
            },
        )
        with pytest.raises(ValueError, match="age"):
            rel.window("COUNT(*)", "age")

    def test_window_order_by_orderby_descending(self, sample_relation: Any) -> None:
        """ROW_NUMBER assigned descending by id must give id=5->rn=1, id=4->rn=2, ..., id=1->rn=5."""
        result = sample_relation.window("ROW_NUMBER()", "rn", order_by=(OrderBy("id", descending=True),))
        ordered = result.order("id")
        assert self.get_column_values(ordered, "rn") == [5, 4, 3, 2, 1]

    def test_window_order_by_mixed_string_and_orderby(self, sample_relation: Any) -> None:
        """Mixed string + OrderBy entries must be honored: partition by category, order by category, id DESC."""
        result = sample_relation.window(
            "ROW_NUMBER()",
            "rn",
            partition_by=("category",),
            order_by=("category", OrderBy("id", descending=True)),
        )
        ordered = result.order("id")
        # cats per id: A,B,A,C,B; per-category rn descending by id:
        # A: id 3 (rn 1), id 1 (rn 2); B: id 5 (rn 1), id 2 (rn 2); C: id 4 (rn 1)
        # so for ids 1..5: 2, 2, 1, 1, 1
        assert self.get_column_values(ordered, "rn") == [2, 2, 1, 1, 1]

    def test_window_order_by_nulls_last_runs(self, sample_relation: Any) -> None:
        """NULLS LAST must render and DuckDB must accept it without error."""
        result = sample_relation.window("ROW_NUMBER()", "rn", order_by=(OrderBy("age", nulls="last"),))
        assert "rn" in result.columns
        assert len(result) == 5

    def test_window_order_by_bare_string_with_space_in_column_name(self, connection: Any, relation_class: Any) -> None:
        """Backward-compat: a bare string must still be treated as a single column name (quoted whole)."""
        rel = relation_class.from_dict(connection, {"odd col": [3, 1, 2], "v": ["a", "b", "c"]})
        result = rel.window("ROW_NUMBER()", "rn", order_by=("odd col",))
        ordered = result.order("rn")
        assert ordered.to_arrow_table().column("odd col").to_pylist() == [1, 2, 3]

    # --- RANGE frame validation: offset bounds require exactly one ORDER BY column ---

    def test_window_range_offset_with_multi_column_order_by_raises(self, sample_relation: Any) -> None:
        """RANGE frame with a Preceding/Following bound requires exactly one ORDER BY column.

        Multi-column ORDER BY plus a RANGE offset bound is rejected by both DuckDB and SQLite
        at execute time; the validator must surface a clear Python ValueError instead.
        """
        with pytest.raises(ValueError) as excinfo:
            sample_relation.window(
                "SUM(age)",
                "s",
                order_by=("id", "category"),
                frame=WindowFrame(kind="range", start=Preceding(1), end=CurrentRow()),
            )
        message = str(excinfo.value)
        assert "RANGE" in message
        assert "ORDER BY" in message

    def test_window_range_offset_with_empty_order_by_raises(self, sample_relation: Any) -> None:
        """RANGE frame with a Preceding/Following bound and no ORDER BY column must raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            sample_relation.window(
                "SUM(age)",
                "s",
                order_by=(),
                frame=WindowFrame(kind="range", start=Preceding(1), end=CurrentRow()),
            )
        message = str(excinfo.value)
        assert "RANGE" in message
        assert "ORDER BY" in message

    def test_window_range_sentinel_only_with_multi_column_order_by_succeeds(self, sample_relation: Any) -> None:
        """RANGE with sentinel-only bounds (Unbounded/CurrentRow) must remain legal regardless of ORDER BY count.

        The validation should only fire when at least one bound is Preceding/Following.
        Counterpart of ``test_window_range_frame_unbounded_to_unbounded`` but with multi-column order_by.
        """
        result = sample_relation.window(
            "SUM(age)",
            "total",
            order_by=("id", "category"),
            frame=WindowFrame(kind="range", start=Unbounded(), end=Unbounded()),
        )
        ordered = result.order("id")
        totals = self.get_column_values(ordered, "total")
        assert totals == [175, 175, 175, 175, 175]
