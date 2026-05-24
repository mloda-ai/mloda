"""Tests for the DuckdbRelation class."""

from typing import Any
import logging

import pytest

from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import (
    CurrentRow,
    DuckdbRelation,
    Following,
    OrderBy,
    Preceding,
    Unbounded,
    WindowFrame,
)
from tests.test_plugins.compute_framework.base_implementations.relation_test_mixin import (
    RelationTestMixin,
)

logger = logging.getLogger(__name__)

try:
    import duckdb
    import pyarrow as pa
except ImportError:
    logger.warning("DuckDB or PyArrow is not installed. Some tests will be skipped.")
    duckdb = None  # type: ignore[assignment]
    pa = None  # type: ignore[assignment]


@pytest.mark.skipif(duckdb is None or pa is None, reason="DuckDB or PyArrow is not installed.")
class TestDuckdbRelation(RelationTestMixin):
    @pytest.fixture
    def sample_relation(self, connection: Any) -> "DuckdbRelation":
        return DuckdbRelation.from_dict(
            connection,
            {
                "id": [1, 2, 3, 4, 5],
                "age": [25, 30, 35, 40, 45],
                "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
                "category": ["A", "B", "A", "C", "B"],
            },
        )

    @pytest.fixture
    def relation_class(self) -> Any:
        return DuckdbRelation

    def get_column_values(self, result: Any, column: str) -> list[Any]:
        return result.df()[column].tolist()  # type: ignore[no-any-return]

    def test_types_exposes_relation_types_aligned_with_columns(self, connection: Any) -> None:
        arrow = pa.table(
            {
                "id": pa.array([1, 2], type=pa.int64()),
                "score": pa.array([1.5, 2.5], type=pa.float64()),
                "name": pa.array(["Alice", "Bob"], type=pa.string()),
                "flag": pa.array([True, False], type=pa.bool_()),
            }
        )
        rel = DuckdbRelation.from_arrow(connection, arrow)

        assert list(zip(rel.columns, (str(dtype) for dtype in rel.types), strict=True)) == [
            ("id", "BIGINT"),
            ("score", "DOUBLE"),
            ("name", "VARCHAR"),
            ("flag", "BOOLEAN"),
        ]

    def test_types_aligned_with_columns_after_filter(self, sample_relation: "DuckdbRelation") -> None:
        filtered = sample_relation.filter("age >= ?", (35,))

        assert list(zip(filtered.columns, (str(dtype) for dtype in filtered.types), strict=True)) == [
            ("id", "BIGINT"),
            ("age", "BIGINT"),
            ("name", "VARCHAR"),
            ("category", "VARCHAR"),
        ]

    def test_types_reordered_after_select_projection(self, connection: Any) -> None:
        arrow = pa.table(
            {
                "id": pa.array([1, 2], type=pa.int64()),
                "score": pa.array([1.5, 2.5], type=pa.float64()),
                "name": pa.array(["Alice", "Bob"], type=pa.string()),
            }
        )
        rel = DuckdbRelation.from_arrow(connection, arrow)

        projected = rel.select("score", "id")

        assert list(zip(projected.columns, (str(dtype) for dtype in projected.types), strict=True)) == [
            ("score", "DOUBLE"),
            ("id", "BIGINT"),
        ]

    def test_types_after_append_column(self, connection: Any) -> None:
        base = DuckdbRelation.from_arrow(connection, pa.table({"id": pa.array([1, 2], type=pa.int64())}))

        result = base.append_column("flag", [True, False])

        assert list(zip(result.columns, (str(dtype) for dtype in result.types), strict=True)) == [
            ("id", "BIGINT"),
            ("flag", "BOOLEAN"),
        ]

    def test_types_after_inner_join(self, connection: Any) -> None:
        left = DuckdbRelation.from_dict(connection, {"idx": [1, 2, 3], "val": ["a", "b", "c"]})
        right = DuckdbRelation.from_dict(connection, {"idx": [2, 3, 4], "score": [10, 20, 30]})

        left_aliased = left.set_alias("left_rel")
        right_aliased = right.set_alias("right_rel")

        joined = left_aliased.join(right_aliased, "left_rel.idx = right_rel.idx", how="inner")

        assert list(zip(joined.columns, (str(dtype) for dtype in joined.types), strict=True)) == [
            ("idx", "BIGINT"),
            ("val", "VARCHAR"),
            ("score", "BIGINT"),
        ]

    # --- Select ---

    def test_select_raw_sql_expression(self, sample_relation: "DuckdbRelation") -> None:
        result = sample_relation.project("*, age * 2 AS doubled_age")
        assert "doubled_age" in result.columns
        assert len(result) == 5

    def test_select_raw_sql_window_function(self, sample_relation: "DuckdbRelation") -> None:
        result = sample_relation.project("*, AVG(age) OVER () AS avg_age")
        assert "avg_age" in result.columns
        assert len(result) == 5

    # DuckDB-specific tests

    def test_stores_connection_and_relation(self, connection: Any) -> None:
        native_rel = connection.from_arrow(pa.table({"x": [1, 2]}))
        rel = DuckdbRelation(connection, native_rel)
        assert rel.connection is connection
        assert rel.columns == ["x"]

    def test_drop_is_noop(self, sample_relation: "DuckdbRelation") -> None:
        sample_relation.drop()
        assert len(sample_relation) == 5

    def test_right_join(self, connection: Any) -> None:
        left = DuckdbRelation.from_dict(connection, {"idx": [1, 2, 3], "val": ["a", "b", "c"]})
        right = DuckdbRelation.from_dict(connection, {"idx": [2, 3, 4], "score": [10, 20, 30]})

        left_aliased = left.set_alias("left_rel")
        right_aliased = right.set_alias("right_rel")

        result = left_aliased.join(right_aliased, "left_rel.idx = right_rel.idx", how="right")
        assert len(result) == 3

    def test_outer_join(self, connection: Any) -> None:
        left = DuckdbRelation.from_dict(connection, {"idx": [1, 2, 3], "val": ["a", "b", "c"]})
        right = DuckdbRelation.from_dict(connection, {"idx": [2, 3, 4], "score": [10, 20, 30]})

        left_aliased = left.set_alias("left_rel")
        right_aliased = right.set_alias("right_rel")

        result = left_aliased.join(right_aliased, "left_rel.idx = right_rel.idx", how="outer")
        assert len(result) == 4

    def test_join_with_sql_injection_alias(self, connection: Any) -> None:
        """A crafted alias must not be interpreted as SQL."""
        left = DuckdbRelation.from_dict(connection, {"idx": [1, 2], "val": ["a", "b"]})
        right = DuckdbRelation.from_dict(connection, {"idx": [1, 2], "score": [10, 20]})

        malicious_alias = 'a" JOIN duckdb_tables() --'
        left_aliased = left.set_alias(malicious_alias)
        right_aliased = right.set_alias("right_rel")

        result = left_aliased.join(right_aliased, f"{malicious_alias}.idx = right_rel.idx", how="inner")
        assert len(result) == 2
        assert "val" in result.columns
        assert "score" in result.columns

    # --- Order ---

    def test_order_single_column(self, sample_relation: "DuckdbRelation") -> None:
        ordered = sample_relation.order("age")
        ids = self.get_column_values(ordered, "id")
        assert ids == [1, 2, 3, 4, 5]

    def test_order_single_column_desc(self, sample_relation: "DuckdbRelation") -> None:
        ordered = sample_relation.order("age DESC")
        ids = self.get_column_values(ordered, "id")
        assert ids == [5, 4, 3, 2, 1]

    def test_order_multiple_columns(self, sample_relation: "DuckdbRelation") -> None:
        ordered = sample_relation.order("category", "age")
        categories = self.get_column_values(ordered, "category")
        ages = self.get_column_values(ordered, "age")
        assert categories == ["A", "A", "B", "B", "C"]
        assert ages == [25, 35, 30, 45, 40]

    def test_order_preserves_all_columns(self, sample_relation: "DuckdbRelation") -> None:
        ordered = sample_relation.order("age")
        assert ordered.columns == ["id", "age", "name", "category"]

    def test_order_preserves_row_count(self, sample_relation: "DuckdbRelation") -> None:
        ordered = sample_relation.order("age")
        assert len(ordered) == 5

    def test_order_does_not_mutate_original(self, sample_relation: "DuckdbRelation") -> None:
        sample_relation.order("age DESC")
        ids = self.get_column_values(sample_relation, "id")
        assert ids == [1, 2, 3, 4, 5]

    def test_order_returns_duckdb_relation(self, sample_relation: "DuckdbRelation") -> None:
        ordered = sample_relation.order("age")
        assert isinstance(ordered, DuckdbRelation)

    # --- Project (public raw projection) ---

    def test_project_basic_expression(self, sample_relation: "DuckdbRelation") -> None:
        result = sample_relation.project("id, age * 2 AS double_age")
        ages = self.get_column_values(result, "double_age")
        assert ages == [50, 60, 70, 80, 90]

    def test_project_returns_duckdb_relation(self, sample_relation: "DuckdbRelation") -> None:
        result = sample_relation.project("id")
        assert isinstance(result, DuckdbRelation)

    def test_project_preserves_row_count(self, sample_relation: "DuckdbRelation") -> None:
        result = sample_relation.project("id, age")
        assert len(result) == 5

    # --- Aggregate ---

    def test_aggregate_with_group_by(self, sample_relation: "DuckdbRelation") -> None:
        result = sample_relation.aggregate("category, COUNT(*) AS n", "category").order("category")
        cats = self.get_column_values(result, "category")
        counts = self.get_column_values(result, "n")
        assert cats == ["A", "B", "C"]
        assert counts == [2, 2, 1]

    def test_aggregate_without_group_by(self, sample_relation: "DuckdbRelation") -> None:
        result = sample_relation.aggregate("COUNT(*) AS total")
        assert self.get_column_values(result, "total") == [5]

    def test_aggregate_returns_duckdb_relation(self, sample_relation: "DuckdbRelation") -> None:
        result = sample_relation.aggregate("COUNT(*) AS total")
        assert isinstance(result, DuckdbRelation)

    # --- Query ---

    def test_query_basic(self, sample_relation: "DuckdbRelation") -> None:
        result = sample_relation.query("data", "SELECT id, age FROM data WHERE age >= 35 ORDER BY id")
        ids = self.get_column_values(result, "id")
        assert ids == [3, 4, 5]

    def test_query_with_window_function(self, sample_relation: "DuckdbRelation") -> None:
        result = sample_relation.query(
            "data",
            "SELECT id, ROW_NUMBER() OVER (PARTITION BY category ORDER BY age) AS rn FROM data ORDER BY id",
        )
        rns = self.get_column_values(result, "rn")
        assert rns == [1, 1, 2, 1, 2]

    def test_query_returns_duckdb_relation(self, sample_relation: "DuckdbRelation") -> None:
        result = sample_relation.query("data", "SELECT * FROM data LIMIT 1")
        assert isinstance(result, DuckdbRelation)

    # --- Injection-contract (verbatim passthrough) ---

    def test_project_passes_expression_verbatim(self, sample_relation: "DuckdbRelation") -> None:
        """project() must pass the caller's expression unchanged; quoting would turn function calls into string literals."""
        result = sample_relation.project("UPPER(name) AS upper_name")
        assert self.get_column_values(result, "upper_name") == ["ALICE", "BOB", "CHARLIE", "DAVID", "EVE"]

    def test_aggregate_passes_expression_verbatim(self, sample_relation: "DuckdbRelation") -> None:
        """aggregate() must pass the caller's SQL fragment unchanged; quoting would break FILTER (WHERE ...) syntax."""
        result = sample_relation.aggregate("COUNT(*) FILTER (WHERE age > 30) AS over_30")
        assert self.get_column_values(result, "over_30") == [3]

    def test_query_passes_sql_verbatim(self, sample_relation: "DuckdbRelation") -> None:
        """query() must pass the caller's SQL string unchanged; quoting would prevent execution of the SELECT statement."""
        result = sample_relation.query("t", "SELECT UPPER(name) AS upper_name FROM t ORDER BY upper_name")
        assert self.get_column_values(result, "upper_name") == ["ALICE", "BOB", "CHARLIE", "DAVID", "EVE"]

    # --- with_row_number ---

    def test_with_row_number_bare_over(self, sample_relation: "DuckdbRelation") -> None:
        """Empty partition_by and order_by produce a bare ROW_NUMBER() OVER () clause."""
        result = sample_relation.with_row_number("rn")
        assert "rn" in result.columns
        assert len(result) == 5
        rns = self.get_column_values(result, "rn")
        # DuckDB does not guarantee assignment order with bare OVER (), so test set equality
        assert sorted(rns) == [1, 2, 3, 4, 5]

    def test_with_row_number_partition_by_only(self, sample_relation: "DuckdbRelation") -> None:
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

    def test_with_row_number_order_by_only(self, sample_relation: "DuckdbRelation") -> None:
        """order_by without partition_by: rows ordered by age get rn 1..5."""
        result = sample_relation.with_row_number("rn", order_by=("age",))
        assert "rn" in result.columns
        # Order the result by age and verify rn == [1,2,3,4,5]
        ordered = result.order("age")
        rns = self.get_column_values(ordered, "rn")
        assert rns == [1, 2, 3, 4, 5]

    def test_with_row_number_partition_and_order(self, sample_relation: "DuckdbRelation") -> None:
        """partition_by=('category',), order_by=('age',). Verify expected rn per row by id."""
        result = sample_relation.with_row_number("rn", partition_by=("category",), order_by=("age",))
        ordered = result.order("id")
        ids = self.get_column_values(ordered, "id")
        rns = self.get_column_values(ordered, "rn")
        assert ids == [1, 2, 3, 4, 5]
        # id=1 (A,25)->1, id=2 (B,30)->1, id=3 (A,35)->2, id=4 (C,40)->1, id=5 (B,45)->2
        assert rns == [1, 1, 2, 1, 2]

    def test_with_row_number_returns_new_relation(self, sample_relation: "DuckdbRelation") -> None:
        """Original relation is not mutated; returned object is a DuckdbRelation instance."""
        original_columns = list(sample_relation.columns)
        result = sample_relation.with_row_number("rn", order_by=("age",))
        assert isinstance(result, DuckdbRelation)
        assert sample_relation.columns == original_columns
        assert "rn" not in sample_relation.columns

    def test_with_row_number_alias_with_special_chars(self, sample_relation: "DuckdbRelation") -> None:
        """The method must quote the alias; an alias containing a double quote must round-trip verbatim."""
        weird = 'weird"name'
        result = sample_relation.with_row_number(weird, order_by=("age",))
        assert weird in result.columns
        assert len(result) == 5

    def test_with_row_number_partition_column_with_special_chars(self, connection: Any) -> None:
        """The method must quote partition_by columns; a column literally named 'odd col' must work."""
        rel = DuckdbRelation.from_dict(
            connection,
            {
                "odd col": ["x", "x", "y"],
                "v": [1, 2, 3],
            },
        )
        result = rel.with_row_number("rn", partition_by=("odd col",))
        assert "rn" in result.columns
        assert len(result) == 3

    def test_with_row_number_preserves_original_columns(self, sample_relation: "DuckdbRelation") -> None:
        """Original columns appear first (in original order); the new column appears last."""
        original_columns = list(sample_relation.columns)
        result = sample_relation.with_row_number("rn", order_by=("age",))
        assert result.columns[: len(original_columns)] == original_columns
        assert result.columns[-1] == "rn"

    def test_with_row_number_partition_by_keyword_only(self, sample_relation: "DuckdbRelation") -> None:
        """partition_by must be keyword-only to match the sibling window() API."""
        with pytest.raises(TypeError):
            sample_relation.with_row_number("rn", ("category",))  # type: ignore[misc]

    def test_with_row_number_order_by_keyword_only(self, sample_relation: "DuckdbRelation") -> None:
        """order_by must be keyword-only to match the sibling window() API."""
        with pytest.raises(TypeError):
            sample_relation.with_row_number("rn", (), ("age",))  # type: ignore[misc]

    # --- append_column: helper-name collision (issue #405 subtask 2) ---

    def test_append_column_when_existing_column_named_mloda_rn_zero(self, connection: Any) -> None:
        """Helper picker must skip __mloda_rn0__ if already present and use __mloda_rn1__."""
        rel = DuckdbRelation.from_dict(connection, {"__mloda_rn0__": [1, 2, 3], "b": [4, 5, 6]})
        result = rel.append_column("c", [7, 8, 9])
        assert set(result.columns) == {"__mloda_rn0__", "b", "c"}
        arrow = result.to_arrow_table()
        assert arrow.column("__mloda_rn0__").to_pylist() == [1, 2, 3]
        assert arrow.column("c").to_pylist() == [7, 8, 9]

    def test_append_column_when_existing_column_named_mloda_rn_legacy(self, connection: Any) -> None:
        """Helper picker must not collide with a pre-existing __mloda_rn__ column (the OLD hardcoded name)."""
        rel = DuckdbRelation.from_dict(connection, {"__mloda_rn__": ["x", "y", "z"], "b": [4, 5, 6]})
        result = rel.append_column("c", [7, 8, 9])
        assert set(result.columns) == {"__mloda_rn__", "b", "c"}
        arrow = result.to_arrow_table()
        assert arrow.column("__mloda_rn__").to_pylist() == ["x", "y", "z"]
        assert arrow.column("c").to_pylist() == [7, 8, 9]

    def test_append_column_when_name_param_collides_with_helper_candidate(self, connection: Any) -> None:
        """Helper picker must consider both self.columns AND the incoming name parameter."""
        rel = DuckdbRelation.from_dict(connection, {"a": [1, 2, 3]})
        result = rel.append_column("__mloda_rn0__", [7, 8, 9])
        assert set(result.columns) == {"a", "__mloda_rn0__"}
        assert len(result) == 3
        arrow = result.to_arrow_table()
        assert arrow.column("__mloda_rn0__").to_pylist() == [7, 8, 9]

    def test_append_column_when_multiple_helper_candidates_exist(self, connection: Any) -> None:
        """Helper picker must scan upward and pick the lowest free __mloda_rn{n}__."""
        rel = DuckdbRelation.from_dict(
            connection,
            {"__mloda_rn0__": [1, 2], "__mloda_rn1__": [3, 4], "x": [5, 6]},
        )
        result = rel.append_column("y", [7, 8])
        assert set(result.columns) == {"__mloda_rn0__", "__mloda_rn1__", "x", "y"}
        arrow = result.to_arrow_table()
        assert arrow.column("__mloda_rn0__").to_pylist() == [1, 2]
        assert arrow.column("__mloda_rn1__").to_pylist() == [3, 4]
        assert arrow.column("x").to_pylist() == [5, 6]
        assert arrow.column("y").to_pylist() == [7, 8]

    # --- append_column / with_row_number / window: alias collision with existing column ---

    def test_append_column_raises_when_name_already_exists(self, connection: Any) -> None:
        """append_column must reject ``name`` colliding with an existing column instead of silently corrupting the schema."""
        rel = DuckdbRelation.from_dict(connection, {"a": [1, 2, 3], "b": [4, 5, 6]})
        with pytest.raises(ValueError, match="b"):
            rel.append_column("b", [10, 20, 30])

    def test_with_row_number_raises_when_alias_already_exists(self, sample_relation: "DuckdbRelation") -> None:
        """with_row_number must reject an alias colliding with an existing column instead of producing a duplicated name."""
        with pytest.raises(ValueError, match="category"):
            sample_relation.with_row_number("category")

    def test_window_raises_when_alias_already_exists(self, sample_relation: "DuckdbRelation") -> None:
        """window must reject an alias colliding with an existing column instead of producing a duplicated name."""
        with pytest.raises(ValueError, match="age"):
            sample_relation.window("COUNT(*)", "age")

    def test_append_column_raises_on_case_only_collision(self, connection: Any) -> None:
        """DuckDB identifiers are case-insensitive: 'foo' must collide with existing 'Foo'."""
        rel = DuckdbRelation.from_dict(connection, {"Foo": [1, 2, 3], "b": [4, 5, 6]})
        with pytest.raises(ValueError, match="foo"):
            rel.append_column("foo", [10, 20, 30])

    def test_with_row_number_raises_on_case_only_collision(self, connection: Any) -> None:
        """DuckDB identifiers are case-insensitive: alias 'category' must collide with existing 'Category'."""
        rel = DuckdbRelation.from_dict(
            connection,
            {
                "id": [1, 2, 3, 4, 5],
                "Category": ["A", "B", "A", "C", "B"],
            },
        )
        with pytest.raises(ValueError, match="category"):
            rel.with_row_number("category")

    def test_window_raises_on_case_only_collision(self, connection: Any) -> None:
        """DuckDB identifiers are case-insensitive: alias 'age' must collide with existing 'Age'."""
        rel = DuckdbRelation.from_dict(
            connection,
            {
                "id": [1, 2, 3, 4, 5],
                "Age": [25, 30, 35, 40, 45],
            },
        )
        with pytest.raises(ValueError, match="age"):
            rel.window("COUNT(*)", "age")

    # --- window() ---

    def test_window_partition_only(self, sample_relation: "DuckdbRelation") -> None:
        """SUM(age) partitioned by category, no order, no frame: each row gets its category total."""
        result = sample_relation.window("SUM(age)", "cat_sum", partition_by=("category",))
        assert "cat_sum" in result.columns
        ordered = result.order("id")
        sums = self.get_column_values(ordered, "cat_sum")
        # Categories per id (1..5): A,B,A,C,B; sums: A=25+35=60, B=30+45=75, C=40
        assert sums == [60, 75, 60, 40, 75]

    def test_window_partition_and_order_running_sum(self, sample_relation: "DuckdbRelation") -> None:
        """SUM(age) partitioned by category, ordered by id, default RANGE UNBOUNDED PRECEDING..CURRENT ROW."""
        result = sample_relation.window("SUM(age)", "rs", partition_by=("category",), order_by=("id",))
        ordered = result.order("id")
        rs = self.get_column_values(ordered, "rs")
        assert rs == [25, 30, 60, 40, 75]

    def test_window_no_partition_no_order(self, sample_relation: "DuckdbRelation") -> None:
        """COUNT(*) with no partition / order / frame returns total row count for every row."""
        result = sample_relation.window("COUNT(*)", "n")
        ns = self.get_column_values(result, "n")
        assert ns == [5, 5, 5, 5, 5]

    def test_window_returns_new_relation_and_preserves_columns(self, sample_relation: "DuckdbRelation") -> None:
        """window() returns a new DuckdbRelation with original columns plus alias; original is untouched."""
        original_columns = list(sample_relation.columns)
        result = sample_relation.window("COUNT(*)", "n")
        assert isinstance(result, DuckdbRelation)
        for col in original_columns:
            assert col in result.columns
        assert "n" in result.columns
        # Original relation untouched
        assert sample_relation.columns == original_columns
        assert "n" not in sample_relation.columns

    def test_window_rows_frame_unbounded_to_current(self, sample_relation: "DuckdbRelation") -> None:
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

    def test_window_range_frame_unbounded_to_unbounded(self, sample_relation: "DuckdbRelation") -> None:
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

    def test_window_groups_frame_preceding_to_current(self, sample_relation: "DuckdbRelation") -> None:
        """GROUPS BETWEEN 1 PRECEDING AND CURRENT ROW: just verify the frame variant compiles and produces a column."""
        result = sample_relation.window(
            "SUM(age)",
            "g",
            order_by=("category",),
            frame=WindowFrame(kind="groups", start=Preceding(1), end=CurrentRow()),
        )
        assert "g" in result.columns
        assert len(result) == 5

    def test_window_rows_frame_preceding_and_following(self, sample_relation: "DuckdbRelation") -> None:
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

    def test_window_alias_quoted(self, sample_relation: "DuckdbRelation") -> None:
        """alias must be quoted via quote_ident; an alias with a double quote must round-trip verbatim."""
        weird = 'weird"alias'
        result = sample_relation.window("COUNT(*)", weird)
        assert weird in result.columns
        assert len(result) == 5

    def test_window_partition_column_quoted(self, connection: Any) -> None:
        """partition_by column names must be quoted; a column literally named 'odd col' must work."""
        rel = DuckdbRelation.from_dict(
            connection,
            {
                "odd col": ["x", "x", "y"],
                "v": [1, 2, 3],
            },
        )
        result = rel.window("COUNT(*)", "n", partition_by=("odd col",))
        assert isinstance(result, DuckdbRelation)
        assert "n" in result.columns

    def test_window_func_passed_verbatim(self, sample_relation: "DuckdbRelation") -> None:
        """func is a raw SQL fragment passed verbatim; LEAD(age, 1) must execute as a function call, not be quoted."""
        result = sample_relation.window("LEAD(age, 1)", "next_age", order_by=("id",))
        ordered = result.order("id")
        next_ages = ordered.to_arrow_table().column("next_age").to_pylist()
        assert next_ages == [30, 35, 40, 45, None]

    # --- OrderBy ---

    def test_orderby_bare_column_constructs(self) -> None:
        """OrderBy('age') must construct cleanly."""
        OrderBy("age")

    def test_orderby_descending_true_constructs(self) -> None:
        """OrderBy('age', descending=True) must construct cleanly."""
        OrderBy("age", descending=True)

    def test_orderby_nulls_first_constructs(self) -> None:
        """OrderBy('age', nulls='first') must construct cleanly."""
        OrderBy("age", nulls="first")

    def test_orderby_nulls_last_constructs(self) -> None:
        """OrderBy('age', nulls='last') must construct cleanly."""
        OrderBy("age", nulls="last")

    def test_orderby_invalid_nulls_raises(self) -> None:
        """OrderBy must reject a nulls value outside {'first','last'} at construction time."""
        with pytest.raises(ValueError, match="nulls"):
            OrderBy("age", nulls="middle")  # type: ignore[arg-type]

    def test_window_order_by_orderby_descending(self, sample_relation: "DuckdbRelation") -> None:
        """ROW_NUMBER assigned descending by id must give id=5->rn=1, id=4->rn=2, ..., id=1->rn=5."""
        result = sample_relation.window(
            "ROW_NUMBER()", "rn", order_by=(OrderBy("id", descending=True),)
        )
        ordered = result.order("id")
        assert self.get_column_values(ordered, "rn") == [5, 4, 3, 2, 1]

    def test_with_row_number_order_by_orderby_descending(self, sample_relation: "DuckdbRelation") -> None:
        """with_row_number must honor OrderBy(descending=True) and produce the descending row numbers."""
        result = sample_relation.with_row_number("rn", order_by=(OrderBy("id", descending=True),))
        ordered = result.order("id")
        assert self.get_column_values(ordered, "rn") == [5, 4, 3, 2, 1]

    def test_window_order_by_mixed_string_and_orderby(self, sample_relation: "DuckdbRelation") -> None:
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

    def test_window_order_by_nulls_last_runs(self, sample_relation: "DuckdbRelation") -> None:
        """NULLS LAST must render and DuckDB must accept it without error."""
        result = sample_relation.window("ROW_NUMBER()", "rn", order_by=(OrderBy("age", nulls="last"),))
        assert "rn" in result.columns
        assert len(result) == 5

    def test_window_order_by_bare_string_with_space_in_column_name(self, connection: Any) -> None:
        """Backward-compat: a bare string must still be treated as a single column name (quoted whole)."""
        rel = DuckdbRelation.from_dict(connection, {"odd col": [3, 1, 2], "v": ["a", "b", "c"]})
        result = rel.window("ROW_NUMBER()", "rn", order_by=("odd col",))
        ordered = result.order("rn")
        assert ordered.to_arrow_table().column("odd col").to_pylist() == [1, 2, 3]

    # --- runtime validation on frame dataclasses ---

    def test_window_frame_kind_invalid_string_raises(self) -> None:
        """WindowFrame must reject any kind outside the Literal alphabet at construction time."""
        with pytest.raises(ValueError, match="kind"):
            WindowFrame(kind="invalid", start=Unbounded(), end=Unbounded())  # type: ignore[arg-type]

    def test_window_frame_kind_uppercase_rejected(self) -> None:
        """WindowFrame must reject 'ROWS' even though .upper() would render it harmlessly — the Literal alphabet is lowercase."""
        with pytest.raises(ValueError, match="kind"):
            WindowFrame(kind="ROWS", start=Unbounded(), end=Unbounded())  # type: ignore[arg-type]

    def test_window_frame_kind_sql_injection_attempt_raises(self) -> None:
        """Construction-time validation must block the canonical injection payload before it can reach the SQL renderer."""
        with pytest.raises(ValueError, match="kind"):
            WindowFrame(
                kind="ROWS) AS x; DROP TABLE y; --",  # type: ignore[arg-type]
                start=Unbounded(),
                end=Unbounded(),
            )

    def test_window_frame_accepts_each_valid_kind(self) -> None:
        """The three valid kinds — rows, range, groups — must all construct without error."""
        for kind in ("rows", "range", "groups"):
            WindowFrame(kind=kind, start=Unbounded(), end=Unbounded())

    def test_preceding_offset_string_raises(self) -> None:
        """Preceding must reject a non-int offset; otherwise a string payload would land verbatim in the f-string."""
        with pytest.raises(TypeError, match="offset"):
            Preceding(offset="1; DROP TABLE x; --")  # type: ignore[arg-type]

    def test_preceding_offset_float_raises(self) -> None:
        """Preceding must reject a float offset; the Literal-equivalent contract is int only."""
        with pytest.raises(TypeError, match="offset"):
            Preceding(offset=1.5)  # type: ignore[arg-type]

    def test_preceding_offset_bool_raises(self) -> None:
        """Preceding must reject bool — isinstance(True, int) is True in Python but bool violates the int contract."""
        with pytest.raises(TypeError, match="offset"):
            Preceding(offset=True)

    def test_preceding_offset_int_accepted(self) -> None:
        """Preceding(offset=1) must construct cleanly."""
        Preceding(offset=1)

    def test_following_offset_string_raises(self) -> None:
        """Following must reject a non-int offset (parallel to Preceding)."""
        with pytest.raises(TypeError, match="offset"):
            Following(offset="1; DROP TABLE x; --")  # type: ignore[arg-type]

    def test_following_offset_float_raises(self) -> None:
        """Following must reject a float offset."""
        with pytest.raises(TypeError, match="offset"):
            Following(offset=2.0)  # type: ignore[arg-type]

    def test_following_offset_bool_raises(self) -> None:
        """Following must reject bool — same reason as Preceding."""
        with pytest.raises(TypeError, match="offset"):
            Following(offset=False)

    def test_following_offset_int_accepted(self) -> None:
        """Following(offset=2) must construct cleanly."""
        Following(offset=2)

    def test_preceding_offset_negative_raises(self) -> None:
        """Preceding must reject a negative offset; SQL n PRECEDING requires n >= 0."""
        with pytest.raises(ValueError, match="offset|negative"):
            Preceding(offset=-1)

    def test_following_offset_negative_raises(self) -> None:
        """Following must reject a negative offset; SQL n FOLLOWING requires n >= 0."""
        with pytest.raises(ValueError, match="offset|negative"):
            Following(offset=-1)

    def test_preceding_offset_zero_accepted(self) -> None:
        """Preceding(offset=0) must construct cleanly; 0 PRECEDING is valid SQL equivalent to CURRENT ROW."""
        Preceding(offset=0)

    def test_following_offset_zero_accepted(self) -> None:
        """Following(offset=0) must construct cleanly; 0 FOLLOWING is valid SQL equivalent to CURRENT ROW."""
        Following(offset=0)

    def test_window_frame_following_then_preceding_raises(self) -> None:
        """WindowFrame must reject start=Following(2), end=Preceding(1); start must not be after end on the row axis."""
        with pytest.raises(ValueError, match="start|end|order"):
            WindowFrame(kind="rows", start=Following(2), end=Preceding(1))

    def test_window_frame_preceding_descending_raises(self) -> None:
        """WindowFrame must reject Preceding(1) -> Preceding(3); -1 > -3 on the row axis."""
        with pytest.raises(ValueError, match="start|end|order"):
            WindowFrame(kind="rows", start=Preceding(1), end=Preceding(3))

    def test_window_frame_following_descending_raises(self) -> None:
        """WindowFrame must reject Following(3) -> Following(1); 3 > 1 on the row axis."""
        with pytest.raises(ValueError, match="start|end|order"):
            WindowFrame(kind="rows", start=Following(3), end=Following(1))

    def test_window_frame_current_row_to_current_row_accepted(self) -> None:
        """WindowFrame(CurrentRow, CurrentRow) must construct cleanly; equality on the row axis is allowed."""
        WindowFrame(kind="rows", start=CurrentRow(), end=CurrentRow())

    def test_window_frame_unbounded_both_sides_accepted(self) -> None:
        """WindowFrame(Unbounded, Unbounded) must construct cleanly; -inf < +inf."""
        WindowFrame(kind="rows", start=Unbounded(), end=Unbounded())

    def test_window_frame_preceding_to_following_accepted(self) -> None:
        """WindowFrame(Preceding(2), Following(3)) must construct cleanly; -2 < 3."""
        WindowFrame(kind="rows", start=Preceding(2), end=Following(3))
