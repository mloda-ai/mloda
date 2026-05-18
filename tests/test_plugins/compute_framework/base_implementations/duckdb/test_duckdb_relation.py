"""Tests for the DuckdbRelation class."""

from typing import Any
import logging

import pytest

from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
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
