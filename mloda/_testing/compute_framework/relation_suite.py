"""Conformance suites for SqlBaseRelation implementations, run through a framework adapter."""

from __future__ import annotations

from typing import Any

import pytest

from mloda._testing.compute_framework.host import ComputeFrameworkSuite
from mloda._testing.framework_adapter import FrameworkAdapter
from mloda._testing.typed_data import TypedTestData
from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.optional_dependency import require
from mloda_plugins.compute_framework.base_implementations.sql.sql_window import (
    CurrentRow,
    Following,
    OrderBy,
    Preceding,
    Unbounded,
    WindowFrame,
)

SAMPLE_DATA = TypedTestData(
    {
        "id": (DataType.INT64, [1, 2, 3, 4, 5]),
        "age": (DataType.INT64, [25, 30, 35, 40, 45]),
        "name": (DataType.STRING, ["Alice", "Bob", "Charlie", "David", "Eve"]),
        "category": (DataType.STRING, ["A", "B", "A", "C", "B"]),
    }
)


class RelationSuite(ComputeFrameworkSuite, suite_base=True):
    """Contract tests of the SQL relation of the host's compute framework."""

    aspect = "sql_relation"

    def _serves(self, adapter: FrameworkAdapter, framework: type[ComputeFramework]) -> bool:
        return adapter.capabilities.sql_relation

    @pytest.fixture
    def sample_relation(self, adapter: FrameworkAdapter, framework: type[ComputeFramework]) -> Any:
        return self.frame(adapter, "sample", SAMPLE_DATA)

    @pytest.fixture
    def relation_class(self, framework: type[ComputeFramework]) -> Any:
        return framework.expected_data_framework()

    @pytest.fixture
    def connection(self, adapter: FrameworkAdapter, framework: type[ComputeFramework]) -> Any:
        return adapter.connection

    def raw_projection(self, relation: Any, sql: str) -> Any:
        return relation.select(_raw_sql=sql)

    # --- Basics ---

    def test_columns(self, sample_relation: Any) -> None:
        assert sample_relation.columns == ["id", "age", "name", "category"]

    def test_len(self, sample_relation: Any) -> None:
        assert len(sample_relation) == 5

    def test_to_arrow_table(self, sample_relation: Any) -> None:
        pa = require("pyarrow", "building Arrow tables")
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
        selected = self.raw_projection(sample_relation, "*, age * 2 AS doubled_age")
        assert "doubled_age" in selected.columns
        assert len(selected) == 5

    def test_select_raw_sql_window_function(self, sample_relation: Any) -> None:
        selected = self.raw_projection(sample_relation, "*, AVG(age) OVER () AS avg_age")
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
        pa = require("pyarrow", "building Arrow tables")
        original = pa.Table.from_pydict({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        rel = relation_class.from_arrow(connection, original)
        result = rel.to_arrow_table()
        assert result.num_rows == 3
        assert set(result.column_names) == {"a", "b"}

    def test_from_arrow_empty_table(self, connection: Any, relation_class: Any) -> None:
        pa = require("pyarrow", "building Arrow tables")
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

    # --- Append Column ---

    def test_append_column_adds_new_column(self, connection: Any, relation_class: Any) -> None:
        rel = relation_class.from_dict(connection, {"a": [1, 2, 3], "b": [4, 5, 6]})
        result = rel.append_column("c", [7, 8, 9])
        assert set(result.columns) == {"a", "b", "c"}
        assert len(result) == 3

    def test_append_column_preserves_existing_data(self, connection: Any, relation_class: Any) -> None:
        rel = relation_class.from_dict(connection, {"a": [1, 2, 3], "b": [4, 5, 6]})
        result = rel.append_column("c", [7, 8, 9])
        arrow = result.to_arrow_table()
        assert arrow.column("a").to_pylist() == [1, 2, 3]
        assert arrow.column("b").to_pylist() == [4, 5, 6]
        assert arrow.column("c").to_pylist() == [7, 8, 9]

    def test_append_column_does_not_mutate_original(self, connection: Any, relation_class: Any) -> None:
        rel = relation_class.from_dict(connection, {"a": [1, 2, 3], "b": [4, 5, 6]})
        rel.append_column("c", [7, 8, 9])
        assert rel.columns == ["a", "b"]

    def test_append_column_after_filter(self, connection: Any, relation_class: Any) -> None:
        rel = relation_class.from_dict(connection, {"a": [1, 2, 3], "b": [4, 5, 6]})
        filtered = rel.filter("a >= ?", (2,))
        result = filtered.append_column("c", [10, 20])
        assert set(result.columns) == {"a", "b", "c"}
        assert len(result) == 2

    def test_append_column_after_select(self, connection: Any, relation_class: Any) -> None:
        rel = relation_class.from_dict(connection, {"a": [1, 2, 3], "b": [4, 5, 6]})
        projected = rel.select("a")
        result = projected.append_column("c", [7, 8, 9])
        assert set(result.columns) == {"a", "c"}
        assert len(result) == 3

    def test_append_columns_keeps_column_and_row_order(self, connection: Any, relation_class: Any) -> None:
        rel = relation_class.from_dict(connection, {"a": [1, 2, 3], "b": ["x", "y", "z"]})
        result = rel.append_columns({"x": [7, 8, 9], "y": [True, False, True]})
        assert result.columns == ["a", "b", "x", "y"]
        arrow = result.to_arrow_table()
        assert arrow.column("a").to_pylist() == [1, 2, 3]
        assert arrow.column("b").to_pylist() == ["x", "y", "z"]
        assert arrow.column("x").to_pylist() == [7, 8, 9]
        assert arrow.column("y").to_pylist() == [True, False, True]

    # --- append_column: helper-name collision and name-collision contract ---

    def test_append_column_when_name_param_collides_with_helper_candidate(
        self, connection: Any, relation_class: Any
    ) -> None:
        """Helper picker must consider both self.columns AND the incoming name parameter."""
        rel = relation_class.from_dict(connection, {"a": [1, 2, 3]})
        result = rel.append_column("__mloda_rn0__", [7, 8, 9])
        assert set(result.columns) == {"a", "__mloda_rn0__"}
        assert len(result) == 3
        arrow = result.to_arrow_table()
        assert arrow.column("__mloda_rn0__").to_pylist() == [7, 8, 9]

    def test_append_column_when_multiple_helper_candidates_exist(self, connection: Any, relation_class: Any) -> None:
        """Helper picker must scan upward and pick the lowest free __mloda_rn{n}__."""
        rel = relation_class.from_dict(
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

    def test_append_column_raises_when_name_already_exists(self, connection: Any, relation_class: Any) -> None:
        """append_column must reject ``name`` colliding with an existing column instead of silently corrupting the schema."""
        rel = relation_class.from_dict(connection, {"a": [1, 2, 3], "b": [4, 5, 6]})
        with pytest.raises(ValueError, match="b"):
            rel.append_column("b", [10, 20, 30])

    def test_append_column_raises_on_case_only_collision(self, connection: Any, relation_class: Any) -> None:
        """SQL identifiers are case-insensitive: 'foo' must collide with existing 'Foo'."""
        rel = relation_class.from_dict(connection, {"Foo": [1, 2, 3], "b": [4, 5, 6]})
        with pytest.raises(ValueError, match="foo"):
            rel.append_column("foo", [10, 20, 30])

    # --- Join: reserved-word column name ---

    def test_join_bare_column_reserved_word(self, connection: Any, relation_class: Any) -> None:
        """Joining on a bare column name that is a SQL reserved word must work."""
        left = relation_class.from_dict(connection, {"order": [1, 2, 3], "val": ["a", "b", "c"]})
        right = relation_class.from_dict(connection, {"order": [2, 3, 4], "score": [10, 20, 30]})
        left_aliased = left.set_alias("left_rel")
        right_aliased = right.set_alias("right_rel")
        result = left_aliased.join(right_aliased, "order", how="inner")
        assert len(result) == 2

    # --- Exact-case identifiers ---

    def test_append_column_non_ascii_case_variant_is_distinct(self, connection: Any, relation_class: Any) -> None:
        rel = relation_class.from_dict(connection, {"É": [1, 2, 3]})
        result = rel.append_column("é", [7, 8, 9])
        arrow = result.to_arrow_table()
        assert arrow.column("É").to_pylist() == [1, 2, 3]
        assert arrow.column("é").to_pylist() == [7, 8, 9]

    @pytest.mark.parametrize("how", ["inner", "left", "right", "outer"])
    def test_join_case_only_output_collision_raises(self, connection: Any, relation_class: Any, how: str) -> None:
        left = relation_class.from_dict(connection, {"id": [1, 2], "Val": [10, 20]}).set_alias("l")
        right = relation_class.from_dict(connection, {"id": [1, 2], "val": [30, 40]}).set_alias("r")
        with pytest.raises(ValueError, match="rename"):
            left.join(right, "l.id = r.id", how=how)

    def test_join_keys_differing_only_by_case_raises(self, connection: Any, relation_class: Any) -> None:
        left = relation_class.from_dict(connection, {"ID": [1, 2], "x": [10, 20]}).set_alias("l")
        right = relation_class.from_dict(connection, {"id": [1, 2], "y": [30, 40]}).set_alias("r")
        with pytest.raises(ValueError, match="rename"):
            left.join(right, "l.ID = r.id", how="inner")

    @pytest.mark.parametrize("bad", ["missing", "VAL"])
    def test_select_missing_or_case_mismatched_name_raises(
        self, connection: Any, relation_class: Any, bad: str
    ) -> None:
        rel = relation_class.from_dict(connection, {"id": [1, 2], "val": [10, 20]})
        with pytest.raises(ValueError, match=bad):
            rel.select("id", bad)

    def test_join_non_ascii_case_variants_are_distinct(self, connection: Any, relation_class: Any) -> None:
        left = relation_class.from_dict(connection, {"id": [1, 2], "é": [10, 20]}).set_alias("l")
        right = relation_class.from_dict(connection, {"id": [1, 2], "É": [30, 40]}).set_alias("r")
        result = left.join(right, "l.id = r.id", how="inner")
        arrow = result.to_arrow_table()
        assert arrow.column("é").to_pylist() == [10, 20]
        assert arrow.column("É").to_pylist() == [30, 40]

    def test_from_arrow_case_only_collision_raises(self, connection: Any, relation_class: Any) -> None:
        pa = require("pyarrow", "building Arrow tables")
        table = pa.Table.from_arrays([pa.array([1, 2]), pa.array([3, 4])], names=["a", "A"])
        with pytest.raises(ValueError, match="rename"):
            relation_class.from_arrow(connection, table)

    def test_from_dict_case_only_collision_raises(self, connection: Any, relation_class: Any) -> None:
        with pytest.raises(ValueError, match="rename"):
            relation_class.from_dict(connection, {"a": [1, 2], "A": [3, 4]})


class SqlRelationWindowSuite(RelationSuite, suite_base=True):
    """Contract tests of the window helpers of the host's SQL relation."""

    # --- with_row_number ---

    def test_with_row_number_bare_over(self, adapter: FrameworkAdapter, sample_relation: Any) -> None:
        """Empty partition_by and order_by produce a bare ROW_NUMBER() OVER () clause."""
        result = sample_relation.with_row_number("rn")
        assert "rn" in result.columns
        assert len(result) == 5
        rns = self.column_values(adapter, result, "rn")
        # DuckDB does not guarantee assignment order with bare OVER (), so test set equality
        assert sorted(rns) == [1, 2, 3, 4, 5]

    def test_with_row_number_partition_by_only(self, adapter: FrameworkAdapter, sample_relation: Any) -> None:
        """partition_by without order_by: each row's rn is within partition size; sum is deterministic."""
        result = sample_relation.with_row_number("rn", partition_by=("category",))
        assert "rn" in result.columns
        assert len(result) == 5
        rns = self.column_values(adapter, result, "rn")
        # Categories: A,A,B,B,C -> partition sizes 2,2,1 -> rns are permutations of [1,2],[1,2],[1]
        # Sum is 1+2+1+2+1 = 7
        assert sum(rns) == 7
        # Each rn must be between 1 and its partition size (max partition size is 2)
        for rn in rns:
            assert 1 <= rn <= 2

    def test_with_row_number_order_by_only(self, adapter: FrameworkAdapter, sample_relation: Any) -> None:
        """order_by without partition_by: rows ordered by age get rn 1..5."""
        result = sample_relation.with_row_number("rn", order_by=("age",))
        assert "rn" in result.columns
        # Order the result by age and verify rn == [1,2,3,4,5]
        ordered = result.order("age")
        rns = self.column_values(adapter, ordered, "rn")
        assert rns == [1, 2, 3, 4, 5]

    def test_with_row_number_partition_and_order(self, adapter: FrameworkAdapter, sample_relation: Any) -> None:
        """partition_by=('category',), order_by=('age',). Verify expected rn per row by id."""
        result = sample_relation.with_row_number("rn", partition_by=("category",), order_by=("age",))
        ordered = result.order("id")
        ids = self.column_values(adapter, ordered, "id")
        rns = self.column_values(adapter, ordered, "rn")
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
            sample_relation.with_row_number("rn", ("category",))

    def test_with_row_number_order_by_keyword_only(self, sample_relation: Any) -> None:
        """order_by must be keyword-only to match the sibling window() API."""
        with pytest.raises(TypeError):
            sample_relation.with_row_number("rn", (), ("age",))

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

    def test_with_row_number_order_by_orderby_descending(self, adapter: FrameworkAdapter, sample_relation: Any) -> None:
        """with_row_number must honor OrderBy(descending=True) and produce the descending row numbers."""
        result = sample_relation.with_row_number("rn", order_by=(OrderBy("id", descending=True),))
        ordered = result.order("id")
        assert self.column_values(adapter, ordered, "rn") == [5, 4, 3, 2, 1]

    # --- window() ---

    @pytest.mark.parametrize(
        "apply",
        [
            pytest.param(lambda r: r.window("SUM(v)", "w", partition_by=("g",)), id="window-partition"),
            pytest.param(
                lambda r: r.window("SUM(v)", "w", partition_by=("g",), order_by=("v",)), id="window-partition-order"
            ),
            pytest.param(
                lambda r: r.window("SUM(v)", "w", order_by=(OrderBy("v", descending=True),)), id="window-order-desc"
            ),
            pytest.param(lambda r: r.with_row_number("w", partition_by=("g",)), id="row-number-partition"),
            pytest.param(
                lambda r: r.with_row_number("w", order_by=(OrderBy("v", descending=True),)), id="row-number-order-desc"
            ),
        ],
    )
    def test_window_keeps_input_order(
        self, adapter: FrameworkAdapter, connection: Any, relation_class: Any, apply: Any
    ) -> None:
        """A non-empty OVER clause must not reorder rows or leak helper columns."""
        rel = relation_class.from_dict(
            connection,
            {"id": list(range(12)), "g": ["a", "b", "c"] * 4, "v": [10 * i for i in range(12)]},
        )
        result = apply(rel)
        assert self.column_values(adapter, result, "id") == list(range(12))
        assert result.columns == ["id", "g", "v", "w"]

    def test_window_partition_only(self, adapter: FrameworkAdapter, sample_relation: Any) -> None:
        """SUM(age) partitioned by category, no order, no frame: each row gets its category total."""
        result = sample_relation.window("SUM(age)", "cat_sum", partition_by=("category",))
        assert "cat_sum" in result.columns
        ordered = result.order("id")
        sums = self.column_values(adapter, ordered, "cat_sum")
        # Categories per id (1..5): A,B,A,C,B; sums: A=25+35=60, B=30+45=75, C=40
        assert sums == [60, 75, 60, 40, 75]

    def test_window_partition_and_order_running_sum(self, adapter: FrameworkAdapter, sample_relation: Any) -> None:
        """SUM(age) partitioned by category, ordered by id, default RANGE UNBOUNDED PRECEDING..CURRENT ROW."""
        result = sample_relation.window("SUM(age)", "rs", partition_by=("category",), order_by=("id",))
        ordered = result.order("id")
        rs = self.column_values(adapter, ordered, "rs")
        assert rs == [25, 30, 60, 40, 75]

    def test_window_no_partition_no_order(self, adapter: FrameworkAdapter, sample_relation: Any) -> None:
        """COUNT(*) with no partition / order / frame returns total row count for every row."""
        result = sample_relation.window("COUNT(*)", "n")
        ns = self.column_values(adapter, result, "n")
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

    def test_window_rows_frame_unbounded_to_current(self, adapter: FrameworkAdapter, sample_relation: Any) -> None:
        """ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW: cumulative sum ordered by id."""
        result = sample_relation.window(
            "SUM(age)",
            "cum",
            order_by=("id",),
            frame=WindowFrame(kind="rows", start=Unbounded(), end=CurrentRow()),
        )
        ordered = result.order("id")
        cum = self.column_values(adapter, ordered, "cum")
        assert cum == [25, 55, 90, 130, 175]

    def test_window_range_frame_unbounded_to_unbounded(self, adapter: FrameworkAdapter, sample_relation: Any) -> None:
        """RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING: grand total per row."""
        result = sample_relation.window(
            "SUM(age)",
            "total",
            order_by=("id",),
            frame=WindowFrame(kind="range", start=Unbounded(), end=Unbounded()),
        )
        ordered = result.order("id")
        totals = self.column_values(adapter, ordered, "total")
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

    def test_window_rows_frame_preceding_and_following(self, adapter: FrameworkAdapter, sample_relation: Any) -> None:
        """ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING ordered by id: sliding 3-row window sum."""
        result = sample_relation.window(
            "SUM(age)",
            "win",
            order_by=("id",),
            frame=WindowFrame(kind="rows", start=Preceding(1), end=Following(1)),
        )
        ordered = result.order("id")
        win = self.column_values(adapter, ordered, "win")
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

    def test_window_order_by_orderby_descending(self, adapter: FrameworkAdapter, sample_relation: Any) -> None:
        """ROW_NUMBER assigned descending by id must give id=5->rn=1, id=4->rn=2, ..., id=1->rn=5."""
        result = sample_relation.window("ROW_NUMBER()", "rn", order_by=(OrderBy("id", descending=True),))
        ordered = result.order("id")
        assert self.column_values(adapter, ordered, "rn") == [5, 4, 3, 2, 1]

    def test_window_order_by_mixed_string_and_orderby(self, adapter: FrameworkAdapter, sample_relation: Any) -> None:
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
        assert self.column_values(adapter, ordered, "rn") == [2, 2, 1, 1, 1]

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

    # --- Exact-case column references ---

    @pytest.mark.parametrize("bad", ["missing", "CATEGORY"])
    def test_with_row_number_inexact_partition_by_raises(self, sample_relation: Any, bad: str) -> None:
        with pytest.raises(ValueError, match=bad):
            sample_relation.with_row_number("rn", partition_by=(bad,))

    @pytest.mark.parametrize("bad", ["missing", "CATEGORY"])
    def test_window_inexact_partition_by_raises(self, sample_relation: Any, bad: str) -> None:
        with pytest.raises(ValueError, match=bad):
            sample_relation.window("COUNT(*)", "n", partition_by=(bad,))

    @pytest.mark.parametrize("bad", ["AGE", OrderBy("AGE")], ids=["str", "orderby"])
    def test_with_row_number_case_mismatched_order_by_raises(self, sample_relation: Any, bad: Any) -> None:
        with pytest.raises(ValueError, match="AGE"):
            sample_relation.with_row_number("rn", order_by=(bad,))

    @pytest.mark.parametrize("bad", ["AGE", OrderBy("AGE")], ids=["str", "orderby"])
    def test_window_case_mismatched_order_by_raises(self, sample_relation: Any, bad: Any) -> None:
        with pytest.raises(ValueError, match="AGE"):
            sample_relation.window("COUNT(*)", "n", order_by=(bad,))

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

    def test_window_range_sentinel_only_with_multi_column_order_by_succeeds(
        self, adapter: FrameworkAdapter, sample_relation: Any
    ) -> None:
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
        totals = self.column_values(adapter, ordered, "total")
        assert totals == [175, 175, 175, 175, 175]
