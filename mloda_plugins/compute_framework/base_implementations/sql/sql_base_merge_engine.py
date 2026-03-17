import uuid
from typing import Any

from mloda.user import Index
from mloda.provider import BaseMergeEngine

from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident


class SqlBaseMergeEngine(BaseMergeEngine):
    """Shared SQL merge logic for SQL-based merge engines (DuckDB, SQLite, etc.).

    Subclasses must implement:
    - _execute_sql(sql): Execute SQL and return a relation/result
    - _register_table(name, data): Register data as a named table for SQL queries
    - _set_alias(data, alias): Set an alias on a relation
    - _join_relation(left, right, condition, how): Execute a join between two relations
    """

    def _execute_sql(self, sql: str) -> Any:
        raise NotImplementedError

    def _register_table(self, name: str, data: Any) -> None:
        raise NotImplementedError

    def _set_alias(self, data: Any, alias: str) -> Any:
        raise NotImplementedError

    def _join_relation(self, left: Any, right: Any, condition: str, how: str) -> Any:
        raise NotImplementedError

    def merge_inner(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.join_logic("inner", left_data, right_data, left_index, right_index)

    def merge_left(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.join_logic("left", left_data, right_data, left_index, right_index)

    def merge_right(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.join_logic("right", left_data, right_data, left_index, right_index)

    def merge_full_outer(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.join_logic("outer", left_data, right_data, left_index, right_index)

    def merge_union(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self._merge_relations(left_data, right_data, union_all=False)

    def merge_append(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self._merge_relations(left_data, right_data, union_all=True)

    def _merge_relations(self, left_data: Any, right_data: Any, union_all: bool) -> Any:
        def build_projection(cols_present: Any, all_cols: list[str]) -> str:
            return ", ".join(
                [
                    f"{quote_ident(col)} AS {quote_ident(col)}"
                    if col in cols_present
                    else f"NULL AS {quote_ident(col)}"
                    for col in all_cols
                ]
            )

        left_cols = set(self.get_column_names(left_data))
        right_cols = set(self.get_column_names(right_data))
        all_cols = sorted(left_cols.union(right_cols))

        left_proj = build_projection(left_cols, all_cols)
        right_proj = build_projection(right_cols, all_cols)

        union_keyword = "UNION ALL" if union_all else "UNION"

        if self.framework_connection is None:
            raise ValueError("Framework connection is not set. Please set the framework connection before merging.")

        left_name = f"_left_{uuid.uuid4().hex}"
        right_name = f"_right_{uuid.uuid4().hex}"
        self._register_table(left_name, left_data)
        self._register_table(right_name, right_data)

        sql = f' SELECT {left_proj} FROM "{left_name}" {union_keyword} SELECT {right_proj} FROM "{right_name}" '  # nosec
        return self._execute_sql(sql)

    def get_column_names(self, data: Any) -> list[str]:
        if hasattr(data, "columns"):
            return list(data.columns)
        raise ValueError("Data does not have column names.")

    def is_empty_data(self, data: Any) -> bool:
        if hasattr(data, "__len__"):
            return len(data) == 0
        return False

    def column_exists_in_result(self, result: Any, column_name: str) -> bool:
        if hasattr(result, "columns"):
            return column_name in result.columns
        return False

    def handle_empty_data(
        self, left_data: Any, right_data: Any, left_idx: Any, right_idx: Any, join_type: str = "inner"
    ) -> Any:
        left_empty = self.is_empty_data(left_data)
        right_empty = self.is_empty_data(right_data)

        if join_type == "inner":
            if left_empty or right_empty:
                return left_data.limit(0)
        elif join_type == "left":
            if left_empty:
                return left_data.limit(0)
            # right_empty: left join still returns all left rows — don't short-circuit
        elif join_type == "right":
            if right_empty:
                return right_data.limit(0)
            # left_empty: right join still returns all right rows — don't short-circuit
        elif join_type == "outer":
            if left_empty and right_empty:
                return left_data.limit(0)
            # one side empty: let the join handle it (outer join preserves both sides)

        return None

    def join_logic(
        self,
        join_type: str,
        left_data: Any,
        right_data: Any,
        left_index: Index,
        right_index: Index,
    ) -> Any:
        if self.framework_connection is None:
            raise ValueError("Framework connection not set. SQL merge engine requires a connection from the framework.")

        left_idx = left_index.index if left_index.is_multi_index() else left_index.index[0]
        right_idx = right_index.index if right_index.is_multi_index() else right_index.index[0]

        empty_result = self.handle_empty_data(left_data, right_data, left_idx, right_idx, join_type)
        if empty_result is not None:
            return empty_result

        left_aliased = self._set_alias(left_data, "left_rel")
        right_aliased = self._set_alias(right_data, "right_rel")

        if left_index.is_multi_index() or right_index.is_multi_index():
            conditions = []
            for left_col, right_col in zip(left_idx, right_idx):
                conditions.append(f"left_rel.{quote_ident(left_col)}=right_rel.{quote_ident(right_col)}")
            join_condition = " AND ".join(conditions)
            return self._join_relation(left_aliased, right_aliased, join_condition, join_type)
        else:
            if not isinstance(left_idx, str):
                raise ValueError(f"Expected a single string index, got {type(left_idx)}: {left_idx!r}")
            if not isinstance(right_idx, str):
                raise ValueError(f"Expected a single string index, got {type(right_idx)}: {right_idx!r}")
            if left_idx == right_idx:
                return self._join_relation(left_aliased, right_aliased, quote_ident(left_idx), join_type)
            else:
                return self._join_relation(
                    left_aliased, right_aliased, f"{quote_ident(left_idx)}={quote_ident(right_idx)}", join_type
                )
