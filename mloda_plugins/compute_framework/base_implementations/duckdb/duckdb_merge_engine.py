from typing import Any

from mloda_core.abstract_plugins.components.index.index import Index
from mloda_core.abstract_plugins.components.link import JoinType
from mloda_core.abstract_plugins.components.merge.base_merge_engine import BaseMergeEngine

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore


class DuckDBMergeEngine(BaseMergeEngine):
    def check_import(self) -> None:
        if duckdb is None:
            raise ImportError("DuckDB is not installed. To be able to use this framework, please install duckdb.")

    def merge_inner(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.join_logic("inner", left_data, right_data, left_index, right_index, JoinType.INNER)

    def merge_left(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.join_logic("left", left_data, right_data, left_index, right_index, JoinType.LEFT)

    def merge_right(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.join_logic("right", left_data, right_data, left_index, right_index, JoinType.RIGHT)

    def merge_full_outer(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.join_logic("outer", left_data, right_data, left_index, right_index, JoinType.OUTER)

    def merge_union(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self._merge_relations(left_data, right_data, union_all=False)

    def merge_append(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self._merge_relations(left_data, right_data, union_all=True)

    def _merge_relations(self, left_data: Any, right_data: Any, union_all: bool) -> Any:
        """
        Internal helper to merge two DuckDB relations with aligned schemas.
        If union_all=True → UNION ALL (append), else UNION (removes duplicates).
        """

        def quote_ident(col: str) -> str:
            escaped_col = col.replace('"', '""')  # escape double quotes
            return f'"{escaped_col}"'

        def build_projection(cols_present: Any, all_cols: list[str]) -> str:
            return ", ".join(
                [
                    f"IFNULL({quote_ident(col)}, NULL) AS {quote_ident(col)}"
                    if col in cols_present
                    else f"NULL AS {quote_ident(col)}"
                    for col in all_cols
                ]
            )

        # Extract column sets
        left_cols = set(left_data.columns)
        right_cols = set(right_data.columns)
        all_cols = sorted(left_cols.union(right_cols))  # consistent order

        # Build projections
        left_proj = build_projection(left_cols, all_cols)
        right_proj = build_projection(right_cols, all_cols)

        # UNION vs UNION ALL
        union_keyword = "UNION ALL" if union_all else "UNION"

        # Run query (lazy)
        if self.framework_connection is None:
            raise ValueError("Framework connection is not set. Please set the framework connection before merging.")
        con = self.framework_connection
        con.register("left_temp", left_data)
        con.register("right_temp", right_data)

        sql = f" SELECT {left_proj} FROM left_temp {union_keyword} SELECT {right_proj} FROM right_temp "  # nosec

        return con.sql(sql)

    def get_column_names(self, data: Any) -> list[str]:
        """Get column names from data. Override in subclasses for different data types."""
        if hasattr(data, "columns"):
            return list(data.columns)
        # For DuckDB relations, get columns from the relation
        if hasattr(data, "columns"):
            return list(data.columns)
        raise ValueError("Data does not have column names or is not a DuckDB relation.")

    def is_empty_data(self, data: Any) -> Any:
        """Check if data is empty. Override in subclasses for different data types."""
        if hasattr(data, "__len__"):
            return len(data) == 0
        # For DuckDB relations, check if count is 0
        try:
            return data.count("*").fetchone()[0] == 0
        except Exception:
            return False

    def column_exists_in_result(self, result: Any, column_name: str) -> bool:
        """Check if column exists in result. Override in subclasses for different data types."""
        if hasattr(result, "columns"):
            return column_name in result.columns
        return False

    def handle_empty_data(self, left_data: Any, right_data: Any, left_idx: str, right_idx: str) -> Any:
        """Handle empty data cases. Override in subclasses for different data types."""
        if self.is_empty_data(left_data) or self.is_empty_data(right_data):
            if self.is_empty_data(left_data) and self.is_empty_data(right_data):
                # Both empty - return empty DataFrame
                return left_data.limit(0)
            elif self.is_empty_data(left_data):
                # Left empty - return empty DataFrame with left schema
                return left_data.limit(0)
            else:
                # Right empty - return empty DataFrame with right schema
                return right_data.limit(0)
        return None

    def join_logic(
        self, join_type: str, left_data: Any, right_data: Any, left_index: Index, right_index: Index, jointype: JoinType
    ) -> Any:
        # Check if framework connection is set
        if self.framework_connection is None:
            raise ValueError(
                "Framework connection not set. DuckDB merge engine requires a connection from the framework."
            )

        if left_index.is_multi_index() or right_index.is_multi_index():
            raise ValueError(f"MultiIndex is not yet implemented {self.__class__.__name__}")

        left_idx = left_index.index[0]
        right_idx = right_index.index[0]

        # Handle empty data cases
        empty_result = self.handle_empty_data(left_data, right_data, left_idx, right_idx)
        if empty_result is not None:
            return empty_result

        # Set unique aliases to avoid DuckDB alias conflicts
        left_aliased = left_data.set_alias("left_rel")
        right_aliased = right_data.set_alias("right_rel")

        # Use DuckDB relation's native join method
        # DuckDB relations support join operations directly
        if left_idx == right_idx:
            # Same column names - join on the column
            join_relation = left_aliased.join(right_aliased, left_idx, how=join_type)
        else:
            # Different column names - need to specify the condition
            # For DuckDB, we might need to use a more explicit approach
            join_relation = left_aliased.join(right_aliased, f"{left_idx}={right_idx}", how=join_type)

        # Return as lazy DuckDB relation
        return join_relation
