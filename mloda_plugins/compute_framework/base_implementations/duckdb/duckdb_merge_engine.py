from typing import Any

from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_base_merge_engine import SqlBaseMergeEngine

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore[assignment]


class DuckDBMergeEngine(SqlBaseMergeEngine):
    def check_import(self) -> None:
        if duckdb is None:
            raise ImportError("DuckDB is not installed. To be able to use this framework, please install duckdb.")

    def _execute_sql(self, sql: str) -> Any:
        if self.framework_connection is None:
            raise ValueError("Framework connection is not set.")
        result = self.framework_connection.sql(sql)
        return DuckdbRelation(self.framework_connection, result)

    def _register_table(self, name: str, data: Any) -> None:
        if self.framework_connection is None:
            raise ValueError("Framework connection is not set.")
        if isinstance(data, DuckdbRelation):
            data._relation.create_view(name, replace=True)
        else:
            self.framework_connection.register(name, data)

    def _set_alias(self, data: Any, alias: str) -> Any:
        return data.set_alias(alias)

    def _join_relation(self, left: Any, right: Any, condition: str, how: str) -> Any:
        return left.join(right, condition, how=how)
