from typing import Any

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
        assert self.framework_connection is not None
        return self.framework_connection.sql(sql)

    def _register_table(self, name: str, data: Any) -> None:
        assert self.framework_connection is not None
        self.framework_connection.register(name, data)

    def _set_alias(self, data: Any, alias: str) -> Any:
        return data.set_alias(alias)

    def _join_relation(self, left: Any, right: Any, condition: str, how: str) -> Any:
        return left.join(right, condition, how=how)
