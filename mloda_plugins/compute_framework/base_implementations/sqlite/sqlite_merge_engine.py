import sqlite3
from typing import Any

from mloda_plugins.compute_framework.base_implementations.sql.sql_base_merge_engine import SqlBaseMergeEngine
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation, _next_table_name


class SqliteMergeEngine(SqlBaseMergeEngine):
    def _execute_sql(self, sql: str) -> Any:
        if self.framework_connection is None:
            raise ValueError("Framework connection is not set.")
        conn: sqlite3.Connection = self.framework_connection
        new_table = _next_table_name()
        conn.execute(f'CREATE TEMP TABLE "{new_table}" AS {sql}')  # nosec
        return SqliteRelation(conn, new_table)

    def _register_table(self, name: str, data: Any) -> None:
        if self.framework_connection is None:
            raise ValueError("Framework connection is not set.")
        conn: sqlite3.Connection = self.framework_connection
        rel: SqliteRelation = data
        conn.execute(f'DROP VIEW IF EXISTS "{name}"')  # nosec
        conn.execute(f'CREATE TEMP VIEW "{name}" AS SELECT * FROM "{rel.table_name}"')  # nosec

    def _set_alias(self, data: Any, alias: str) -> Any:
        return data.set_alias(alias)

    def _join_relation(self, left: Any, right: Any, condition: str, how: str) -> Any:
        return left.join(right, condition, how=how)
