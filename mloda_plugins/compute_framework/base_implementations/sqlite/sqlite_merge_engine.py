import sqlite3
import uuid
from typing import Any

from mloda.core.abstract_plugins.components.index.index import Index
from mloda.core.abstract_plugins.components.link import AsOfJoinConfig
from mloda_plugins.compute_framework.base_implementations.sql.sql_base_merge_engine import SqlBaseMergeEngine
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation, _next_table_name


class SqliteMergeEngine(SqlBaseMergeEngine):
    def merge_asof(
        self,
        left_data: Any,
        right_data: Any,
        left_index: Index,
        right_index: Index,
        asof_config: AsOfJoinConfig,
    ) -> Any:
        if self.framework_connection is None:
            raise ValueError("Framework connection not set. SQL merge engine requires a connection from the framework.")
        if asof_config.direction == "nearest":
            raise ValueError("SqliteMergeEngine asof does not support direction='nearest'.")

        left_by = left_index.index if left_index.is_multi_index() else (left_index.index[0],)
        right_by = right_index.index if right_index.is_multi_index() else (right_index.index[0],)

        if asof_config.direction == "backward":
            op = "<=" if asof_config.allow_exact_matches else "<"
            agg = "MAX"
        else:
            op = ">=" if asof_config.allow_exact_matches else ">"
            agg = "MIN"

        left_name = f"_left_{uuid.uuid4().hex}"
        right_name = f"_right_{uuid.uuid4().hex}"
        self._register_table(left_name, left_data)
        self._register_table(right_name, right_data)

        lt = quote_ident(asof_config.left_time_column)
        rt = quote_ident(asof_config.right_time_column)

        sub_conds = [f"R2.{quote_ident(right)} = L.{quote_ident(left)}" for left, right in zip(left_by, right_by)]
        sub_conds.append(f"R2.{rt} {op} L.{lt}")
        if asof_config.tolerance is not None:
            tol = float(asof_config.tolerance)
            sub_conds.append(f"ABS(L.{lt} - R2.{rt}) <= {tol}")
        subquery = (
            f"SELECT {agg}(R2.{rt}) FROM {quote_ident(right_name)} AS R2 WHERE " + " AND ".join(sub_conds)  # nosec
        )

        on_conds = [f"L.{quote_ident(left)} = R.{quote_ident(right)}" for left, right in zip(left_by, right_by)]
        on_conds.append(f"R.{rt} = ({subquery})")
        on_clause = " AND ".join(on_conds)

        left_cols = self.get_column_names(left_data)
        right_cols = self.get_column_names(right_data)
        proj = [f"L.{quote_ident(c)} AS {quote_ident(c)}" for c in left_cols]
        proj += [f"R.{quote_ident(c)} AS {quote_ident(c)}" for c in right_cols if c not in left_cols]
        projection = ", ".join(proj)

        sql = (
            f"SELECT {projection} FROM {quote_ident(left_name)} AS L "  # nosec
            f"LEFT JOIN {quote_ident(right_name)} AS R ON {on_clause}"
        )
        return self._execute_sql(sql)

    def _execute_sql(self, sql: str) -> Any:
        if self.framework_connection is None:
            raise ValueError("Framework connection is not set.")
        conn: sqlite3.Connection = self.framework_connection
        new_table = _next_table_name()
        conn.execute(f"CREATE TEMP VIEW {quote_ident(new_table)} AS {sql}")
        return SqliteRelation(conn, new_table, _is_view=True)

    def _register_table(self, name: str, data: Any) -> None:
        if self.framework_connection is None:
            raise ValueError("Framework connection is not set.")
        conn: sqlite3.Connection = self.framework_connection
        rel: SqliteRelation = data
        conn.execute(f"DROP VIEW IF EXISTS {quote_ident(name)}")
        conn.execute(f"CREATE TEMP VIEW {quote_ident(name)} AS SELECT * FROM {quote_ident(rel.table_name)}")  # nosec

    def _set_alias(self, data: Any, alias: str) -> Any:
        return data.set_alias(alias)

    def _join_relation(self, left: Any, right: Any, condition: str, how: str) -> Any:
        return left.join(right, condition, how=how)
