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
            time_order = "DESC"
        else:
            op = ">=" if asof_config.allow_exact_matches else ">"
            time_order = "ASC"

        left_name = f"_left_{uuid.uuid4().hex}"
        right_name = f"_right_{uuid.uuid4().hex}"
        self._register_table(left_name, left_data)
        self._register_table(right_name, right_data)

        lt = quote_ident(asof_config.left_time_column)
        rt = quote_ident(asof_config.right_time_column)

        on_conds = [f"L.{quote_ident(left)} = R.{quote_ident(right)}" for left, right in zip(left_by, right_by)]
        on_conds.append(f"R.{rt} {op} L.{lt}")
        if asof_config.tolerance is not None:
            tol = float(asof_config.tolerance)
            on_conds.append(f"ABS(L.{lt} - R.{rt}) <= {tol}")
        on_clause = " AND ".join(on_conds)

        left_cols = self.get_column_names(left_data)
        right_cols = self.get_column_names(right_data)
        right_extra = [c for c in right_cols if c not in left_cols]

        lid = quote_ident("_mloda_lid")
        rn = quote_ident("_mloda_rn")

        order_keys = [f"(R.{rt} IS NULL)", f"R.{rt} {time_order}"]
        order_keys += [f"R.{quote_ident(c)} ASC" for c in right_extra]
        order_clause = ", ".join(order_keys)

        inner_proj = ["L.*"]
        inner_proj += [f"R.{quote_ident(c)} AS {quote_ident(c)}" for c in right_extra]
        inner_projection = ", ".join(inner_proj)

        outer_cols = [quote_ident(c) for c in left_cols] + [quote_ident(c) for c in right_extra]
        outer_projection = ", ".join(outer_cols)

        sql = (
            f"SELECT {outer_projection} FROM ("  # nosec
            f"SELECT {inner_projection}, "
            f"ROW_NUMBER() OVER (PARTITION BY L.{lid} ORDER BY {order_clause}) AS {rn} "
            f"FROM (SELECT *, ROW_NUMBER() OVER () AS {lid} FROM {quote_ident(left_name)}) AS L "
            f"LEFT JOIN {quote_ident(right_name)} AS R ON {on_clause}"
            f") WHERE {rn} = 1"
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
