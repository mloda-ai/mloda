import uuid
from typing import Any

from mloda.core.abstract_plugins.components.index.index import Index
from mloda.core.abstract_plugins.components.link import AsOfJoinConfig
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_base_merge_engine import SqlBaseMergeEngine
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore[assignment]


class DuckDBMergeEngine(SqlBaseMergeEngine):
    def check_import(self) -> None:
        if duckdb is None:
            raise ImportError("DuckDB is not installed. To be able to use this framework, please install duckdb.")

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
            raise ValueError("DuckDBMergeEngine asof does not support direction='nearest'.")

        left_by = left_index.index if left_index.is_multi_index() else (left_index.index[0],)
        right_by = right_index.index if right_index.is_multi_index() else (right_index.index[0],)

        if asof_config.direction == "backward":
            op = ">=" if asof_config.allow_exact_matches else ">"
        else:
            op = "<=" if asof_config.allow_exact_matches else "<"

        left_name = f"_left_{uuid.uuid4().hex}"
        right_name = f"_right_{uuid.uuid4().hex}"
        self._register_table(left_name, left_data)
        self._register_table(right_name, right_data)

        conds = [
            f"left_rel.{quote_ident(left)} = right_rel.{quote_ident(right)}" for left, right in zip(left_by, right_by)
        ]
        lt = quote_ident(asof_config.left_time_column)
        rt = quote_ident(asof_config.right_time_column)
        conds.append(f"left_rel.{lt} {op} right_rel.{rt}")
        on_clause = " AND ".join(conds)

        left_cols = self.get_column_names(left_data)
        right_cols = self.get_column_names(right_data)

        # tolerance: NULL out the right columns (LEFT semantics keep the row) when the
        # time gap exceeds tolerance. Expressed in the projection so the asof inequality
        # remains the last ON condition (a DuckDB ASOF JOIN requirement).
        tol_guard = ""
        if asof_config.tolerance is not None:
            tol = float(asof_config.tolerance)
            if asof_config.direction == "backward":
                gap = f"(left_rel.{lt} - right_rel.{rt})"
            else:
                gap = f"(right_rel.{rt} - left_rel.{lt})"
            tol_guard = f"{gap} <= {tol}"

        def project(col: str, rel: str) -> str:
            ref = f"{rel}.{quote_ident(col)}"
            if rel == "right_rel" and tol_guard:
                return f"CASE WHEN {tol_guard} THEN {ref} ELSE NULL END AS {quote_ident(col)}"
            return f"{ref} AS {quote_ident(col)}"

        proj = [project(c, "left_rel") for c in left_cols]
        proj += [project(c, "right_rel") for c in right_cols if c not in left_cols]
        projection = ", ".join(proj)

        sql = (
            f"SELECT {projection} FROM {quote_ident(left_name)} AS left_rel "  # nosec
            f"ASOF LEFT JOIN {quote_ident(right_name)} AS right_rel ON {on_clause}"
        )
        return self._execute_sql(sql)

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
