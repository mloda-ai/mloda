# Annotations stay strings so the pa.DataType signature below does not dereference pa at import time.
from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import replace
from datetime import timedelta
from typing import Any

from mloda.core.abstract_plugins.components.contract.comparison_contract import ColumnSemantics
from mloda.core.abstract_plugins.components.index.index import Index
from mloda.core.abstract_plugins.components.link import AsOfJoinConfig
from mloda_plugins.compute_framework.base_implementations.sql import sql_type_semantics
from mloda_plugins.compute_framework.base_implementations.sql.sql_base_merge_engine import SqlBaseMergeEngine
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import is_ordered_arrow_type, quote_ident
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation, _next_table_name
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_value_sample import sample_string_values

try:
    import pyarrow as pa
except ImportError:
    pa = None  # type: ignore[assignment]


class SqliteMergeEngine(SqlBaseMergeEngine):
    provides_column_semantics = True

    def _merge_relations(self, left_data: Any, right_data: Any, union_all: bool) -> Any:
        if self.framework_connection is None:
            raise ValueError("Framework connection is not set. Please set the framework connection before merging.")
        left_proj, right_proj = self._union_projections(left_data, right_data)
        union_keyword = "UNION ALL" if union_all else "UNION"
        left_ref = quote_ident(left_data.table_name)
        right_ref = quote_ident(right_data.table_name)
        sql = f" SELECT {left_proj} FROM {left_ref} {union_keyword} SELECT {right_proj} FROM {right_ref} "  # nosec
        return self._execute_sql(sql)

    def merge_asof(
        self,
        left_data: Any,
        right_data: Any,
        left_index: Index,
        right_index: Index,
        asof_config: AsOfJoinConfig,
    ) -> Any:
        left_data, right_data = self.validate_asof_time_columns(left_data, right_data, asof_config)
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

        left_ref = quote_ident(left_data.table_name)
        right_ref = quote_ident(right_data.table_name)

        lt = quote_ident(asof_config.left_time_column)
        rt = quote_ident(asof_config.right_time_column)

        on_conds = [f"L.{quote_ident(left)} = R.{quote_ident(right)}" for left, right in zip(left_by, right_by)]
        on_conds.append(f"R.{rt} {op} L.{lt}")
        if asof_config.tolerance is not None:
            if isinstance(asof_config.tolerance, timedelta):
                raise ValueError(
                    f"{self.__class__.__name__} ASOF does not support a timedelta tolerance; provide a numeric "
                    "tolerance (e.g. epoch seconds matching the time column)."
                )
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
            f"FROM (SELECT *, ROW_NUMBER() OVER () AS {lid} FROM {left_ref}) AS L "
            f"LEFT JOIN {right_ref} AS R ON {on_clause}"
            f") WHERE {rn} = 1"
        )
        return self._execute_sql(sql)

    def validate_asof_time_columns(
        self, left_data: Any, right_data: Any, asof_config: AsOfJoinConfig
    ) -> tuple[Any, Any]:
        # One-sided coercion is unsafe: julianday day numbers cannot be assumed compatible
        # with an already-numeric column of unknown unit (e.g. epoch seconds).
        if asof_config.coerce_time_columns:
            left_ordered = self._asof_time_column_is_ordered(left_data, asof_config.left_time_column)
            right_ordered = self._asof_time_column_is_ordered(right_data, asof_config.right_time_column)
            if left_ordered != right_ordered:
                column = asof_config.right_time_column if left_ordered else asof_config.left_time_column
                raise ValueError(
                    f"Cannot coerce as-of time column '{column}': sqlite coercion yields julianday day "
                    f"numbers, which cannot be assumed compatible with the other side's existing numeric "
                    f"column; cast the column manually before joining."
                )
        return super().validate_asof_time_columns(left_data, right_data, asof_config)

    def _schema_maybe_temporal(self, data: Any, column: str) -> bool:
        # Temporal only if a real temporal arrow type or a string (sqlite stores datetimes as TEXT).
        arrow_type = data.types[data.columns.index(column)]
        return bool(pa.types.is_temporal(arrow_type)) or sql_type_semantics.is_string_like_arrow_type(arrow_type)

    def _column_semantics(self, data: Any, column: str) -> ColumnSemantics:
        arrow_type = data.types[data.columns.index(column)]
        # Cost note: value-sampling runs one LIMIT 100 query per string-typed column during
        # equi-join validation, including genuine string keys where the result is discarded;
        # a per-relation sample cache is a possible future optimization.
        is_string_like = sql_type_semantics.is_string_like_arrow_type(arrow_type)
        value_sample = sample_string_values(data, column) if is_string_like else None
        sem = sql_type_semantics.column_semantics_from_arrow(arrow_type, value_sample=value_sample)
        # Value-inspection may recognize an ISO-TEXT column as temporal, but the as-of gate
        # keys off physical order: TEXT storage still needs coercion, so is_ordered is pinned
        # to the arrow-type check (mirrors PythonDictMergeEngine).
        return replace(sem, is_ordered=self._asof_time_column_is_ordered(data, column))

    def _asof_time_column_is_ordered(self, data: Any, column: str) -> bool:
        idx = data.columns.index(column)
        return is_ordered_arrow_type(data.types[idx])

    def _coerce_asof_time_column(self, data: Any, column: str) -> Any:
        """Coerce an ISO-8601 TEXT time column to julianday NUMERIC (days since epoch origin).

        julianday silently yields NULL for non-ISO values, so an eager check fails hard first.
        """
        qc = quote_ident(column)
        table_ref = quote_ident(data.table_name)
        check_sql = f"SELECT COUNT(*) FROM {table_ref} WHERE {qc} IS NOT NULL AND julianday({qc}) IS NULL"  # nosec
        bad_count: int = data.connection.execute(check_sql).fetchone()[0]
        if bad_count > 0:
            raise ValueError(
                f"As-of time column '{column}' contains {bad_count} non-ISO-8601 value(s) that "
                f"cannot be coerced via julianday."
            )
        projection = ", ".join(
            f"julianday({quote_ident(c)}) AS {quote_ident(c)}" if c == column else quote_ident(c) for c in data.columns
        )
        types = [pa.float64() if c == column else t for c, t in zip(data.columns, data.types)]
        return self._execute_sql(f"SELECT {projection} FROM {table_ref}", types=types)  # nosec

    def _execute_sql(self, sql: str, types: Sequence[pa.DataType | None] | None = None) -> Any:
        if self.framework_connection is None:
            raise ValueError("Framework connection is not set.")
        conn: sqlite3.Connection = self.framework_connection
        new_table = _next_table_name()
        conn.execute(f"CREATE TEMP VIEW {quote_ident(new_table)} AS {sql}")  # nosec
        return SqliteRelation(conn, new_table, _is_view=True, _types=types)

    def _set_alias(self, data: Any, alias: str) -> Any:
        return data.set_alias(alias)

    def _join_relation(self, left: Any, right: Any, condition: str, how: str) -> Any:
        return left.join(right, condition, how=how)
