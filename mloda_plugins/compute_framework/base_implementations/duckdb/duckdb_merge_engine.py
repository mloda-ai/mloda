from datetime import timedelta
from typing import Any

from mloda.core.abstract_plugins.components.contract.comparison_contract import ColumnSemantics
from mloda.core.abstract_plugins.components.index.index import Index
from mloda.core.abstract_plugins.components.link import AsOfJoinConfig
from mloda_plugins.compute_framework.base_implementations.duckdb import duckdb_type_semantics
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_base_merge_engine import SqlBaseMergeEngine
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import (
    fold_identifier,
    pick_helper_column_name,
    quote_ident,
)

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore[assignment]


class DuckDBMergeEngine(SqlBaseMergeEngine):
    provides_column_semantics = True

    def check_import(self) -> None:
        if duckdb is None:
            raise ImportError("DuckDB is not installed. To be able to use this framework, please install duckdb.")

    def _merge_relations(self, left_data: Any, right_data: Any, union_all: bool) -> Any:
        if self.framework_connection is None:
            raise ValueError("Framework connection is not set. Please set the framework connection before merging.")
        left_proj, right_proj = self._union_projections(left_data, right_data)
        left_rel = left_data._relation.project(left_proj)
        right_rel = right_data._relation.project(right_proj)
        unioned = left_rel.union(right_rel)
        if not union_all:
            unioned = unioned.distinct()
        return DuckdbRelation(self.framework_connection, unioned)

    def merge_asof(
        self,
        left_data: Any,
        right_data: Any,
        left_index: Index,
        right_index: Index,
        asof_config: AsOfJoinConfig,
    ) -> Any:
        left_by = left_index.index if left_index.is_multi_index() else (left_index.index[0],)
        right_by = right_index.index if right_index.is_multi_index() else (right_index.index[0],)
        left_cols, right_cols, right_extra = self._check_asof_columns(
            left_data, right_data, left_by, right_by, asof_config
        )

        left_data, right_data = self.validate_asof_time_columns(left_data, right_data, asof_config)
        if self.framework_connection is None:
            raise ValueError("Framework connection not set. SQL merge engine requires a connection from the framework.")
        if asof_config.direction == "nearest":
            raise ValueError("DuckDBMergeEngine asof does not support direction='nearest'.")

        if asof_config.direction == "backward":
            op, time_order = (">=" if asof_config.allow_exact_matches else ">"), "DESC"
        else:
            op, time_order = ("<=" if asof_config.allow_exact_matches else "<"), "ASC"

        taken = {*left_cols, *right_cols}
        lid_name = pick_helper_column_name(taken)
        rn_name = pick_helper_column_name({*taken, lid_name})
        reserved = {fold_identifier(c) for c in (*left_cols, lid_name, rn_name)}
        n = 0
        while any(fold_identifier(f"_mloda_r{n}_{c}") in reserved for c in right_cols):
            n += 1
        lid, rn = quote_ident(lid_name), quote_ident(rn_name)

        lt, rt = asof_config.left_time_column, asof_config.right_time_column

        rmap = {c: f"_mloda_r{n}_{c}" for c in right_cols}
        left_rel = left_data._relation.project(f"*, ROW_NUMBER() OVER () AS {lid}").set_alias("L")
        right_proj = ", ".join(f"{quote_ident(c)} AS {quote_ident(rmap[c])}" for c in right_cols)
        right_rel = right_data._relation.project(right_proj).set_alias("R")

        conds = [f"L.{quote_ident(lb)} = R.{quote_ident(rmap[rb])}" for lb, rb in zip(left_by, right_by)]
        conds.append(f"L.{quote_ident(lt)} {op} R.{quote_ident(rmap[rt])}")
        if asof_config.tolerance is not None:
            if isinstance(asof_config.tolerance, timedelta):
                raise ValueError(
                    f"{self.__class__.__name__} ASOF does not support a timedelta tolerance; provide a numeric "
                    "tolerance (e.g. epoch seconds matching the time column)."
                )
            tol = float(asof_config.tolerance)
            conds.append(f"ABS(L.{quote_ident(lt)} - R.{quote_ident(rmap[rt])}) <= {tol}")
        on_clause = " AND ".join(conds)

        joined = left_rel.join(right_rel, on_clause, how="left")

        qrt = quote_ident(rmap[rt])
        order_keys = [f"({qrt} IS NULL)", f"{qrt} {time_order}"]
        order_keys += [f"{quote_ident(rmap[c])} ASC" for c in right_extra]
        ranked = joined.project(f"*, ROW_NUMBER() OVER (PARTITION BY {lid} ORDER BY {', '.join(order_keys)}) AS {rn}")
        picked = ranked.filter(f"{rn} = 1")

        final_proj = [f"{quote_ident(c)} AS {quote_ident(c)}" for c in left_cols]
        final_proj += [f"{quote_ident(rmap[c])} AS {quote_ident(c)}" for c in right_extra]
        result = picked.project(", ".join(final_proj))
        return DuckdbRelation(self.framework_connection, result)

    def _coerce_asof_time_column(self, data: Any, column: str) -> Any:
        qc = quote_ident(column)
        # CAST AS TIMESTAMP silently drops UTC offsets, corrupting as-of ordering, so fail eagerly.
        offset_pattern = "([+-][0-9]{2}:?[0-9]{2}|Z)$"
        offending = data.filter(f"{qc} IS NOT NULL AND regexp_matches(CAST({qc} AS VARCHAR), '{offset_pattern}')")
        offset_count = len(offending)
        if offset_count > 0:
            raise ValueError(
                f"As-of time column '{column}' contains {offset_count} value(s) with a UTC offset or "
                f"trailing 'Z'; duckdb's CAST AS TIMESTAMP silently drops the offset. Cast the column "
                f"manually (e.g. via TIMESTAMPTZ) before joining."
            )
        return data.project(f"* REPLACE (CAST({qc} AS TIMESTAMP) AS {qc})")

    def _column_semantics(self, data: Any, column: str) -> ColumnSemantics:
        return duckdb_type_semantics.column_semantics(data, column)

    def _asof_time_column_is_ordered(self, data: Any, column: str) -> bool:
        idx = data.columns.index(column)
        t = data.types[idx]
        dtype_id = getattr(t, "id", str(t)).lower()
        ordered = ("int", "decimal", "float", "double", "real", "numeric", "date", "time", "interval")
        return any(k in dtype_id for k in ordered)

    def _set_alias(self, data: Any, alias: str) -> Any:
        return data.set_alias(alias)

    def _join_relation(self, left: Any, right: Any, condition: str, how: str) -> Any:
        return left.join(right, condition, how=how)
