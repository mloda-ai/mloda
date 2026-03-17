import sqlite3
import threading
from typing import Any, List, Optional, Tuple

import pyarrow as pa

from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident


_counter_lock = threading.Lock()
_counter = 0


def _next_table_name() -> str:
    global _counter
    with _counter_lock:
        _counter += 1
        return f"_tmp_{_counter}"


def _arrow_type_to_sqlite(arrow_type: pa.DataType) -> str:
    if pa.types.is_integer(arrow_type):
        return "INTEGER"
    if pa.types.is_floating(arrow_type):
        return "REAL"
    if pa.types.is_boolean(arrow_type):
        return "INTEGER"
    return "TEXT"


def _infer_sqlite_type_from_values(values: list[Any]) -> str:
    result = "INTEGER"  # default when all non-None are bool/int
    found_non_none = False
    for v in values:
        if v is None:
            continue
        found_non_none = True
        if isinstance(v, float):
            return "REAL"  # widest numeric type -- return immediately
        if not isinstance(v, (bool, int)):
            result = "TEXT"
    if not found_non_none:
        return "TEXT"
    return result


def _sqlite_affinity_to_arrow_type(affinity: str) -> pa.DataType:
    upper = affinity.upper()
    if "INT" in upper:
        return pa.int64()
    if "REAL" in upper or "FLOA" in upper or "DOUB" in upper:
        return pa.float64()
    if "BLOB" in upper:
        return pa.large_binary()
    return pa.string()


class SqliteRelation:
    """Lazy relation wrapper around a sqlite3 connection and table name.

    Mirrors the subset of the DuckDB DuckDBPyRelation API that mloda uses:
    .columns, .filter(), .select(), .df(), .to_arrow_table(), len(), .set_alias(), .limit()

    Each mutating operation (filter, select) creates a new temporary table
    so that downstream code can compose operations without side effects.
    Temporary tables are cleaned up when the connection closes.
    """

    def __init__(self, connection: sqlite3.Connection, table_name: str) -> None:
        self._connection = connection
        self._table_name = table_name
        self._alias: Optional[str] = None

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection

    @property
    def table_name(self) -> str:
        return self._table_name

    @property
    def columns(self) -> List[str]:
        cursor = self._connection.execute(f'PRAGMA table_info("{self._table_name}")')  # nosec
        return [row[1] for row in cursor.fetchall()]

    def filter(self, condition: str, params: Tuple[Any, ...] = ()) -> "SqliteRelation":
        new_table = _next_table_name()
        sql = (
            f'CREATE TEMP TABLE "{new_table}" AS '  # nosec
            f'SELECT * FROM "{self._table_name}" WHERE {condition}'  # nosec
        )
        self._connection.execute(sql, params)
        return SqliteRelation(self._connection, new_table)

    def select(self, *columns: str, raw_sql: Optional[str] = None) -> "SqliteRelation":
        new_table = _next_table_name()
        if raw_sql is not None:
            projection = raw_sql
        else:
            projection = ", ".join(quote_ident(c) for c in columns)
        sql = (
            f'CREATE TEMP TABLE "{new_table}" AS '  # nosec
            f'SELECT {projection} FROM "{self._table_name}"'  # nosec
        )
        self._connection.execute(sql)
        return SqliteRelation(self._connection, new_table)

    def set_alias(self, alias: str) -> "SqliteRelation":
        rel = SqliteRelation(self._connection, self._table_name)
        rel._alias = alias
        return rel

    def get_alias(self) -> Optional[str]:
        return self._alias

    def limit(self, n: int) -> "SqliteRelation":
        new_table = _next_table_name()
        sql = (
            f'CREATE TEMP TABLE "{new_table}" AS '  # nosec
            f'SELECT * FROM "{self._table_name}" LIMIT {int(n)}'  # nosec
        )
        self._connection.execute(sql)
        return SqliteRelation(self._connection, new_table)

    def drop(self) -> None:
        """Drop the underlying temp table. Call this when the relation is no longer needed."""
        self._connection.execute(f'DROP TABLE IF EXISTS "{self._table_name}"')  # nosec

    def join(self, other: "SqliteRelation", condition: str, how: str = "inner") -> "SqliteRelation":
        new_table = _next_table_name()

        self_alias = self._alias or "left_rel"
        other_alias = other._alias or "right_rel"

        # If condition is a bare column name (no "="), expand to qualified equality.
        # The condition may already be quoted by the merge engine (e.g. "col_name").
        if "=" not in condition:
            condition = f"{self_alias}.{condition} = {other_alias}.{condition}"

        self_cols = self.columns
        other_cols = other.columns
        shared = set(self_cols) & set(other_cols)

        if how == "outer":
            return self._full_outer_join(other, condition, new_table, self_alias, other_alias, shared)

        join_map = {
            "inner": "INNER JOIN",
            "left": "LEFT JOIN",
            "right": "LEFT JOIN",
        }
        join_clause = join_map.get(how)
        if join_clause is None:
            raise ValueError(f"Unsupported join type: {how}")

        if how == "right":
            sql_left_table = other._table_name
            sql_left_alias = other_alias
            sql_left_cols = other_cols
            sql_right_table = self._table_name
            sql_right_alias = self_alias
            sql_right_cols = self_cols
        else:
            sql_left_table = self._table_name
            sql_left_alias = self_alias
            sql_left_cols = self_cols
            sql_right_table = other._table_name
            sql_right_alias = other_alias
            sql_right_cols = other_cols

        projection = self._build_join_projection(sql_left_alias, sql_left_cols, sql_right_alias, sql_right_cols, shared)

        sql = (
            f'CREATE TEMP TABLE "{new_table}" AS '  # nosec
            f'SELECT {projection} FROM "{sql_left_table}" AS {sql_left_alias} '  # nosec
            f'{join_clause} "{sql_right_table}" AS {sql_right_alias} '  # nosec
            f"ON {condition}"  # nosec
        )
        self._connection.execute(sql)
        return SqliteRelation(self._connection, new_table)

    def _full_outer_join(
        self,
        other: "SqliteRelation",
        condition: str,
        new_table: str,
        self_alias: str,
        other_alias: str,
        shared: set[str],
    ) -> "SqliteRelation":
        """Emulate FULL OUTER JOIN: LEFT JOIN UNION ALL reversed LEFT JOIN (excluding matches)."""
        self_cols = self.columns
        other_cols = other.columns

        # Left part: self LEFT JOIN other
        left_proj = self._build_join_projection(self_alias, self_cols, other_alias, other_cols, shared)
        left_sql = (
            f'SELECT {left_proj} FROM "{self._table_name}" AS {self_alias} '  # nosec
            f'LEFT JOIN "{other._table_name}" AS {other_alias} '  # nosec
            f"ON {condition}"  # nosec
        )

        # Right part: other LEFT JOIN self WHERE self keys are NULL (unmatched right rows)
        right_proj = self._build_join_projection(
            self_alias, self_cols, other_alias, other_cols, shared, prefer_right_shared=True
        )
        # Use rowid (always non-NULL for matched rows) to identify unmatched rows.
        # Using a data column is unsafe: nullable columns would falsely match rows with NULL values.
        where_clause = f"WHERE {self_alias}.rowid IS NULL"

        right_sql = (
            f'SELECT {right_proj} FROM "{other._table_name}" AS {other_alias} '  # nosec
            f'LEFT JOIN "{self._table_name}" AS {self_alias} '  # nosec
            f"ON {condition} "  # nosec
            f"{where_clause}"  # nosec
        )

        sql = f'CREATE TEMP TABLE "{new_table}" AS {left_sql} UNION ALL {right_sql}'  # nosec
        self._connection.execute(sql)
        return SqliteRelation(self._connection, new_table)

    @staticmethod
    def _build_join_projection(
        left_alias: str,
        left_cols: list[str],
        right_alias: str,
        right_cols: list[str],
        shared: set[str],
        prefer_right_shared: bool = False,
    ) -> str:
        select_parts: list[str] = []
        for c in left_cols:
            qc = quote_ident(c)
            if c in shared and prefer_right_shared:
                select_parts.append(f"COALESCE({left_alias}.{qc}, {right_alias}.{qc}) AS {qc}")
            else:
                select_parts.append(f"{left_alias}.{qc} AS {qc}")
        for c in right_cols:
            if c not in shared:
                qc = quote_ident(c)
                select_parts.append(f"{right_alias}.{qc} AS {qc}")
        return ", ".join(select_parts)

    def to_arrow_table(self) -> pa.Table:
        cursor = self._connection.execute(f'SELECT * FROM "{self._table_name}"')  # nosec
        cols = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        if not rows:
            pragma_cursor = self._connection.execute(f'PRAGMA table_info("{self._table_name}")')  # nosec
            type_map = {row[1]: row[2] for row in pragma_cursor.fetchall()}
            arrays = [pa.array([], type=_sqlite_affinity_to_arrow_type(type_map.get(c, "TEXT"))) for c in cols]
            return pa.table(dict(zip(cols, arrays)))

        col_data: dict[str, list[Any]] = {c: [] for c in cols}
        for row in rows:
            for c, val in zip(cols, row):
                col_data[c].append(val)

        return pa.table(col_data)

    def df(self) -> Any:
        return self.to_arrow_table().to_pandas()

    def __len__(self) -> int:
        cursor = self._connection.execute(f'SELECT COUNT(*) FROM "{self._table_name}"')  # nosec
        result: int = cursor.fetchone()[0]
        return result

    @classmethod
    def from_arrow(cls, connection: sqlite3.Connection, arrow_table: pa.Table) -> "SqliteRelation":
        table_name = _next_table_name()
        cols = arrow_table.column_names

        col_defs = ", ".join(
            f"{quote_ident(c)} {_arrow_type_to_sqlite(arrow_table.schema.field(c).type)}" for c in cols
        )
        connection.execute(f'CREATE TEMP TABLE "{table_name}" ({col_defs})')  # nosec

        if arrow_table.num_rows > 0:
            placeholders = ", ".join("?" for _ in cols)
            insert_sql = f'INSERT INTO "{table_name}" VALUES ({placeholders})'  # nosec

            rows = []
            for i in range(arrow_table.num_rows):
                row = tuple(arrow_table.column(c)[i].as_py() for c in range(arrow_table.num_columns))
                rows.append(row)
            connection.executemany(insert_sql, rows)

        return cls(connection, table_name)

    @classmethod
    def from_dict(cls, connection: sqlite3.Connection, data: dict[str, list[Any]]) -> "SqliteRelation":
        table_name = _next_table_name()
        cols = list(data.keys())

        if not cols:
            raise ValueError("Cannot create relation from empty dictionary")

        col_defs = ", ".join(f"{quote_ident(c)} {_infer_sqlite_type_from_values(data[c])}" for c in cols)
        connection.execute(f'CREATE TEMP TABLE "{table_name}" ({col_defs})')  # nosec

        num_rows = len(data[cols[0]])
        if num_rows > 0:
            placeholders = ", ".join("?" for _ in cols)
            insert_sql = f'INSERT INTO "{table_name}" VALUES ({placeholders})'  # nosec

            rows = []
            for i in range(num_rows):
                row = tuple(data[c][i] for c in cols)
                rows.append(row)
            connection.executemany(insert_sql, rows)

        return cls(connection, table_name)
