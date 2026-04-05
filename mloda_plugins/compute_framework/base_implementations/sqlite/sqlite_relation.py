import re
import sqlite3
import uuid
from typing import Any, Optional

import pyarrow as pa

from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident


def _next_table_name() -> str:
    return f"_tmp_{uuid.uuid4().hex}"


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
        if not isinstance(v, (bool, int, float)):
            return "TEXT"  # TEXT dominates; safe to return early
        if isinstance(v, float) and result == "INTEGER":
            result = "REAL"  # REAL upgrades INTEGER but not TEXT
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

    Each mutating operation (filter, select) creates a new temp view/table
    so that downstream code can compose operations without side effects.
    Temporary objects are cleaned up when the connection closes.

    SQL injection prevention:
    - Identifiers go through ``quote_ident`` (SQL-standard double-quote escaping).
    - Literal values use PEP 249 parameterized queries via ``execute(sql, params)``.
    - ``_raw_sql`` on ``select()`` bypasses quoting; callers must not pass
      user-controlled input.
    """

    def __init__(self, connection: sqlite3.Connection, table_name: str, _is_view: bool = False) -> None:
        self._connection = connection
        self._table_name = table_name
        self._alias: Optional[str] = None
        self._is_view = _is_view
        self._cached_columns: Optional[list[str]] = None

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection

    @property
    def table_name(self) -> str:
        return self._table_name

    @property
    def columns(self) -> list[str]:
        if self._cached_columns is None:
            cursor = self._connection.execute(f"SELECT * FROM {quote_ident(self._table_name)} LIMIT 0")  # nosec
            self._cached_columns = [desc[0] for desc in cursor.description]
        return self._cached_columns

    def filter(self, condition: str, params: tuple[Any, ...] = ()) -> "SqliteRelation":
        """Apply a filter condition using PEP 249 parameterized queries."""
        new_name = _next_table_name()
        sql = (
            f"CREATE TEMP TABLE {quote_ident(new_name)} AS "  # nosec
            f"SELECT * FROM {quote_ident(self._table_name)} WHERE {condition}"
        )
        self._connection.execute(sql, params)
        return SqliteRelation(self._connection, new_name, _is_view=False)

    def select(self, *columns: str, _raw_sql: Optional[str] = None) -> "SqliteRelation":
        """Project columns. _raw_sql bypasses quoting: never pass user-controlled input."""
        new_name = _next_table_name()
        if _raw_sql is not None:
            projection = _raw_sql
        else:
            projection = ", ".join(quote_ident(c) for c in columns)
        sql = f"CREATE TEMP VIEW {quote_ident(new_name)} AS SELECT {projection} FROM {quote_ident(self._table_name)}"  # nosec
        self._connection.execute(sql)
        return SqliteRelation(self._connection, new_name, _is_view=True)

    def set_alias(self, alias: str) -> "SqliteRelation":
        rel = SqliteRelation(self._connection, self._table_name, self._is_view)
        rel._alias = alias
        return rel

    def get_alias(self) -> Optional[str]:
        return self._alias

    def limit(self, n: int) -> "SqliteRelation":
        new_name = _next_table_name()
        sql = (
            f"CREATE TEMP VIEW {quote_ident(new_name)} AS SELECT * FROM {quote_ident(self._table_name)} LIMIT {int(n)}"  # nosec
        )
        self._connection.execute(sql)
        return SqliteRelation(self._connection, new_name, _is_view=True)

    def drop(self) -> None:
        """Drop the underlying temp table or view."""
        if self._is_view:
            self._connection.execute(f"DROP VIEW IF EXISTS {quote_ident(self._table_name)}")
        else:
            self._connection.execute(f"DROP TABLE IF EXISTS {quote_ident(self._table_name)}")

    def join(self, other: "SqliteRelation", condition: str, how: str = "inner") -> "SqliteRelation":
        new_table = _next_table_name()

        self_alias = self._alias or "left_rel"
        other_alias = other._alias or "right_rel"

        # If condition is a bare column name (no "="), expand to qualified equality.
        # The condition may already be quoted by the merge engine (e.g. "col_name").
        if "=" not in condition:
            col = condition.strip('"')
            condition = f"{quote_ident(self_alias)}.{quote_ident(col)} = {quote_ident(other_alias)}.{quote_ident(col)}"
        else:
            condition = re.sub(r"(?<!\w)" + re.escape(self_alias) + r"(?=\.)", quote_ident(self_alias), condition)
            condition = re.sub(r"(?<!\w)" + re.escape(other_alias) + r"(?=\.)", quote_ident(other_alias), condition)

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
            f"CREATE TEMP VIEW {quote_ident(new_table)} AS "  # nosec
            f"SELECT {projection} FROM {quote_ident(sql_left_table)} AS {quote_ident(sql_left_alias)} "
            f"{join_clause} {quote_ident(sql_right_table)} AS {quote_ident(sql_right_alias)} "
            f"ON {condition}"
        )
        self._connection.execute(sql)
        return SqliteRelation(self._connection, new_table, _is_view=True)

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

        q_self_alias = quote_ident(self_alias)
        q_other_alias = quote_ident(other_alias)

        # Left part: self LEFT JOIN other
        left_proj = self._build_join_projection(self_alias, self_cols, other_alias, other_cols, shared)
        left_sql = (
            f"SELECT {left_proj} FROM {quote_ident(self._table_name)} AS {q_self_alias} "  # nosec
            f"LEFT JOIN {quote_ident(other._table_name)} AS {q_other_alias} "
            f"ON {condition}"
        )

        # Right part: other LEFT JOIN self WHERE self keys are NULL (unmatched right rows)
        right_proj = self._build_join_projection(
            self_alias, self_cols, other_alias, other_cols, shared, prefer_right_shared=True
        )
        first_shared_col = next(iter(shared))
        where_clause = f"WHERE {q_self_alias}.{quote_ident(first_shared_col)} IS NULL"

        right_sql = (
            f"SELECT {right_proj} FROM {quote_ident(other._table_name)} AS {q_other_alias} "  # nosec
            f"LEFT JOIN {quote_ident(self._table_name)} AS {q_self_alias} "
            f"ON {condition} "
            f"{where_clause}"
        )

        sql = f"CREATE TEMP VIEW {quote_ident(new_table)} AS {left_sql} UNION ALL {right_sql}"
        self._connection.execute(sql)
        return SqliteRelation(self._connection, new_table, _is_view=True)

    @staticmethod
    def _build_join_projection(
        left_alias: str,
        left_cols: list[str],
        right_alias: str,
        right_cols: list[str],
        shared: set[str],
        prefer_right_shared: bool = False,
    ) -> str:
        ql = quote_ident(left_alias)
        qr = quote_ident(right_alias)
        select_parts: list[str] = []
        for c in left_cols:
            qc = quote_ident(c)
            if c in shared and prefer_right_shared:
                select_parts.append(f"COALESCE({ql}.{qc}, {qr}.{qc}) AS {qc}")
            else:
                select_parts.append(f"{ql}.{qc} AS {qc}")
        for c in right_cols:
            if c not in shared:
                qc = quote_ident(c)
                select_parts.append(f"{qr}.{qc} AS {qc}")
        return ", ".join(select_parts)

    def append_column(self, name: str, values: list[Any]) -> "SqliteRelation":
        """Return a new relation with an additional column appended positionally."""
        new_col_rel = SqliteRelation.from_dict(self._connection, {name: values})
        rn = "__mloda_rn__"
        qrn = quote_ident(rn)
        left_name = _next_table_name()
        right_name = _next_table_name()
        result_name = _next_table_name()

        left_cols = ", ".join(quote_ident(c) for c in self.columns)
        self._connection.execute(
            f"CREATE TEMP VIEW {quote_ident(left_name)} AS "  # nosec
            f"SELECT {left_cols}, ROW_NUMBER() OVER () AS {qrn} "
            f"FROM {quote_ident(self._table_name)}"
        )
        self._connection.execute(
            f"CREATE TEMP VIEW {quote_ident(right_name)} AS "  # nosec
            f"SELECT {quote_ident(name)}, ROW_NUMBER() OVER () AS {qrn} "
            f"FROM {quote_ident(new_col_rel._table_name)}"
        )

        keep = ", ".join(f"{quote_ident(left_name)}.{quote_ident(c)}" for c in self.columns)
        keep += f", {quote_ident(right_name)}.{quote_ident(name)}"
        self._connection.execute(
            f"CREATE TEMP VIEW {quote_ident(result_name)} AS "  # nosec
            f"SELECT {keep} FROM {quote_ident(left_name)} "
            f"INNER JOIN {quote_ident(right_name)} "
            f"ON {quote_ident(left_name)}.{qrn} = {quote_ident(right_name)}.{qrn}"
        )
        return SqliteRelation(self._connection, result_name, _is_view=True)

    def to_arrow_table(self) -> pa.Table:
        cursor = self._connection.execute(f"SELECT * FROM {quote_ident(self._table_name)}")  # nosec
        cols = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        if not rows:
            pragma_cursor = self._connection.execute(f"PRAGMA table_info({quote_ident(self._table_name)})")
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
        cursor = self._connection.execute(f"SELECT COUNT(*) FROM {quote_ident(self._table_name)}")  # nosec
        result: int = cursor.fetchone()[0]
        return result

    @classmethod
    def from_arrow(cls, connection: sqlite3.Connection, arrow_table: pa.Table) -> "SqliteRelation":
        table_name = _next_table_name()
        cols = arrow_table.column_names

        col_defs = ", ".join(
            f"{quote_ident(c)} {_arrow_type_to_sqlite(arrow_table.schema.field(c).type)}" for c in cols
        )
        connection.execute(f"CREATE TEMP TABLE {quote_ident(table_name)} ({col_defs})")

        if arrow_table.num_rows > 0:
            placeholders = ", ".join("?" for _ in cols)
            insert_sql = f"INSERT INTO {quote_ident(table_name)} VALUES ({placeholders})"  # nosec
            rows = list(zip(*(col.to_pylist() for col in arrow_table.columns)))
            connection.executemany(insert_sql, rows)

        return cls(connection, table_name)

    @classmethod
    def from_dict(cls, connection: sqlite3.Connection, data: dict[str, list[Any]]) -> "SqliteRelation":
        table_name = _next_table_name()
        cols = list(data.keys())

        if not cols:
            raise ValueError("Cannot create relation from empty dictionary")

        col_defs = ", ".join(f"{quote_ident(c)} {_infer_sqlite_type_from_values(data[c])}" for c in cols)
        connection.execute(f"CREATE TEMP TABLE {quote_ident(table_name)} ({col_defs})")

        num_rows = len(data[cols[0]])
        if not all(len(data[c]) == num_rows for c in cols):
            raise ValueError("All columns must have the same length.")
        if num_rows > 0:
            placeholders = ", ".join("?" for _ in cols)
            insert_sql = f"INSERT INTO {quote_ident(table_name)} VALUES ({placeholders})"  # nosec
            rows = []
            for i in range(num_rows):
                row = tuple(data[c][i] for c in cols)
                rows.append(row)
            connection.executemany(insert_sql, rows)

        return cls(connection, table_name)
