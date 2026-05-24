import datetime
import re
import sqlite3
import uuid
from collections.abc import Sequence
from typing import Any, Optional

import pyarrow as pa

from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import pick_helper_column_name, quote_ident


# Python 3.12 deprecated the built-in sqlite3 datetime adapters; explicit ISO-format
# registration silences the per-row DeprecationWarning emitted by executemany().
# Side effect: sqlite3.register_adapter is module-global, so every sqlite3.Connection
# in the process (including ones not created via SqliteRelation) picks these up.
# Adapter-only, no converter: mloda reads sqlite data back through Arrow as text and
# never sets detect_types, so the read path is unaffected. The T-separator output
# matches the existing literal-SQL pipeline in sql_utils.quote_value (value.isoformat()).
sqlite3.register_adapter(datetime.date, datetime.date.isoformat)
sqlite3.register_adapter(datetime.datetime, datetime.datetime.isoformat)


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
    non_null_values = [value for value in values if value is not None]
    if not non_null_values:
        return "TEXT"
    if all(isinstance(value, (bytes, bytearray)) for value in non_null_values):
        return "BLOB"

    result = "INTEGER"  # default when all non-None are bool/int
    for v in non_null_values:
        if not isinstance(v, (bool, int, float)):
            return "TEXT"  # TEXT dominates; safe to return early
        if isinstance(v, float) and result == "INTEGER":
            result = "REAL"  # REAL upgrades INTEGER but not TEXT
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


def _infer_arrow_type_from_values(values: list[Any]) -> pa.DataType:
    non_null_values = [value for value in values if value is not None]
    if not non_null_values:
        return pa.string()
    if all(isinstance(value, bool) for value in non_null_values):
        return pa.bool_()
    if all(isinstance(value, datetime.datetime) for value in non_null_values):
        return pa.timestamp("us")
    if all(isinstance(value, datetime.date) and not isinstance(value, datetime.datetime) for value in non_null_values):
        return pa.date32()
    if all(isinstance(value, (bytes, bytearray)) for value in non_null_values):
        return pa.large_binary()
    if all(isinstance(value, (bool, int, float)) for value in non_null_values):
        if any(isinstance(value, float) for value in non_null_values):
            return pa.float64()
        return pa.int64()
    return pa.string()


def _raw_sql_projection_starts_with_star(projection: str) -> bool:
    return re.match(r"^\*\s*(?:,|$)", projection.lstrip()) is not None


def _values_for_arrow_type(values: list[Any], arrow_type: pa.DataType) -> list[Any]:
    if pa.types.is_boolean(arrow_type):
        return [None if value is None else bool(value) for value in values]
    if pa.types.is_timestamp(arrow_type):
        return [datetime.datetime.fromisoformat(value) if isinstance(value, str) else value for value in values]
    if pa.types.is_date(arrow_type):
        return [datetime.date.fromisoformat(value) if isinstance(value, str) else value for value in values]
    return values


def _compatible_arrow_type(left_type: pa.DataType | None, right_type: pa.DataType | None) -> pa.DataType | None:
    if left_type is None or right_type is None:
        return None
    if left_type == right_type:
        return left_type
    if pa.types.is_integer(left_type) and pa.types.is_integer(right_type):
        if pa.types.is_signed_integer(left_type) or pa.types.is_signed_integer(right_type):
            width = max(left_type.bit_width, right_type.bit_width)
            if width <= 8:
                return pa.int8()
            if width <= 16:
                return pa.int16()
            if width <= 32:
                return pa.int32()
            return pa.int64()

        width = max(left_type.bit_width, right_type.bit_width)
        if width <= 8:
            return pa.uint8()
        if width <= 16:
            return pa.uint16()
        if width <= 32:
            return pa.uint32()
        return pa.uint64()
    if pa.types.is_floating(left_type) and pa.types.is_floating(right_type):
        return pa.float64() if max(left_type.bit_width, right_type.bit_width) > 32 else pa.float32()
    if (
        pa.types.is_integer(left_type)
        and pa.types.is_floating(right_type)
        or pa.types.is_floating(left_type)
        and pa.types.is_integer(right_type)
    ):
        return pa.float64()
    return left_type


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

    def __init__(
        self,
        connection: sqlite3.Connection,
        table_name: str,
        _is_view: bool = False,
        _types: Optional[Sequence[pa.DataType | None]] = None,
    ) -> None:
        self._connection = connection
        self._table_name = table_name
        self._alias: Optional[str] = None
        self._is_view = _is_view
        self._cached_columns: Optional[list[str]] = None
        self._cached_types: Optional[list[pa.DataType | None]] = list(_types) if _types is not None else None

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

    @property
    def types(self) -> list[pa.DataType]:
        columns = self.columns
        type_hints = self._types_for_current_columns()
        if type_hints is not None:
            return type_hints
        type_map = self._sqlite_affinity_types_by_column()
        return [type_map.get(column, pa.string()) for column in columns]

    def _sqlite_affinity_types_by_column(self) -> dict[str, pa.DataType]:
        cursor = self._connection.execute(f"PRAGMA table_info({quote_ident(self._table_name)})")
        return {row[1]: _sqlite_affinity_to_arrow_type(str(row[2])) for row in cursor.fetchall()}

    def _type_hints_for_current_columns(self) -> Optional[list[pa.DataType | None]]:
        if self._cached_types is None or len(self._cached_types) != len(self.columns):
            return None
        return list(self._cached_types)

    def _types_for_current_columns(self) -> Optional[list[pa.DataType]]:
        type_hints = self._type_hints_for_current_columns()
        if type_hints is None:
            return None
        if any(arrow_type is None for arrow_type in type_hints):
            return self._types_with_inferred_unknown_hints(type_hints)
        return [arrow_type for arrow_type in type_hints if arrow_type is not None]

    def _types_with_inferred_unknown_hints(self, type_hints: list[pa.DataType | None]) -> list[pa.DataType]:
        unknown_indexes = [idx for idx, arrow_type in enumerate(type_hints) if arrow_type is None]
        values_by_index: dict[int, list[Any]] = {idx: [] for idx in unknown_indexes}
        cursor = self._connection.execute(f"SELECT * FROM {quote_ident(self._table_name)}")  # nosec
        for row in cursor.fetchall():
            for idx in unknown_indexes:
                values_by_index[idx].append(row[idx])

        resolved_hints = list(type_hints)
        for idx in unknown_indexes:
            resolved_hints[idx] = _infer_arrow_type_from_values(values_by_index[idx])
        self._cached_types = resolved_hints
        return [arrow_type for arrow_type in resolved_hints if arrow_type is not None]

    def _types_for_projection(self, columns: list[str]) -> Optional[list[pa.DataType | None]]:
        current_type_hints = self._types_for_current_columns()
        if current_type_hints is None:
            return None
        type_map = dict(zip(self.columns, current_type_hints, strict=True))
        if any(column not in type_map for column in columns):
            return None
        return [type_map[column] for column in columns]

    def _types_for_raw_sql_projection(
        self, projection: str, output_columns: list[str]
    ) -> Optional[list[pa.DataType | None]]:
        if not _raw_sql_projection_starts_with_star(projection):
            return None
        source_columns = self.columns
        source_type_hints = self._types_for_current_columns()
        if source_type_hints is None or len(output_columns) < len(source_columns):
            return None
        if output_columns[: len(source_columns)] != source_columns:
            return None
        return source_type_hints + [None] * (len(output_columns) - len(source_columns))

    @staticmethod
    def _types_for_join_projection(
        left_cols: list[str],
        left_types: Optional[Sequence[pa.DataType | None]],
        right_cols: list[str],
        right_types: Optional[Sequence[pa.DataType | None]],
        shared: set[str],
        coalesce_shared: bool = False,
    ) -> Optional[list[pa.DataType | None]]:
        if left_types is None or right_types is None:
            return None
        left_type_map = dict(zip(left_cols, left_types, strict=True))
        right_type_map = dict(zip(right_cols, right_types, strict=True))
        output_types = [
            _compatible_arrow_type(left_type_map[column], right_type_map[column])
            if coalesce_shared and column in shared
            else left_type_map[column]
            for column in left_cols
        ]
        output_types.extend(right_type_map[column] for column in right_cols if column not in shared)
        return output_types

    def filter(self, condition: str, params: tuple[Any, ...] = ()) -> "SqliteRelation":
        """Apply a filter condition using PEP 249 parameterized queries."""
        new_name = _next_table_name()
        types = self._types_for_current_columns()
        sql = (
            f"CREATE TEMP TABLE {quote_ident(new_name)} AS "  # nosec
            f"SELECT * FROM {quote_ident(self._table_name)} WHERE {condition}"
        )
        self._connection.execute(sql, params)
        return SqliteRelation(
            self._connection,
            new_name,
            _is_view=False,
            _types=types,
        )

    def select(self, *columns: str, _raw_sql: Optional[str] = None) -> "SqliteRelation":
        """Project columns. _raw_sql bypasses quoting: never pass user-controlled input."""
        new_name = _next_table_name()
        if _raw_sql is not None:
            projection = _raw_sql
            types: Optional[list[pa.DataType | None]] = None
        else:
            projection = ", ".join(quote_ident(c) for c in columns)
            types = self._types_for_projection(list(columns))
        sql = f"CREATE TEMP VIEW {quote_ident(new_name)} AS SELECT {projection} FROM {quote_ident(self._table_name)}"  # nosec
        self._connection.execute(sql)
        if _raw_sql is not None:
            cursor = self._connection.execute(f"SELECT * FROM {quote_ident(new_name)} LIMIT 0")  # nosec
            output_columns = [desc[0] for desc in cursor.description]
            types = self._types_for_raw_sql_projection(projection, output_columns)
        return SqliteRelation(self._connection, new_name, _is_view=True, _types=types)

    def set_alias(self, alias: str) -> "SqliteRelation":
        rel = SqliteRelation(self._connection, self._table_name, self._is_view, self._types_for_current_columns())
        rel._alias = alias
        return rel

    def get_alias(self) -> Optional[str]:
        return self._alias

    def limit(self, n: int) -> "SqliteRelation":
        new_name = _next_table_name()
        types = self._types_for_current_columns()
        sql = (
            f"CREATE TEMP VIEW {quote_ident(new_name)} AS SELECT * FROM {quote_ident(self._table_name)} LIMIT {int(n)}"  # nosec
        )
        self._connection.execute(sql)
        return SqliteRelation(
            self._connection,
            new_name,
            _is_view=True,
            _types=types,
        )

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
            sql_left_types = other._types_for_current_columns()
            sql_right_table = self._table_name
            sql_right_alias = self_alias
            sql_right_cols = self_cols
            sql_right_types = self._types_for_current_columns()
        else:
            sql_left_table = self._table_name
            sql_left_alias = self_alias
            sql_left_cols = self_cols
            sql_left_types = self._types_for_current_columns()
            sql_right_table = other._table_name
            sql_right_alias = other_alias
            sql_right_cols = other_cols
            sql_right_types = other._types_for_current_columns()

        projection = self._build_join_projection(sql_left_alias, sql_left_cols, sql_right_alias, sql_right_cols, shared)
        types = self._types_for_join_projection(sql_left_cols, sql_left_types, sql_right_cols, sql_right_types, shared)

        sql = (
            f"CREATE TEMP VIEW {quote_ident(new_table)} AS "  # nosec
            f"SELECT {projection} FROM {quote_ident(sql_left_table)} AS {quote_ident(sql_left_alias)} "
            f"{join_clause} {quote_ident(sql_right_table)} AS {quote_ident(sql_right_alias)} "
            f"ON {condition}"
        )
        self._connection.execute(sql)
        return SqliteRelation(self._connection, new_table, _is_view=True, _types=types)

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

        types = self._types_for_join_projection(
            self_cols,
            self._types_for_current_columns(),
            other_cols,
            other._types_for_current_columns(),
            shared,
            coalesce_shared=True,
        )
        sql = f"CREATE TEMP VIEW {quote_ident(new_table)} AS {left_sql} UNION ALL {right_sql}"
        self._connection.execute(sql)
        return SqliteRelation(self._connection, new_table, _is_view=True, _types=types)

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
        if name.casefold() in {c.casefold() for c in self.columns}:
            raise ValueError(f"Column {name!r} already exists in the relation")
        current_types = self._types_for_current_columns()
        new_col_rel = SqliteRelation.from_dict(self._connection, {name: values})
        new_col_types = new_col_rel._types_for_current_columns()
        rn = pick_helper_column_name(taken=set(self.columns) | {name})
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
        types = current_types + new_col_types if current_types is not None and new_col_types is not None else None
        return SqliteRelation(self._connection, result_name, _is_view=True, _types=types)

    def to_arrow_table(self) -> pa.Table:
        cursor = self._connection.execute(f"SELECT * FROM {quote_ident(self._table_name)}")  # nosec
        cols = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        if not rows:
            arrays = [pa.array([], type=arrow_type) for arrow_type in self.types]
            return pa.table(dict(zip(cols, arrays)))

        col_data: dict[str, list[Any]] = {c: [] for c in cols}
        for row in rows:
            for c, val in zip(cols, row):
                col_data[c].append(val)

        type_hints = self._types_for_current_columns()
        if type_hints is None:
            return pa.table(col_data)

        arrays = []
        for column, arrow_type in zip(cols, type_hints, strict=True):
            arrays.append(pa.array(_values_for_arrow_type(col_data[column], arrow_type), type=arrow_type))
        return pa.table(dict(zip(cols, arrays, strict=True)))

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

        return cls(connection, table_name, _types=list(arrow_table.schema.types))

    @classmethod
    def from_dict(cls, connection: sqlite3.Connection, data: dict[str, list[Any]]) -> "SqliteRelation":
        table_name = _next_table_name()
        cols = list(data.keys())

        if not cols:
            raise ValueError("Cannot create relation from empty dictionary")

        sqlite_types = {column: _infer_sqlite_type_from_values(data[column]) for column in cols}
        col_defs = ", ".join(f"{quote_ident(c)} {sqlite_types[c]}" for c in cols)
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

        types = [_infer_arrow_type_from_values(data[column]) for column in cols]
        return cls(connection, table_name, _types=types)
