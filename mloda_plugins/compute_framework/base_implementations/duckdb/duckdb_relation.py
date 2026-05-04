from __future__ import annotations

import re
from typing import Any, Literal, Optional, cast

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore[assignment]

try:
    import pyarrow as pa
except ImportError:
    pa = None  # type: ignore[assignment]

from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import inline_params, quote_ident


class DuckdbRelation:
    """Lazy wrapper around DuckDBPyRelation.

    All operations delegate to DuckDB's native lazy Relational API;
    no temp views or tables are created.

    SQL injection prevention:
    - Identifiers go through ``quote_ident`` (SQL-standard double-quote escaping).
    - DuckDB's Relational API does not support PEP 249 parameterized queries,
      so ``filter()`` falls back to ``inline_params`` / ``quote_value`` for
      literal values. See ``sql_utils`` module docstring for details.
    - ``_raw_sql`` on ``select()`` bypasses quoting; callers must not pass
      user-controlled input.
    """

    def __init__(self, connection: duckdb.DuckDBPyConnection, relation: duckdb.DuckDBPyRelation) -> None:
        self._connection = connection
        self._relation = relation
        self._alias: Optional[str] = None

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        return self._connection

    @property
    def columns(self) -> list[str]:
        return self._relation.columns

    def filter(self, condition: str, params: tuple[Any, ...] = ()) -> "DuckdbRelation":
        """Apply a filter condition. Params are inlined via ``quote_value``
        because DuckDB's Relational API lacks PEP 249 parameter binding."""
        if params:
            condition = inline_params(condition, params)
        new_rel = self._relation.filter(condition)
        return DuckdbRelation(self._connection, new_rel)

    def select(self, *columns: str, _raw_sql: Optional[str] = None) -> "DuckdbRelation":
        """Project columns. _raw_sql bypasses quoting: never pass user-controlled input."""
        if _raw_sql is not None:
            new_rel = self._relation.project(_raw_sql)
        else:
            projection = ", ".join(quote_ident(c) for c in columns)
            new_rel = self._relation.project(projection)
        return DuckdbRelation(self._connection, new_rel)

    def set_alias(self, alias: str) -> "DuckdbRelation":
        new_rel = DuckdbRelation(self._connection, self._relation.set_alias(alias))
        new_rel._alias = alias
        return new_rel

    def get_alias(self) -> Optional[str]:
        return self._alias

    _VALID_JOIN_TYPES = {"inner", "left", "right", "outer"}
    _JoinType = Literal["inner", "left", "right", "outer"]

    def join(self, other: "DuckdbRelation", condition: str, how: str = "inner") -> "DuckdbRelation":
        self_alias = self._alias or "left_rel"
        other_alias = other._alias or "right_rel"

        if how not in self._VALID_JOIN_TYPES:
            raise ValueError(f"Unsupported join type: {how}")

        if "=" not in condition:
            col = condition.strip('"')
            condition = f"{quote_ident(self_alias)}.{quote_ident(col)} = {quote_ident(other_alias)}.{quote_ident(col)}"
        else:
            condition = re.sub(r"(?<!\w)" + re.escape(self_alias) + r"(?=\.)", quote_ident(self_alias), condition)
            condition = re.sub(r"(?<!\w)" + re.escape(other_alias) + r"(?=\.)", quote_ident(other_alias), condition)

        self_cols = self.columns
        other_cols = other.columns
        shared = set(self_cols) & set(other_cols)

        joined = self._relation.join(other._relation, condition, how=cast(DuckdbRelation._JoinType, how))

        projection = self._build_join_projection(self_alias, self_cols, other_alias, other_cols, shared, how)
        result = joined.project(projection)

        return DuckdbRelation(self._connection, result)

    @staticmethod
    def _build_join_projection(
        left_alias: str,
        left_cols: list[str],
        right_alias: str,
        right_cols: list[str],
        shared: set[str],
        how: str = "inner",
    ) -> str:
        ql = quote_ident(left_alias)
        qr = quote_ident(right_alias)
        select_parts: list[str] = []
        for c in left_cols:
            qc = quote_ident(c)
            if c in shared and how == "right":
                select_parts.append(f"{qr}.{qc} AS {qc}")
            elif c in shared and how == "outer":
                select_parts.append(f"COALESCE({ql}.{qc}, {qr}.{qc}) AS {qc}")
            else:
                select_parts.append(f"{ql}.{qc} AS {qc}")
        for c in right_cols:
            if c not in shared:
                qc = quote_ident(c)
                select_parts.append(f"{qr}.{qc} AS {qc}")
        return ", ".join(select_parts)

    def limit(self, n: int) -> "DuckdbRelation":
        new_rel = self._relation.limit(n)
        return DuckdbRelation(self._connection, new_rel)

    def order(self, *columns: str) -> "DuckdbRelation":
        """Return a new relation sorted by the given columns."""
        return DuckdbRelation(self._connection, self._relation.order(", ".join(columns)))

    def drop(self) -> None:
        """No-op: native relations have no persistent state to clean up."""
        pass

    def to_arrow_table(self) -> pa.Table:
        return self._relation.arrow().read_all()

    def df(self) -> Any:
        return self._relation.df()

    def __len__(self) -> int:
        row = self._relation.aggregate("count_star()").fetchone()
        if row is None:
            raise RuntimeError("count_star() returned no rows")
        result: int = row[0]
        return result

    def append_column(self, name: str, values: list[Any]) -> "DuckdbRelation":
        """Return a new relation with an additional column appended positionally."""
        new_col_rel = DuckdbRelation.from_dict(self._connection, {name: values})
        rn = "__mloda_rn__"
        qrn = quote_ident(rn)
        left = self._relation.project(f"*, ROW_NUMBER() OVER () AS {qrn}")
        right = new_col_rel._relation.project(f"*, ROW_NUMBER() OVER () AS {qrn}")
        left = left.set_alias("__l__")
        right = right.set_alias("__r__")
        joined = left.join(right, f'"__l__".{qrn} = "__r__".{qrn}')
        keep = ", ".join(f'"__l__".{quote_ident(c)}' for c in self.columns)
        keep += f', "__r__".{quote_ident(name)}'
        result = joined.project(keep)
        return DuckdbRelation(self._connection, result)

    @classmethod
    def from_arrow(cls, connection: duckdb.DuckDBPyConnection, arrow_table: pa.Table) -> "DuckdbRelation":
        relation = connection.from_arrow(arrow_table)
        return cls(connection, relation)

    @classmethod
    def from_dict(cls, connection: duckdb.DuckDBPyConnection, data: dict[str, list[Any]]) -> "DuckdbRelation":
        if not data:
            raise ValueError("Cannot create relation from empty dictionary")
        arrow_table = pa.table(data)
        return cls.from_arrow(connection, arrow_table)
