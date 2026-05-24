from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Optional, cast

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore[assignment]

try:
    import pyarrow as pa
except ImportError:
    pa = None  # type: ignore[assignment]

from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import (
    inline_params,
    pick_helper_column_name,
    quote_ident,
)


@dataclass(frozen=True)
class CurrentRow:
    """Frame bound: the current row."""


@dataclass(frozen=True)
class Unbounded:
    """Frame bound: unbounded (preceding on the start side, following on the end side)."""


@dataclass(frozen=True)
class Preceding:
    """Frame bound: ``offset`` rows/range/groups before the current row."""

    offset: int

    def __post_init__(self) -> None:
        if isinstance(self.offset, bool) or not isinstance(self.offset, int):
            raise TypeError(f"Preceding offset must be int; got {type(self.offset).__name__}")


@dataclass(frozen=True)
class Following:
    """Frame bound: ``offset`` rows/range/groups after the current row."""

    offset: int

    def __post_init__(self) -> None:
        if isinstance(self.offset, bool) or not isinstance(self.offset, int):
            raise TypeError(f"Following offset must be int; got {type(self.offset).__name__}")


FrameBound = CurrentRow | Unbounded | Preceding | Following


@dataclass(frozen=True)
class WindowFrame:
    """A window frame clause (``ROWS|RANGE|GROUPS BETWEEN <start> AND <end>``)."""

    kind: Literal["rows", "range", "groups"]
    start: FrameBound
    end: FrameBound

    def __post_init__(self) -> None:
        if self.kind not in ("rows", "range", "groups"):
            raise ValueError(f"WindowFrame kind must be one of 'rows', 'range', 'groups'; got {self.kind!r}")


def _render_frame_bound(bound: FrameBound, side: Literal["start", "end"]) -> str:
    """Render a single frame bound for the given ``side`` of a BETWEEN clause."""
    if isinstance(bound, Unbounded):
        return "UNBOUNDED PRECEDING" if side == "start" else "UNBOUNDED FOLLOWING"
    if isinstance(bound, CurrentRow):
        return "CURRENT ROW"
    if isinstance(bound, Preceding):
        return f"{bound.offset} PRECEDING"
    if isinstance(bound, Following):
        return f"{bound.offset} FOLLOWING"
    raise TypeError(f"Unsupported frame bound: {type(bound).__name__}")


def _render_over_clause(
    partition_by: Sequence[str],
    order_by: Sequence[str],
    frame: WindowFrame | None,
) -> str:
    """Render the body of an ``OVER (...)`` clause."""
    parts: list[str] = []
    if partition_by:
        parts.append("PARTITION BY " + ", ".join(quote_ident(c) for c in partition_by))
    if order_by:
        parts.append("ORDER BY " + ", ".join(quote_ident(c) for c in order_by))
    if frame is not None:
        start_sql = _render_frame_bound(frame.start, "start")
        end_sql = _render_frame_bound(frame.end, "end")
        parts.append(f"{frame.kind.upper()} BETWEEN {start_sql} AND {end_sql}")
    return " ".join(parts)


class DuckdbRelation:
    """Lazy wrapper around DuckDBPyRelation.

    All operations delegate to DuckDB's native lazy Relational API;
    no temp views or tables are created.

    SQL injection prevention:
    - Identifiers go through ``quote_ident`` (SQL-standard double-quote escaping).
    - DuckDB's Relational API does not support PEP 249 parameterized queries,
      so ``filter()`` falls back to ``inline_params`` / ``quote_value`` for
      literal values. See ``sql_utils`` module docstring for details.
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

    @property
    def types(self) -> list[Any]:
        return list(self._relation.dtypes)

    def filter(self, condition: str, params: tuple[Any, ...] = ()) -> "DuckdbRelation":
        """Apply a filter condition. Params are inlined via ``quote_value``
        because DuckDB's Relational API lacks PEP 249 parameter binding."""
        if params:
            condition = inline_params(condition, params)
        new_rel = self._relation.filter(condition)
        return DuckdbRelation(self._connection, new_rel)

    def select(self, *columns: str) -> "DuckdbRelation":
        """Project named columns, quoting each identifier."""
        projection = ", ".join(quote_ident(c) for c in columns)
        new_rel = self._relation.project(projection)
        return DuckdbRelation(self._connection, new_rel)

    def project(self, expression: str) -> "DuckdbRelation":
        """Project columns using a raw SQL projection string.

        Bypasses identifier quoting; never pass user-controlled input.
        """
        new_rel = self._relation.project(expression)
        return DuckdbRelation(self._connection, new_rel)

    def aggregate(self, agg_expr: str, group_by: str = "") -> "DuckdbRelation":
        """GROUP BY aggregation. Thin wrapper around ``DuckDBPyRelation.aggregate``.

        Both arguments are SQL fragments inlined verbatim; never pass
        user-controlled input.
        """
        new_rel = self._relation.aggregate(agg_expr, group_by) if group_by else self._relation.aggregate(agg_expr)
        return DuckdbRelation(self._connection, new_rel)

    def query(self, view_name: str, sql: str) -> "DuckdbRelation":
        """Execute a full SELECT statement against this relation under ``view_name``.

        Thin wrapper around ``DuckDBPyRelation.query``. SQL is inlined verbatim;
        never pass user-controlled input.
        """
        new_rel = self._relation.query(view_name, sql)
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

    def with_row_number(
        self,
        alias: str,
        partition_by: Sequence[str] = (),
        order_by: Sequence[str] = (),
    ) -> "DuckdbRelation":
        """Append a ROW_NUMBER() window column named ``alias``.

        All identifiers are quoted via ``quote_ident``; safe to pass column names verbatim.
        """
        if alias in self.columns:
            raise ValueError(f"Column {alias!r} already exists in the relation")
        clauses: list[str] = []
        if partition_by:
            clauses.append("PARTITION BY " + ", ".join(quote_ident(c) for c in partition_by))
        if order_by:
            clauses.append("ORDER BY " + ", ".join(quote_ident(c) for c in order_by))
        over = " ".join(clauses)
        return self.project(f"*, ROW_NUMBER() OVER ({over}) AS {quote_ident(alias)}")

    def window(
        self,
        func: str,
        alias: str,
        *,
        partition_by: Sequence[str] = (),
        order_by: Sequence[str] = (),
        frame: WindowFrame | None = None,
    ) -> "DuckdbRelation":
        """Append a window-function column ``alias`` computed by ``func`` over the OVER clause.

        ``func`` is a raw SQL fragment inlined verbatim; never pass user-controlled input.
        The ``alias`` and every identifier in ``partition_by`` / ``order_by`` are quoted
        via ``quote_ident``.
        """
        if alias in self.columns:
            raise ValueError(f"Column {alias!r} already exists in the relation")
        over_sql = _render_over_clause(partition_by, order_by, frame)
        return self.project(f"*, {func} OVER ({over_sql}) AS {quote_ident(alias)}")

    def append_column(self, name: str, values: list[Any]) -> "DuckdbRelation":
        """Return a new relation with an additional column appended positionally."""
        if name in self.columns:
            raise ValueError(f"Column {name!r} already exists in the relation")
        new_col_rel = DuckdbRelation.from_dict(self._connection, {name: values})
        rn = pick_helper_column_name(taken=set(self.columns) | {name})
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
