"""SQL window and frame primitives shared across SQL-backed relations.

Pure SQL-fragment generators with no engine-specific logic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

__all__ = [
    "CurrentRow",
    "Unbounded",
    "Preceding",
    "Following",
    "FrameBound",
    "WindowFrame",
    "OrderBy",
    "validate_window",
    "render_over_clause",
    "render_frame_bound",
    "render_order_by_item",
    "bound_rank",
]


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
        if self.offset < 0:
            raise ValueError(f"Preceding offset must be >= 0; got {self.offset}")


@dataclass(frozen=True)
class Following:
    """Frame bound: ``offset`` rows/range/groups after the current row."""

    offset: int

    def __post_init__(self) -> None:
        if isinstance(self.offset, bool) or not isinstance(self.offset, int):
            raise TypeError(f"Following offset must be int; got {type(self.offset).__name__}")
        if self.offset < 0:
            raise ValueError(f"Following offset must be >= 0; got {self.offset}")


@dataclass(frozen=True)
class OrderBy:
    """An order_by item with optional direction and NULLS placement.

    Bare strings in order_by lists are quoted as plain column names; use
    OrderBy to express direction or NULLS handling.
    """

    column: str
    descending: bool = False
    nulls: Literal["first", "last"] | None = None

    def __post_init__(self) -> None:
        if self.nulls is not None and self.nulls not in ("first", "last"):
            raise ValueError(f"OrderBy nulls must be 'first', 'last', or None; got {self.nulls!r}")


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
        if bound_rank(self.start, "start") > bound_rank(self.end, "end"):
            raise ValueError(f"WindowFrame start must not come after end; got start={self.start!r}, end={self.end!r}")


def render_frame_bound(bound: FrameBound, side: Literal["start", "end"]) -> str:
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


def render_order_by_item(item: str | OrderBy) -> str:
    if isinstance(item, str):
        return quote_ident(item)
    parts = [quote_ident(item.column)]
    if item.descending:
        parts.append("DESC")
    if item.nulls is not None:
        parts.append(f"NULLS {item.nulls.upper()}")
    return " ".join(parts)


def bound_rank(bound: FrameBound, side: Literal["start", "end"]) -> float:
    if isinstance(bound, Unbounded):
        return float("-inf") if side == "start" else float("inf")
    if isinstance(bound, Preceding):
        return -bound.offset
    if isinstance(bound, CurrentRow):
        return 0.0
    if isinstance(bound, Following):
        return float(bound.offset)
    raise TypeError(f"Unsupported frame bound: {type(bound).__name__}")


def validate_window(
    order_by: Sequence[str | OrderBy],
    frame: WindowFrame | None,
) -> None:
    if frame is None or frame.kind != "range":
        return
    has_offset_bound = isinstance(frame.start, (Preceding, Following)) or isinstance(frame.end, (Preceding, Following))
    if not has_offset_bound:
        return
    if len(order_by) != 1:
        raise ValueError(
            f"RANGE frame with numeric offset bounds requires exactly one ORDER BY column; got {len(order_by)}"
        )


def render_over_clause(
    partition_by: Sequence[str],
    order_by: Sequence[str | OrderBy],
    frame: WindowFrame | None,
) -> str:
    """Render the body of an ``OVER (...)`` clause."""
    parts: list[str] = []
    if partition_by:
        parts.append("PARTITION BY " + ", ".join(quote_ident(c) for c in partition_by))
    if order_by:
        parts.append("ORDER BY " + ", ".join(render_order_by_item(c) for c in order_by))
    if frame is not None:
        start_sql = render_frame_bound(frame.start, "start")
        end_sql = render_frame_bound(frame.end, "end")
        parts.append(f"{frame.kind.upper()} BETWEEN {start_sql} AND {end_sql}")
    return " ".join(parts)
