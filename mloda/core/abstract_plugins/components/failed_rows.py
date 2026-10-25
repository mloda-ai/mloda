"""Rows a feature group could not give a value, reported from calculate_feature and collected per step.

Outside a collector (direct calls, streams) reporting is a no-op.
"""

from __future__ import annotations

import contextvars
import re
from collections.abc import Sequence
from dataclasses import dataclass

_REASON = re.compile(r"[A-Za-z0-9_.-]{1,64}")


@dataclass(frozen=True)
class FailedRows:
    """Rows of one output column that got no value; rows are positions in the feature group's own output."""

    feature_group: str
    column: str
    rows: tuple[int, ...]
    reason: str


class FailedRowsCollector:
    """One step's collector: the feature group name and the entries reported while it is open."""

    def __init__(self, feature_group: str) -> None:
        self.feature_group = feature_group
        self.entries: list[FailedRows] = []
        self._tokens: list[contextvars.Token[FailedRowsCollector | None]] = []

    # Class-based, not @contextmanager: that assigns __traceback__ on the exception and fails for a frozen one.
    def __enter__(self) -> FailedRowsCollector:
        self._tokens.append(_COLLECTOR.set(self))
        return self

    def __exit__(self, *exc_info: object) -> None:
        _COLLECTOR.reset(self._tokens.pop())


_COLLECTOR: contextvars.ContextVar[FailedRowsCollector | None] = contextvars.ContextVar(
    "mloda_failed_rows_collector", default=None
)


def current_failed_rows_collector() -> FailedRowsCollector | None:
    """The collector open in the current context, else None."""
    return _COLLECTOR.get()


def report_failed_rows(column: str, rows: Sequence[int], reason: str) -> None:
    """Report rows of column that got no value. reason is a short code, never row data."""
    if not isinstance(column, str):
        raise TypeError("column must be a str")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("rows must be a sequence of ints")
    for row in rows:
        if isinstance(row, bool) or not isinstance(row, int):
            raise TypeError("rows must contain only ints")
        if row < 0:
            raise ValueError("rows must be non-negative")
    if not isinstance(reason, str) or not _REASON.fullmatch(reason):
        raise ValueError("reason must be a short code of 1 to 64 characters from [A-Za-z0-9_.-]")
    collector = _COLLECTOR.get()
    if collector is None or not rows:
        return
    collector.entries.append(FailedRows(collector.feature_group, column, tuple(rows), reason))


def collect_failed_rows(feature_group: str) -> FailedRowsCollector:
    """A collector to open with `with` around one feature group's calculate step."""
    return FailedRowsCollector(feature_group)
