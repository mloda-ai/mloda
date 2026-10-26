"""Run a verified plan id with mloda and return classified tables with a bounded, untrusted preview.

The request is planned again first, so policy, data-access rules, filters and row limits hold at run time.
"""

import hashlib
import math
import re
import secrets
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from typing import Any

from mloda.core.abstract_plugins.components.failed_rows import FailedRows
from mloda.interfaces.service.catalog import decode_cursor, encode_cursor
from mloda.interfaces.service.errors import (
    error_result,
    failure_result,
    is_error,
    problems_to_details,
    strip_control_characters,
)
from mloda.interfaces.service.planner import Planner
from mloda.interfaces.service.request import MAX_NAME_LENGTH, Problem
from mloda.user import PlanStep

FAILED_ROWS_CAP = 1000
MAX_FAILED_ROW_ENTRIES = 100
RESULT_TTL_SECONDS = 900
MAX_STORED_VALUES = 1_000_000

_WHITESPACE_CONTROLS = re.compile(r"[\t\n\r]")


def _column_label(column: str) -> str:
    return _WHITESPACE_CONTROLS.sub("", strip_control_characters(column)).strip()[:MAX_NAME_LENGTH]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


class ResultStore:
    """In-process store of preview-allowed result columns, bounded by age and total cell count."""

    def __init__(
        self,
        clock: Callable[[], float] = time.time,
        ttl_seconds: int = RESULT_TTL_SECONDS,
        max_values: int = MAX_STORED_VALUES,
    ) -> None:
        self._clock = clock
        self._ttl = ttl_seconds
        self._max = max_values
        self._lock = threading.Lock()
        self._entries: dict[str, tuple[float, int, list[dict[str, list[Any]]]]] = {}

    def _purge(self) -> None:
        now = self._clock()
        for key in [key for key, entry in self._entries.items() if entry[0] <= now]:
            del self._entries[key]

    def put(self, tables: list[dict[str, list[Any]]]) -> str | None:
        """Store ``tables`` and return their id, or None when they exceed the cell cap."""
        cells = sum(len(values) for table in tables for values in table.values())
        if cells > self._max:
            return None
        result_id = secrets.token_urlsafe(16)
        with self._lock:
            self._purge()
            while self._entries and sum(entry[1] for entry in self._entries.values()) + cells > self._max:
                del self._entries[next(iter(self._entries))]
            self._entries[result_id] = (self._clock() + self._ttl, cells, tables)
        return result_id

    def get(self, result_id: str) -> list[dict[str, list[Any]]] | None:
        with self._lock:
            self._purge()
            entry = self._entries.get(result_id)
        return None if entry is None else entry[2]


def _cursor_fingerprint(result_id: str, table: int) -> str:
    return hashlib.sha256(f"{result_id}:{table}".encode()).hexdigest()[:16]


def _invalid(path: str, message: str) -> dict[str, Any]:
    return error_result("validation_failed", message, problems_to_details([Problem(path, "invalid", message)]))


class Runner:
    """Executes plan ids issued by one Planner when its settings allow execution."""

    def __init__(self, planner: Planner, clock: Callable[[], float] = time.time) -> None:
        self._planner = planner
        self._results = ResultStore(clock)

    def run(self, plan_id: Any, full: bool = False) -> dict[str, Any]:
        """Run the plan behind ``plan_id``; ``full`` returns every row and column instead of the preview."""
        if self._planner.settings.mode != "execute":
            return error_result("policy_denied", "Execution is disabled; the service is in plan mode.")
        if full and not self._planner.settings.trusted_caller:
            return error_result("policy_denied", "Returning every row and column is not allowed for this caller.")
        verified = self._planner.verify_plan_id(plan_id)
        if is_error(verified):
            return verified
        planned = self._planner.plan(verified["request"])
        if isinstance(planned, dict):
            return planned
        if planned.digest != verified["digest"]:
            message = "The plan no longer matches how the service resolves this request; call explain again."
            return error_result("plan_stale", message)
        try:
            result = self._planner.execute(planned)
        except Exception as error:  # plugin boundary: the exception text may hold data, so only the type is reported
            return failure_result("The run", error)
        stored: list[dict[str, list[Any]]] = []
        if full:
            tables = [self._full_table(frame, planned.names, result.plan) for frame in result]
        else:
            built = [self._table(frame, planned.names, result.plan) for frame in result]
            tables = [table for table, _ in built]
            stored = [columns for _, columns in built]
        max_rows = self._planner.settings.max_rows
        over = [table["row_count"] for table in tables if max_rows is not None and table["row_count"] > max_rows]
        if over:
            details = [{"reason": "result_over_limit", "rows": rows, "max_rows": max_rows} for rows in over]
            return error_result("limit_exceeded", f"A result table has more rows than max_rows ({max_rows}).", details)
        completed: dict[str, Any] = {"status": "completed", "tables": tables, "audit_id": f"run_{secrets.token_hex(8)}"}
        if result.failed_rows:
            completed["failed_rows"] = self._failed_rows(result.failed_rows, full)
        settings = self._planner.settings
        if not full and not settings.trusted_caller and settings.result_paging and stored:
            self._store(completed, stored)
        return completed

    def _store(self, completed: dict[str, Any], stored: list[dict[str, list[Any]]]) -> None:
        tables = completed["tables"]
        truncated = [i for i, table in enumerate(tables) if table["preview"]["truncated"] and stored[i]]
        result_id = self._results.put(stored) if truncated else None
        if result_id is None:
            return
        completed["result_id"] = result_id
        for index in truncated:
            preview = tables[index]["preview"]
            fingerprint = _cursor_fingerprint(result_id, index)
            preview["next_cursor"] = encode_cursor(len(preview["rows"]), fingerprint)

    def fetch_rows(self, result_id: Any, table: Any = 0, cursor: Any = None, limit: Any = None) -> dict[str, Any]:
        """One page of the preview-allowed columns of a stored run result; never runs the plan again."""
        settings = self._planner.settings
        if settings.mode != "execute" or not settings.result_paging:
            return error_result("policy_denied", "Paging through results is disabled.")
        stored = self._results.get(result_id) if isinstance(result_id, str) else None
        if stored is None:
            return error_result("plan_invalid", "The result id is unknown or expired; run the plan again.")
        if isinstance(table, bool) or not isinstance(table, int) or not 0 <= table < len(stored):
            return _invalid("/table", f"'table' must be an integer from 0 to {len(stored) - 1}.")
        maximum = settings.max_preview_rows
        if limit is None:
            limit = maximum
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= maximum:
            return _invalid("/limit", f"'limit' must be an integer from 1 to {maximum}.")
        fingerprint = _cursor_fingerprint(result_id, table)
        offset = 0
        if cursor is not None:
            decoded = decode_cursor(cursor, fingerprint) if isinstance(cursor, str) else None
            if decoded is None:
                return _invalid("/cursor", "The cursor is malformed or belongs to another result.")
            offset = decoded
        columns = stored[table]
        total = len(next(iter(columns.values()), []))
        end = min(offset + limit, total)
        rows = [{name: _json_safe(values[index]) for name, values in columns.items()} for index in range(offset, end)]
        return {
            "untrusted": True,
            "rows": rows,
            "next_cursor": encode_cursor(end, fingerprint) if end < total else None,
        }

    @staticmethod
    def _failed_rows(entries: Sequence[FailedRows], full: bool) -> list[dict[str, Any]]:
        merged: dict[tuple[str, str], list[int]] = {}
        for entry in entries:
            merged.setdefault((_column_label(entry.column), entry.reason), []).extend(entry.rows)
        listed = []
        for (column, reason), rows in list(merged.items())[:MAX_FAILED_ROW_ENTRIES]:
            ordered = sorted(set(rows))
            shown = ordered if full else ordered[:FAILED_ROWS_CAP]
            listed.append(
                {
                    "column": column,
                    "reason": reason,
                    "row_count": len(ordered),
                    "rows": shown,
                    "truncated": len(shown) < len(ordered),
                }
            )
        return listed

    def _frame_parts(
        self, frame: Mapping[Any, Any], names: Sequence[str], steps: Sequence[PlanStep]
    ) -> tuple[dict[str, list[Any]], list[str], dict[str, str], int]:
        data = {str(name): list(values) for name, values in frame.items()}
        ordered = [name for name in names if name in data] + sorted(set(data) - set(names))
        labels = self._planner.classifications(ordered, steps)
        return data, ordered, labels, len(next(iter(data.values()), []))

    def _full_table(self, frame: Mapping[Any, Any], names: Sequence[str], steps: Sequence[PlanStep]) -> dict[str, Any]:
        data, ordered, labels, row_count = self._frame_parts(frame, names, steps)
        return {
            "row_count": row_count,
            "columns": [{"name": name, "classification": labels[name]} for name in ordered],
            "rows": [{name: _json_safe(data[name][index]) for name in ordered} for index in range(row_count)],
        }

    def _table(
        self, frame: Mapping[Any, Any], names: Sequence[str], steps: Sequence[PlanStep]
    ) -> tuple[dict[str, Any], dict[str, list[Any]]]:
        settings = self._planner.settings
        data, ordered, labels, row_count = self._frame_parts(frame, names, steps)
        allowed = [name for name in ordered if labels[name] in settings.preview_allows]
        rows = [
            {name: _json_safe(data[name][index]) for name in allowed}
            for index in range(min(row_count, settings.max_preview_rows) if allowed else 0)
        ]
        table = {
            "row_count": row_count,
            "columns": [
                {"name": name, "classification": labels[name], "in_preview": name in allowed} for name in ordered
            ],
            "preview": {"untrusted": True, "truncated": row_count > len(rows), "rows": rows},
        }
        return table, {name: data[name] for name in allowed}
