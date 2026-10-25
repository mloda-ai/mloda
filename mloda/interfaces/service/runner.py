"""Run a verified plan id with mloda and return classified tables with a bounded, untrusted preview.

The request is planned again first, so policy, data-access rules, filters and row limits hold at run time.
"""

import math
import re
import secrets
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from mloda.core.abstract_plugins.components.failed_rows import FailedRows
from mloda.interfaces.service.errors import error_result, failure_result, is_error, strip_control_characters
from mloda.interfaces.service.planner import Planner
from mloda.interfaces.service.request import MAX_NAME_LENGTH
from mloda.user import PlanStep

FAILED_ROWS_CAP = 1000
MAX_FAILED_ROW_ENTRIES = 100

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


class Runner:
    """Executes plan ids issued by one Planner when its settings allow execution."""

    def __init__(self, planner: Planner) -> None:
        self._planner = planner

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
        build = self._full_table if full else self._table
        tables = [build(frame, planned.names, result.plan) for frame in result]
        max_rows = self._planner.settings.max_rows
        over = [table["row_count"] for table in tables if max_rows is not None and table["row_count"] > max_rows]
        if over:
            details = [{"reason": "result_over_limit", "rows": rows, "max_rows": max_rows} for rows in over]
            return error_result("limit_exceeded", f"A result table has more rows than max_rows ({max_rows}).", details)
        completed: dict[str, Any] = {"status": "completed", "tables": tables, "audit_id": f"run_{secrets.token_hex(8)}"}
        if result.failed_rows:
            completed["failed_rows"] = self._failed_rows(result.failed_rows, full)
        return completed

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

    def _table(self, frame: Mapping[Any, Any], names: Sequence[str], steps: Sequence[PlanStep]) -> dict[str, Any]:
        settings = self._planner.settings
        data, ordered, labels, row_count = self._frame_parts(frame, names, steps)
        allowed = [name for name in ordered if labels[name] in settings.preview_allows]
        rows = [
            {name: _json_safe(data[name][index]) for name in allowed}
            for index in range(min(row_count, settings.max_preview_rows) if allowed else 0)
        ]
        return {
            "row_count": row_count,
            "columns": [
                {"name": name, "classification": labels[name], "in_preview": name in allowed} for name in ordered
            ],
            "preview": {"untrusted": True, "truncated": row_count > len(rows), "rows": rows},
        }
