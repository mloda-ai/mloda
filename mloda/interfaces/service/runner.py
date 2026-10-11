"""Run a verified plan id with mloda and return classified tables with a bounded, untrusted preview.

The request is planned again first, so policy, data-access rules, filters and row limits hold at run time.
"""

import math
import secrets
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from mloda.interfaces.service.errors import error_result, is_error
from mloda.interfaces.service.planner import Planner
from mloda.user import PlanStep


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

    def run(self, plan_id: Any) -> dict[str, Any]:
        """Execute the plan behind ``plan_id`` and return its tables, or an error result."""
        if self._planner.settings.mode != "execute":
            return error_result("policy_denied", "Execution is disabled; the service is in plan mode.")
        verified = self._planner.verify_plan_id(plan_id)
        if is_error(verified):
            return verified
        planned = self._planner.plan(verified["request"])
        if isinstance(planned, dict):
            return planned
        if planned.digest != verified["digest"]:
            message = "The plan no longer matches how the service resolves this request; call explain again."
            return error_result("plan_stale", message)
        result = self._planner.execute(planned)
        tables = [self._table(frame, planned.names, result.plan) for frame in result]
        return {"status": "completed", "tables": tables, "audit_id": f"run_{secrets.token_hex(8)}"}

    def _table(self, frame: Mapping[Any, Any], names: Sequence[str], steps: Sequence[PlanStep]) -> dict[str, Any]:
        settings = self._planner.settings
        data = {str(name): list(values) for name, values in frame.items()}
        ordered = [name for name in names if name in data] + sorted(set(data) - set(names))
        labels = self._planner.classifications(ordered, steps)
        allowed = [name for name in ordered if labels[name] in settings.preview_allows]
        row_count = len(next(iter(data.values()), []))
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
