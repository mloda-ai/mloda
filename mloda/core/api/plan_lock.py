"""Lock file recording which classes a resolved plan picks, so a change fails loudly."""

import difflib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mloda.core.api.plan_info import PlanStep

PLAN_LOCK_FORMAT = 4


class PlanLockMismatchError(Exception):
    """The resolved plan differs from the lock file, or the lock file is missing."""


def _class_path(cls: Any) -> str | None:
    return None if cls is None else f"{cls.__module__}:{cls.__qualname__}"


def _sorted_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda record: json.dumps(record, sort_keys=True))


def _lock_content(plan: Sequence[PlanStep]) -> dict[str, Any]:
    requested: set[str] = set()
    compute: list[dict[str, Any]] = []
    joins: list[dict[str, Any]] = []
    transforms: list[dict[str, Any]] = []
    for step in plan:
        if step.step_kind == "compute":
            requested.update(step.requested_feature_names)
            compute.append(
                {
                    "feature_names": sorted(step.feature_names),
                    "feature_group": _class_path(step.feature_group),
                    "compute_framework": _class_path(step.compute_framework),
                    "compute_framework_reason": step.compute_framework_reason,
                    "specialized_from": sorted(_class_path(parent) or "" for parent in step.specialized_from),
                    "result_framework": _class_path(step.result_framework),
                }
            )
        elif step.step_kind == "join":
            joins.append(
                {
                    "left_feature_group": _class_path(step.feature_group),
                    "right_feature_group": _class_path(step.source_feature_group),
                    "join_type": step.join_type,
                    "compute_framework": _class_path(step.compute_framework),
                    "source_compute_framework": _class_path(step.source_compute_framework),
                    "destination_side": step.join_destination_side,
                }
            )
        elif step.step_kind == "transform":
            transforms.append(
                {
                    "feature_group": _class_path(step.feature_group),
                    "from_compute_framework": _class_path(step.source_compute_framework),
                    "to_compute_framework": _class_path(step.compute_framework),
                }
            )
        else:
            raise ValueError(f"Unknown plan step kind {step.step_kind!r}.")
    return {
        "format": PLAN_LOCK_FORMAT,
        "requested_features": sorted(requested),
        "compute": _sorted_records(compute),
        "joins": _sorted_records(joins),
        "transforms": _sorted_records(transforms),
    }


def _dump(content: dict[str, Any]) -> str:
    return json.dumps(content, sort_keys=True, indent=2) + "\n"


def _lock_text(plan: Sequence[PlanStep]) -> str:
    return _dump(_lock_content(plan))


_PATH_FIELDS = {
    "compute": ("feature_group", "compute_framework", "result_framework"),
    "joins": ("left_feature_group", "right_feature_group", "compute_framework", "source_compute_framework"),
    "transforms": ("feature_group", "from_compute_framework", "to_compute_framework"),
}


def _main_paths(content: dict[str, Any]) -> set[str]:
    paths: set[str | None] = set()
    for section, fields in _PATH_FIELDS.items():
        for record in content[section]:
            paths.update(record[field] for field in fields)
            paths.update(record.get("specialized_from", []))
    return {path for path in paths if path and path.startswith("__main__:")}


def write_plan_lock(plan: Sequence[PlanStep], path: str | os.PathLike[str]) -> None:
    """Write the canonical lock file for the plan, refusing classes defined in __main__."""
    content = _lock_content(plan)
    mains = _main_paths(content)
    if mains:
        raise ValueError(f"Cannot lock classes defined in __main__, move them into a module: {sorted(mains)}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(_dump(content))


def check_plan_lock(plan: Sequence[PlanStep], path: str | os.PathLike[str]) -> None:
    """Raise PlanLockMismatchError unless the plan matches the lock file; never writes."""
    target = Path(path)
    resolved = _lock_text(plan)
    if not target.exists():
        raise PlanLockMismatchError(
            f"Plan lock file {target} does not exist. Call write_plan_lock(plan, path) to create it "
            f"with this content:\n{resolved}"
        )
    locked = json.loads(target.read_text(encoding="utf-8-sig"))
    if locked == json.loads(resolved):
        return
    diff = difflib.unified_diff(
        (json.dumps(locked, sort_keys=True, indent=2) + "\n").splitlines(keepends=True),
        resolved.splitlines(keepends=True),
        fromfile=str(target),
        tofile="resolved",
    )
    raise PlanLockMismatchError(f"Resolved plan differs from {target}:\n{''.join(diff)}")
