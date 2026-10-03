"""Lock file recording which classes a resolved plan picks, so a change fails loudly."""

import difflib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mloda.core.api.plan_info import PlanStep

PLAN_LOCK_FORMAT = 1


class PlanLockMismatchError(Exception):
    """The resolved plan differs from the lock file, or the lock file is missing."""


def _class_path(cls: type | None) -> str | None:
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
            access = step.reader_data_access
            compute.append(
                {
                    "feature_names": sorted(step.feature_names),
                    "feature_group": _class_path(step.feature_group),
                    "compute_framework": _class_path(step.compute_framework),
                    "specialized_from": sorted(_class_path(parent) or "" for parent in step.specialized_from),
                    "reader": None if access is None else _class_path(access[0]),
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
        else:
            transforms.append(
                {
                    "feature_group": _class_path(step.feature_group),
                    "from_compute_framework": _class_path(step.source_compute_framework),
                    "to_compute_framework": _class_path(step.compute_framework),
                }
            )
    return {
        "format": PLAN_LOCK_FORMAT,
        "requested_features": sorted(requested),
        "compute": _sorted_records(compute),
        "joins": _sorted_records(joins),
        "transforms": _sorted_records(transforms),
    }


def _lock_text(plan: Sequence[PlanStep]) -> str:
    return json.dumps(_lock_content(plan), sort_keys=True, indent=2) + "\n"


def _main_paths(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value} if value.startswith("__main__:") else set()
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, list):
        return {path for item in value for path in _main_paths(item)}
    return set()


def write_plan_lock(plan: Sequence[PlanStep], path: str | os.PathLike[str]) -> None:
    """Write the canonical lock file for the plan, refusing classes defined in __main__."""
    mains = _main_paths(_lock_content(plan))
    if mains:
        raise ValueError(f"Cannot lock classes defined in __main__, move them into a module: {sorted(mains)}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(_lock_text(plan))


def check_plan_lock(plan: Sequence[PlanStep], path: str | os.PathLike[str]) -> None:
    """Raise PlanLockMismatchError unless the plan matches the lock file; never writes."""
    target = Path(path)
    resolved = _lock_text(plan)
    if not target.exists():
        raise PlanLockMismatchError(
            f"Plan lock file {target} does not exist. Call write_plan_lock(plan, path) to create it "
            f"with this content:\n{resolved}"
        )
    locked = json.loads(target.read_text(encoding="utf-8"))
    if locked == json.loads(resolved):
        return
    diff = difflib.unified_diff(
        (json.dumps(locked, sort_keys=True, indent=2) + "\n").splitlines(keepends=True),
        resolved.splitlines(keepends=True),
        fromfile=str(target),
        tofile="resolved",
    )
    raise PlanLockMismatchError(f"Resolved plan differs from {target}:\n{''.join(diff)}")
