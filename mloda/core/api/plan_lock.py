"""Lock file recording which classes a resolved plan picks, so a change fails loudly."""

import difflib
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mloda.core.abstract_plugins.components.credential_scrub import scrub_credentials
from mloda.core.api.plan_info import PlanStep
from mloda.core.prepare.choose_compute_frameworks import stable_text

PLAN_LOCK_FORMAT = 3


class PlanLockMismatchError(Exception):
    """The resolved plan differs from the lock file, or the lock file is missing."""


def _class_path(cls: type | None) -> str | None:
    return None if cls is None else f"{cls.__module__}:{cls.__qualname__}"


def _sorted_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda record: json.dumps(record, sort_keys=True))


def _build(plan: Sequence[PlanStep]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the lock content and the wider content-hash content, built in one pass."""
    requested: set[str] = set()
    lock: dict[str, list[dict[str, Any]]] = {"compute": [], "joins": [], "transforms": []}
    wide: dict[str, list[dict[str, Any]]] = {"compute": [], "joins": [], "transforms": []}
    for step in plan:
        if step.step_kind == "compute":
            requested.update(step.requested_feature_names)
            access = step.reader_data_access
            record = {
                "feature_names": sorted(step.feature_names),
                "feature_group": _class_path(step.feature_group),
                "compute_framework": _class_path(step.compute_framework),
                "compute_framework_reason": step.compute_framework_reason,
                "specialized_from": sorted(_class_path(parent) or "" for parent in step.specialized_from),
                "reader": None if access is None else _class_path(access[0]),
                "result_framework": _class_path(step.result_framework),
            }
            lock["compute"].append(record)
            wide_record: dict[str, Any] = {
                key: value for key, value in record.items() if key not in ("compute_framework_reason", "reader")
            }
            wide_record["input_feature_edges"] = {
                name: sorted(inputs) for name, inputs in step.input_feature_edges.items()
            }
            wide_record["options"] = scrub_credentials(stable_text(step.feature_set_options))
            wide["compute"].append(wide_record)
        elif step.step_kind == "join":
            record = {
                "left_feature_group": _class_path(step.feature_group),
                "right_feature_group": _class_path(step.source_feature_group),
                "join_type": step.join_type,
                "compute_framework": _class_path(step.compute_framework),
                "source_compute_framework": _class_path(step.source_compute_framework),
                "destination_side": step.join_destination_side,
            }
            lock["joins"].append(record)
            wide["joins"].append({**record, "join_keys": None if step.join_keys is None else list(step.join_keys)})
        elif step.step_kind == "transform":
            record = {
                "feature_group": _class_path(step.feature_group),
                "from_compute_framework": _class_path(step.source_compute_framework),
                "to_compute_framework": _class_path(step.compute_framework),
            }
            lock["transforms"].append(record)
            wide["transforms"].append(record)
        else:
            raise ValueError(f"Unknown plan step kind {step.step_kind!r}.")
    sorted_requested = sorted(requested)
    return (
        {
            "format": PLAN_LOCK_FORMAT,
            "requested_features": sorted_requested,
            **{key: _sorted_records(records) for key, records in lock.items()},
        },
        {
            "requested_features": sorted_requested,
            **{key: _sorted_records(records) for key, records in wide.items()},
        },
    )


def _lock_content(plan: Sequence[PlanStep]) -> dict[str, Any]:
    return _build(plan)[0]


def _dump(content: dict[str, Any]) -> str:
    return json.dumps(content, sort_keys=True, indent=2) + "\n"


def _lock_text(plan: Sequence[PlanStep]) -> str:
    return _dump(_lock_content(plan))


def plan_structure_hash(plan: Sequence[PlanStep]) -> str:
    """Return the sha256 of the plan-lock text: an equal hash means an equal lock file."""
    return hashlib.sha256(_lock_text(plan).encode("utf-8")).hexdigest()


def plan_content_hash(plan: Sequence[PlanStep]) -> str:
    """Return the sha256 of the plan's audit content: structure plus scrubbed options, wiring and join keys."""
    text = json.dumps(_build(plan)[1], sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_PATH_FIELDS = {
    "compute": ("feature_group", "compute_framework", "reader", "result_framework"),
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
