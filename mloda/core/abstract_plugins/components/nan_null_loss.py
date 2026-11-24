"""Run-time check for float columns that hold both NaN and null when moved onto a NaN-is-null framework."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID

from mloda.core.optional_dependency import loaded, require

if TYPE_CHECKING:
    from mloda.core.abstract_plugins.compute_framework import ComputeFramework


@dataclass(frozen=True)
class NanNullLoss:
    """A float column that held both NaN and null before a conversion onto a NaN-is-null framework."""

    step_uuid: UUID
    column: str
    source_framework: str
    target_framework: str


class NanNullLossError(ValueError):
    """Raised under ``refuse_nan_null_loss`` when a conversion would merge NaN and null or cannot be checked."""


def nan_null_loss_pair(source: "type[ComputeFramework] | None", target: "type[ComputeFramework] | None") -> bool:
    """True when the target folds NaN into null and the source does not."""
    return source is not None and target is not None and not bool(source.nan_is_null()) and bool(target.nan_is_null())


def nan_and_null_columns(table: Any) -> tuple[str, ...]:
    """Top-level floating columns of a pyarrow table holding at least one NaN and one null."""
    pa = require("pyarrow", "the NaN/null loss check")
    pc = require("pyarrow.compute", "the NaN/null loss check")

    names: list[str] = []
    for field, column in zip(table.schema, table.columns):
        if not pa.types.is_floating(field.type) or column.null_count == 0:
            continue
        if pc.any(pc.is_nan(column)).as_py():
            names.append(field.name)
    return tuple(names)


class NanNullLossInspector:
    """Hop inspector for one marked conversion; acts at the pa.Table hop and tracks whether it saw one."""

    def __init__(
        self,
        step_uuid: UUID,
        source: "type[ComputeFramework]",
        target: "type[ComputeFramework]",
        description: str,
        refuse: bool,
        report: Callable[[list[NanNullLoss]], None],
    ) -> None:
        self.step_uuid = step_uuid
        self.source_name = source.get_class_name()
        self.target_name = target.get_class_name()
        self.description = description
        self.refuse = refuse
        self.report = report
        self.checked = False

    @classmethod
    def for_pair(
        cls,
        step_uuid: UUID,
        source: "type[ComputeFramework]",
        target: "type[ComputeFramework]",
        description: str,
        refuse: bool,
        report: Callable[[list[NanNullLoss]], None],
    ) -> "NanNullLossInspector | None":
        """The inspector for a marked pair, or None when the pair cannot lose."""
        if not nan_null_loss_pair(source, target):
            return None
        return cls(step_uuid, source, target, description, refuse, report)

    def __call__(self, current_fw: Any, data: Any) -> None:
        pa = loaded("pyarrow")
        if pa is None or current_fw is not pa.Table:
            return
        self.checked = True
        columns = nan_and_null_columns(data)
        if columns and self.refuse:
            names = ", ".join(f"'{c}'" for c in columns)
            raise NanNullLossError(
                f"{self.description} (step {self.step_uuid}, {self.source_name} to {self.target_name}): "
                f"column(s) {names} hold both NaN and null, which {self.target_name} cannot tell apart "
                "(refuse_nan_null_loss=True)."
            )
        if columns:
            self.report([NanNullLoss(self.step_uuid, c, self.source_name, self.target_name) for c in columns])

    def finish(self) -> None:
        """Refuses a conversion that never passed an Arrow hop when strict."""
        if self.refuse and not self.checked:
            raise NanNullLossError(
                f"{self.description}: the {self.source_name} to {self.target_name} conversion could not be "
                "checked for NaN and null loss because it does not pass through Arrow (refuse_nan_null_loss=True)."
            )
