"""Extender reporting or refusing float columns that hold both NaN and null when moved onto a NaN-is-null framework."""

import logging
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from mloda.core.optional_dependency import loaded, require
from mloda.steward import Extender, ExtenderHook, HookContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NanNullLoss:
    """A float column that held both NaN and null before a conversion onto a NaN-is-null framework."""

    step_uuid: UUID | None
    column: str
    source_framework: str
    target_framework: str


class NanNullLossError(ValueError):
    """Raised by a refusing NanNullLossExtender when a conversion would merge NaN and null or cannot be checked."""


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


class NanNullLossExtender(Extender):
    """Checks the final hop of a framework conversion for columns holding both NaN and null."""

    def __init__(self, refuse: bool = False) -> None:
        self.refuse = refuse
        if refuse:
            self.never_fall_back = True
        else:
            self.raise_on_error = False

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FRAMEWORK_TRANSFORM}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        context = HookContext.current()
        source = None if context is None else context.source_compute_framework
        target = None if context is None else context.target_compute_framework
        # Mirrors the plan mark PlanStep.potential_nan_null_loss.
        if context is None or source is None or target is None or source.nan_is_null() or not target.nan_is_null():
            return func(*args, **kwargs)
        if args[1] != target.expected_data_framework():
            return func(*args, **kwargs)

        source_name = source.get_class_name()
        target_name = target.get_class_name()
        pa = loaded("pyarrow")
        if pa is None or args[0] is not pa.Table:
            if self.refuse:
                raise NanNullLossError(
                    f"Step {context.step_uuid}: the {source_name} to {target_name} conversion could not be "
                    "checked for NaN and null loss because its final hop does not start from Arrow."
                )
            return func(*args, **kwargs)

        columns = nan_and_null_columns(args[2])
        if columns and self.refuse:
            names = ", ".join(f"'{c}'" for c in columns)
            raise NanNullLossError(
                f"Step {context.step_uuid} ({source_name} to {target_name}): column(s) {names} "
                f"hold both NaN and null, which {target_name} cannot tell apart."
            )
        for column in columns:
            self.report(NanNullLoss(context.step_uuid, column, source_name, target_name))
        return func(*args, **kwargs)

    def report(self, loss: NanNullLoss) -> None:
        """Log one WARNING for the loss; override to route records elsewhere."""
        logger.warning(
            "Step %s: column '%s' holds both NaN and null, which %s cannot tell apart (from %s).",
            loss.step_uuid,
            loss.column,
            loss.target_framework,
            loss.source_framework,
        )
