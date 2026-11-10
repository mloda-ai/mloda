"""Per-calculation marker a FeatureSet carries so a thread-hopped reader load finds its run's INPUT_DATA_LOAD hook."""

import contextvars
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mloda.core.abstract_plugins.compute_framework import ComputeFramework


class InputDataLoadMarker:
    """Holds a snapshot of the calculation's context while it runs; survives copies, pickles without it."""

    __slots__ = ("context", "framework", "gate")

    def __init__(self, gate: bool) -> None:
        self.context: contextvars.Context | None = None
        self.framework: ComputeFramework | None = None
        self.gate = gate

    def __copy__(self) -> "InputDataLoadMarker":
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> "InputDataLoadMarker":
        return self

    def __reduce__(self) -> tuple[Any, ...]:
        return (InputDataLoadMarker, (self.gate,))


current_input_data_load_marker: ContextVar[InputDataLoadMarker | None] = ContextVar(
    "current_input_data_load_marker", default=None
)
