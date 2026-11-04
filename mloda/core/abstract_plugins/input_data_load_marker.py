"""Per-calculation marker a FeatureSet carries so a thread-hopped reader load finds its run's INPUT_DATA_LOAD hook."""

from contextvars import Context, ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mloda.core.abstract_plugins.compute_framework import ComputeFramework
    from mloda.core.abstract_plugins.hook_context import HookContext


class InputDataLoadMarker:
    """Holds (framework, calculate HookContext) and a snapshot of the calculation's contextvars while its
    calculation runs; a thread hop re-dispatches inside a copy of the snapshot instead of restoring values
    one by one. Survives copies, pickles inactive (the snapshot is runtime-only and never pickled)."""

    __slots__ = ("active", "context", "framework", "gate", "has_extender")

    def __init__(self, has_extender: bool, gate: bool) -> None:
        self.active: tuple[ComputeFramework, HookContext] | None = None
        self.framework: ComputeFramework | None = None
        self.context: Context | None = None
        self.has_extender = has_extender
        self.gate = gate

    def __copy__(self) -> "InputDataLoadMarker":
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> "InputDataLoadMarker":
        return self

    def __reduce__(self) -> tuple[Any, ...]:
        return (InputDataLoadMarker, (self.has_extender, self.gate))


current_input_data_load_marker: ContextVar[InputDataLoadMarker | None] = ContextVar(
    "current_input_data_load_marker", default=None
)
