from dataclasses import dataclass, field
from typing import Any

from mloda.core.abstract_plugins.components.failed_rows import FailedRows
from mloda.core.abstract_plugins.components.step_duration import StepDuration


@dataclass(frozen=True)
class RunOutcome:
    """Per-run outputs a run hands to its result."""

    artifacts: dict[str, Any] = field(default_factory=dict)
    failed_rows: tuple[FailedRows, ...] = ()
    step_durations: tuple[StepDuration, ...] = ()
