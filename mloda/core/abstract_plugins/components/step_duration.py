"""Wall-clock time of one feature group step's calculate_feature call; step_uuid is PlanStep.step_uuid."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class StepDuration:
    """Provisional: seconds one feature group step spent in calculate_feature; may grow into per-step metrics."""

    feature_group: str
    step_uuid: UUID
    duration_seconds: float
