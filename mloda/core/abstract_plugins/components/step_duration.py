"""Wall-clock time of one feature group step's calculate_feature call."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class StepDuration:
    """Seconds one feature group step spent in calculate_feature; step_uuid is PlanStep.step_uuid."""

    feature_group: str
    step_uuid: UUID
    duration_seconds: float
