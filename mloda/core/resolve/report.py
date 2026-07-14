from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mloda.core.resolve.environment import EnvironmentBuildOutcome
from mloda.core.resolve.outcome import ResolutionOutcome


@dataclass(frozen=True)
class FeatureResolutionRecord:
    """One identification of the planning pass: dependency path plus its structured outcome."""

    dependency_path: tuple[str, ...]
    outcome: ResolutionOutcome

    def to_payload(self) -> dict[str, Any]:
        return {"dependency_path": list(self.dependency_path), "outcome": self.outcome.to_payload()}


@dataclass(frozen=True)
class ResolutionReport:
    """Whole-request diagnostics: the environment build plus every per-feature outcome."""

    environment: EnvironmentBuildOutcome | None
    features: tuple[FeatureResolutionRecord, ...]
    complete: bool

    def to_payload(self) -> dict[str, Any]:
        return {
            "environment": self.environment.to_payload() if self.environment is not None else None,
            "features": [record.to_payload() for record in self.features],
            "complete": self.complete,
        }
