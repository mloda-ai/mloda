from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.resolve.identity import PluginIdentity


class ResolutionStatus(Enum):
    """Final verdict of one resolve() call."""

    RESOLVED = "resolved"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"
    FAILED = "failed"


class CandidateStatus(Enum):
    """Per-candidate verdict inside one resolve() call."""

    WINNER = "winner"
    SURVIVOR = "survivor"
    SHADOWED = "shadowed"
    REJECTED = "rejected"
    FAILED = "failed"


class FrameworkStatus(Enum):
    """Per-candidate, per-framework verdict."""

    SUPPORTED = "supported"
    NOT_ENABLED = "not_enabled"
    UNAVAILABLE = "unavailable"
    PIN_EXCLUDED = "pin_excluded"
    CAPABILITY_REJECTED = "capability_rejected"
    HOOK_FAILED = "hook_failed"


class RejectionReason(Enum):
    """Structured reason attached to every candidate rejection."""

    CRITERIA = "criteria"
    DOMAIN = "domain"
    SCOPE = "scope"
    ABSTRACT = "abstract"
    NO_ACCESSIBLE_FRAMEWORK = "no_accessible_framework"
    CAPABILITY = "capability"
    FRAMEWORK_PIN = "framework_pin"
    LINK_INDEX = "link_index"
    VALUE_REJECTION = "value_rejection"
    PROVIDER_FAILURE = "provider_failure"
    SUBCLASS_SHADOWED = "subclass_shadowed"


@dataclass(frozen=True)
class Rejection:
    """One structured rejection: a reason plus optional safe detail text."""

    reason: RejectionReason
    detail: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {"reason": self.reason.value, "detail": self.detail}


@dataclass(frozen=True)
class PluginFailure:
    """One provider failure as plain data; never carries the exception object."""

    plugin: PluginIdentity
    stage: str
    category: str
    message: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "plugin": self.plugin.render(),
            "stage": self.stage,
            "category": self.category,
            "message": self.message,
        }


@dataclass(frozen=True)
class FrameworkEvaluation:
    """Verdict for one framework of one candidate."""

    framework: type[ComputeFramework]
    identity: PluginIdentity
    status: FrameworkStatus
    failure: PluginFailure | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "identity": self.identity.render(),
            "status": self.status.value,
            "failure": self.failure.to_payload() if self.failure is not None else None,
        }


@dataclass(frozen=True)
class CandidateEvaluation:
    """Full evaluation record of one accessible candidate."""

    feature_group: type[FeatureGroup]
    identity: PluginIdentity
    status: CandidateStatus
    frameworks: tuple[FrameworkEvaluation, ...] = ()
    rejections: tuple[Rejection, ...] = ()
    shadowed_by: PluginIdentity | None = None
    failure: PluginFailure | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "identity": self.identity.render(),
            "status": self.status.value,
            "frameworks": [evaluation.to_payload() for evaluation in self.frameworks],
            "rejections": [rejection.to_payload() for rejection in self.rejections],
            "shadowed_by": self.shadowed_by.render() if self.shadowed_by is not None else None,
            "failure": self.failure.to_payload() if self.failure is not None else None,
        }


@dataclass(frozen=True)
class ResolvedCandidate:
    """The winner: the feature group and its supported frameworks for this feature."""

    feature_group: type[FeatureGroup]
    identity: PluginIdentity
    compute_frameworks: tuple[type[ComputeFramework], ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "feature_group": self.identity.render(),
            "compute_frameworks": [
                PluginIdentity.from_class(framework).render() for framework in self.compute_frameworks
            ],
        }


@dataclass(frozen=True)
class ResolutionOutcome:
    """Structured result of one resolve() call; a winner exists iff status is RESOLVED."""

    status: ResolutionStatus
    winner: ResolvedCandidate | None
    candidates: tuple[CandidateEvaluation, ...]
    failures: tuple[PluginFailure, ...]
    environment_fingerprint: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "winner": self.winner.to_payload() if self.winner is not None else None,
            "candidates": [candidate.to_payload() for candidate in self.candidates],
            "failures": [failure.to_payload() for failure in self.failures],
            "environment_fingerprint": self.environment_fingerprint,
        }


class FeatureResolutionError(ValueError):
    """Engine-facing resolution error carrying the structured outcome."""

    def __init__(self, message: str, outcome: ResolutionOutcome) -> None:
        super().__init__(message)
        self.outcome = outcome
