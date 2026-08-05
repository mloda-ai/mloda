"""Per-candidate match-rejection window shared by feature-group matching and input-data reader selection.
Neutral home so reader code does not depend on the feature-chainer module."""

from __future__ import annotations

import contextvars
from dataclasses import dataclass


@dataclass(frozen=True)
class MatchRejection:
    """One recorded rejection: the reason plus a free-form stage hint the engine maps at the harvest.

    Unknown stage values fall back to value_rejection there; this neutral module does not validate them.
    """

    reason: str
    stage: str = "value_rejection"


# The owned stage marks a veto recorded while the user explicitly addressed the reader family.
INPUT_DATA_STAGE = "input_data"
INPUT_DATA_OWNED_STAGE = "input_data_owned"


# Active for one candidate's match call: the engine opens a window per candidate. Maps the recording
# site's owner name to the first structured rejection the real match pass produced, and the
# engine attributes the harvest to the candidate class object it called.
MATCH_REJECTION_REASONS: contextvars.ContextVar[dict[str, MatchRejection] | None] = contextvars.ContextVar(
    "mloda_match_rejection_reasons", default=None
)


def record_match_rejection(owner_name: str, reason: str, stage: str = "value_rejection") -> None:
    """Record a match rejection; the first per owner wins, and outside an active evaluation it is a no-op."""
    reasons = MATCH_REJECTION_REASONS.get()
    if reasons is None:
        return
    reasons.setdefault(owner_name, MatchRejection(reason, stage))


def match_rejection_owners() -> frozenset[str]:
    """Owner names already recorded in the active window; empty outside one."""
    reasons = MATCH_REJECTION_REASONS.get()
    if reasons is None:
        return frozenset()
    return frozenset(reasons)


def restamp_match_rejections_since(known_owners: frozenset[str], from_stage: str, to_stage: str) -> None:
    """Re-stamp owners recorded after the snapshot whose stage is exactly from_stage; no-op outside a window."""
    # Scoping by snapshot delta, not by owner name, covers whatever name an inner delegation stamped.
    reasons = MATCH_REJECTION_REASONS.get()
    if reasons is None:
        return
    for owner_name, rejection in list(reasons.items()):
        if owner_name in known_owners or rejection.stage != from_stage:
            continue
        reasons[owner_name] = MatchRejection(rejection.reason, to_stage)


def has_match_rejection(stage: str) -> bool:
    """True iff the active window holds a rejection with exactly this stage; False without an active window."""
    reasons = MATCH_REJECTION_REASONS.get()
    if reasons is None:
        return False
    return any(rejection.stage == stage for rejection in reasons.values())
