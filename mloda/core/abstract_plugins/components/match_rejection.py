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
