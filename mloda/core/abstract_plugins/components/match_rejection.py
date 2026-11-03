"""Per-candidate match-rejection window shared by feature-group matching and input-data source selection.
Neutral home so source group code does not depend on the feature-chainer module."""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum


class DeclineCode(StrEnum):
    """Memberless base for machine-readable decline codes; plugins subclass it (prefix values)."""


class CoreDeclineCode(DeclineCode):
    """Decline codes recorded by core match paths."""

    OUTPUT_PART_UNKNOWN = "output_part_unknown"
    IN_FEATURE_COUNT = "in_feature_count"
    OPTION_VALUE_REJECTED = "option_value_rejected"
    MATCH_GUARD_REJECTED = "match_guard_rejected"
    REQUIRED_OPTION_ABSENT = "required_option_absent"
    EXTENDER_REFUSED = "extender_refused"
    ABSTRACT_SOURCE = "abstract_source"
    AMBIGUOUS_SOURCE = "ambiguous_source"
    NO_QUERY_ROUTE = "no_query_route"
    COLUMN_NOT_FOUND = "column_not_found"
    COLUMNS_UNKNOWN = "columns_unknown"
    POINTED_ELSEWHERE = "pointed_elsewhere"
    NAME_NOT_DECLARED = "name_not_declared"
    SOURCE_NOT_FOUND = "source_not_found"
    POINTER_INVALID = "pointer_invalid"
    HANDLE_UNKNOWN = "handle_unknown"
    SUFFIX_NOT_OWNED = "suffix_not_owned"
    NO_READER_FOR_SUFFIX = "no_reader_for_suffix"
    HANDED_OVER = "handed_over"
    FOLDER_ESCAPE = "folder_escape"
    SOURCE_UNREADABLE = "source_unreadable"
    QUERY_INVALID = "query_invalid"
    QUERY_ROUTE_TAKEN = "query_route_taken"
    CREDENTIALS_INVALID = "credentials_invalid"


@dataclass(frozen=True)
class MatchRejection:
    """One recorded rejection: the reason plus a free-form stage hint the engine maps at the harvest.
    `candidates` names the sources an ambiguity abort found, else empty.

    Unknown stage values fall back to value_rejection there; this neutral module does not validate them.
    """

    reason: str
    stage: str = "value_rejection"
    candidates: tuple[str, ...] = ()
    refusal: Exception | None = field(default=None, compare=False, repr=False)
    code: DeclineCode | None = None


INPUT_DATA_STAGE = "input_data"

# The candidate refused the feature name itself, not an option value.
NAME_STAGE = "name"


# Active for one candidate's match call: the engine opens a window per candidate. Maps the recording
# site's owner name to the first structured rejection the real match pass produced, and the
# engine attributes the harvest to the candidate class object it called.
MATCH_REJECTION_REASONS: contextvars.ContextVar[dict[str, MatchRejection] | None] = contextvars.ContextVar(
    "mloda_match_rejection_reasons", default=None
)


def record_match_rejection(
    owner_name: str,
    reason: str,
    stage: str = "value_rejection",
    candidates: tuple[str, ...] = (),
    refusal: Exception | None = None,
    code: DeclineCode | None = None,
) -> None:
    """Record a match rejection; the first per owner wins, and outside an active evaluation it is a no-op."""
    reasons = MATCH_REJECTION_REASONS.get()
    if reasons is None:
        return
    reasons.setdefault(owner_name, MatchRejection(reason, stage, candidates, refusal, code))


@contextmanager
def collect_match_rejections() -> Iterator[dict[str, MatchRejection]]:
    """Open one rejection window for one direct match call, as the engine does per candidate, and yield its dict."""
    reasons: dict[str, MatchRejection] = {}
    token = MATCH_REJECTION_REASONS.set(reasons)
    try:
        yield reasons
    finally:
        MATCH_REJECTION_REASONS.reset(token)


def replace_probe_match_rejection(
    owner_name: str,
    reason: str,
    known_owners: frozenset[str],
    stage: str = "value_rejection",
    code: DeclineCode | None = None,
) -> None:
    """Record, replacing the owner's rejection only when it was recorded after the known_owners snapshot."""
    reasons = MATCH_REJECTION_REASONS.get()
    if reasons is None:
        return
    if owner_name in known_owners:
        reasons.setdefault(owner_name, MatchRejection(reason, stage, code=code))
    else:
        reasons[owner_name] = MatchRejection(reason, stage, code=code)


def match_rejection_owners() -> frozenset[str]:
    """Owner names already recorded in the active window; empty outside one."""
    reasons = MATCH_REJECTION_REASONS.get()
    if reasons is None:
        return frozenset()
    return frozenset(reasons)


def drop_match_rejections_since(known_owners: frozenset[str]) -> None:
    """Drop every owner recorded after the snapshot; no-op outside a window."""
    reasons = MATCH_REJECTION_REASONS.get()
    if reasons is None:
        return
    for owner_name in list(reasons):
        if owner_name not in known_owners:
            del reasons[owner_name]


def context_forwarding_remedy(context: bool, keys: list[str] | None = None) -> str:
    """Remedy clause for an absent-option reason; empty for context=False keys. `keys` names them instead of "it"."""
    if not context:
        return ""
    subject, pronoun = "it", "it"
    if keys:
        subject = ", ".join(keys)
        pronoun = "them" if len(keys) > 1 else "it"
    return (
        f"; pass {subject} in Options(context=...); for a direct input of a feature you construct, "
        f"list {pronoun} in that feature's propagate_context_keys (the push is one hop); "
        f"for an input further down a chained name, list {pronoun} in the requested feature's "
        f"cascade_context_keys, or pass {pronoun} in Options(group=...) to make {pronoun} part of feature identity"
    )
