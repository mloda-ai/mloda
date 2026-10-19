"""One home for the match-hook call and the shared criteria probe: the containment, the marked-abort
re-raise and the return coercion.

Both match seams held their own try around the plugin match hook and drifted apart twice (#991).
Each keeps only its own recording and option isolation now, driven by the outcome this helper hands back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import (
    ANY_OUTPUT_PART,
    COLUMN_SEPARATOR,
    PropertyValueRejection,
)
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.input_data.base_input_data import RESERVED_READER_OPTION_KEY
from mloda.core.abstract_plugins.components.match_rejection import (
    MATCH_REJECTION_REASONS,
    MatchRejection,
    record_match_rejection,
)
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.utils import is_match_abort, safe_exc_str, safe_field

if TYPE_CHECKING:
    from mloda.core.abstract_plugins.feature_group import FeatureGroup


@dataclass(frozen=True)
class MatchHookOutcome:
    """One match-hook call: the coerced verdict, the raw return, and the raise that was contained."""

    matched: bool
    returned: Any
    error: Exception | None
    # (base, part) when matched as a declared output part.
    part: tuple[str, str] | None = None


def _in_window(call: Callable[[], Any], sink: list[dict[str, MatchRejection]]) -> Any:
    """Run a call under its own rejection window, appended to sink even on a raise."""
    window: dict[str, MatchRejection] = {}
    token = MATCH_REJECTION_REASONS.set(window)
    try:
        return call()
    finally:
        MATCH_REJECTION_REASONS.reset(token)
        sink.append(window)


def _merge_windows(priority: list[dict[str, MatchRejection]], windows: list[dict[str, MatchRejection]]) -> None:
    """Rebuild the outer window: priority reasons, prior ones, then the latest calls first."""
    outer = MATCH_REJECTION_REASONS.get()
    if outer is None:
        return
    prior = dict(outer)
    outer.clear()
    for window in [*priority, prior, *reversed(windows)]:
        for owner, rejection in window.items():
            outer.setdefault(owner, rejection)


def _ask_hook(
    feature_group: type[FeatureGroup],
    feature_name: FeatureName | str,
    options: Options,
    data_access_collection: DataAccessCollection | None,
) -> tuple[bool, Any]:
    """One call of the plugin's match hook and its bool() coercion (#927)."""
    returned = feature_group.match_feature_group_criteria(feature_name, options, data_access_collection)
    return bool(returned), returned


def call_match_hook(
    feature_group: type[FeatureGroup],
    feature_name: FeatureName | str,
    options: Options,
    data_access_collection: DataAccessCollection | None = None,
    resolve_parts: bool = False,
) -> MatchHookOutcome:
    """Ask one candidate's match hook: a raise out of it is a non-match for that candidate only (#845).

    Policy: mark a raise with escalate_match_abort when it reports a misconfiguration, a contradiction or a
    framework defect; leave it contained when it is one candidate's own judgment or own defect. Every raise
    the match path reaches must say which, at the raise: tests/test_core/test_prepare/test_match_abort_sweep.py
    enforces that. A mark only survives if every handler in between re-raises it (``safe_field`` does not). An
    option-write conflict during reader selection escalates as a contradiction, decided at the reader-selection
    raise.

    The contained exception is handed back rather than judged here: each seam records and isolates options its own way.
    resolve_parts (feature resolution only) reads `<base>~<part>` against the declared parts.
    """
    windows: list[dict[str, MatchRejection]] = []
    priority: list[dict[str, MatchRejection]] = []
    try:
        split = feature_group.split_output_part(str(feature_name)) if resolve_parts else None
        if split is None:
            matched, returned = _ask_hook(feature_group, feature_name, options, data_access_collection)
            return MatchHookOutcome(matched, returned, None)
        base, part = split
        base_options = options._fork()
        matched, returned = _in_window(
            lambda: _ask_hook(feature_group, feature_name, options, data_access_collection), windows
        )
        full = MatchHookOutcome(matched, returned, None)
        if matched and feature_group.get_column_base_feature(str(feature_name)) != base:
            return full
        declared = feature_group.output_parts(base, options)
        if declared is None:
            return full
        if not matched:
            base_matched, _ = _in_window(
                lambda: _ask_hook(feature_group, base, base_options, data_access_collection), windows
            )
            if not base_matched:
                return full
        if ANY_OUTPUT_PART in declared or part.split(COLUMN_SEPARATOR)[0] in declared:
            if not matched:
                options._adopt(base_options)
            return MatchHookOutcome(True, True, None, split)
        reason = f"'{part}' is not an output part of '{base}'; declared parts: {list(declared)}"
        priority.append({feature_group.get_class_name(): MatchRejection(reason)})
        return MatchHookOutcome(False, False, None)
    except Exception as exc:  # noqa: BLE001  (contained: one broken matcher must not poison the whole run)
        if is_match_abort(exc):
            raise
        # The outcome outlives this except block, and the traceback's frames pin the plugin class and its raw
        # return. with_traceback, not the attribute: a frozen-dataclass exception rejects the assignment.
        return MatchHookOutcome(False, None, exc.with_traceback(None))
    finally:
        _merge_windows(priority, windows)


@dataclass(frozen=True)
class CriteriaProbeOutcome:
    """One probed match: verdict, raw return, the contained raise split by kind, the first harvested rejection."""

    matched: bool
    returned: Any
    # Disjoint: the one contained raise is a typed decline (value_rejection) or the candidate's own defect.
    matcher_error: Exception | None
    value_rejection: PropertyValueRejection | None
    rejection: MatchRejection | None
    # The (group, access) pair a matching probe wrote under the reserved key; None otherwise.
    written: tuple[Any, Any] | None
    part: tuple[str, str] | None = None


def _call_and_take_reader_pair(
    feature_group: type[FeatureGroup],
    feature_name: FeatureName | str,
    options: Options,
    data_access_collection: DataAccessCollection | None,
    resolve_parts: bool,
) -> tuple[MatchHookOutcome, Any]:
    """Call the hook, then pop the reserved reader key whatever the outcome, a marked abort included."""
    try:
        outcome = call_match_hook(feature_group, feature_name, options, data_access_collection, resolve_parts)
    finally:
        popped = options.group.pop(RESERVED_READER_OPTION_KEY, None)
    return outcome, popped


def probe_match_criteria(
    feature_group: type[FeatureGroup],
    feature_name: FeatureName | str,
    options: Options,
    data_access_collection: DataAccessCollection | None = None,
    resolve_parts: bool = False,
) -> CriteriaProbeOutcome:
    """Ask one candidate under its own rejection window; a marked abort propagates, the finally only resets."""
    token = MATCH_REJECTION_REASONS.set({})
    try:
        outcome, popped = _call_and_take_reader_pair(
            feature_group, feature_name, options, data_access_collection, resolve_parts
        )
        value_rejection = outcome.error if isinstance(outcome.error, PropertyValueRejection) else None
        matcher_error = outcome.error if value_rejection is None else None
        if value_rejection is not None:
            # Parity with the canonical seam: the class name owns the record and the reason is str(exc);
            # the owner name is harvested by value, never by key, so a degraded name is harmless.
            owner = safe_field(lambda: feature_group.get_class_name(), "<unnamed feature group>")
            reason = safe_exc_str(value_rejection)
            record_match_rejection(owner, reason)
        recorded = MATCH_REJECTION_REASONS.get() or {}
        # Harvest only on a non-match; the first recorded reason wins (insertion order).
        rejection = next(iter(recorded.values())) if not outcome.matched and recorded else None
        written = popped if outcome.matched and isinstance(popped, tuple) and len(popped) == 2 else None
        return CriteriaProbeOutcome(
            outcome.matched, outcome.returned, matcher_error, value_rejection, rejection, written, outcome.part
        )
    finally:
        MATCH_REJECTION_REASONS.reset(token)
