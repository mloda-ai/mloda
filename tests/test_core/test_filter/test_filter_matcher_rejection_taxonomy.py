"""Issue #728: GlobalFilter.criteria shares the canonical criteria probe, so a PropertyValueRejection or a
recorded match rejection becomes a typed DEBUG drop carrying the same reason text on both seams, while a plain
matcher error keeps its WARNING. Probe classes live inside factory functions and are dropped before any assert
runs, so a failing assert never pins a throwaway FeatureGroup and trips the no-leak fixture in tests/conftest.py.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable
from dataclasses import dataclass, is_dataclass
from functools import partial
from typing import Any, ClassVar, TypeVar

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import PropertyValueRejection
from mloda.core.abstract_plugins.components.match_rejection import (
    INPUT_DATA_OWNED_STAGE,
    INPUT_DATA_STAGE,
    MATCH_REJECTION_REASONS,
    MatchRejection,
    match_rejection_owners,
    record_match_rejection,
)
from mloda.core.abstract_plugins.components.utils import escalate_match_abort
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.core.prepare.resolution_types import EvaluationResult
from mloda.user import Feature, FeatureName, FilterType, GlobalFilter, Options, SingleFilter
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


GF_LOGGER_NAME = "mloda.core.filter.global_filter"

HOST_FEATURE = "rtx_host_feat_728"  # the resolved feature the filters are matched against
FILTER_FEATURE = "rtx_filter_feat_728"  # the declared filter feature every probe is asked about

VALUE_REJECTION_CLASS_NAME = "RtxValueRejectionFG728"
RECORDED_DECLINE_CLASS_NAME = "RtxRecordedDeclineFG728"
RECORD_THEN_VALUE_CLASS_NAME = "RtxRecordThenValueRaiseFG728"
STAGE_DECLINE_CLASS_NAME = "RtxStageDeclineFG728"
RECORD_THEN_ERROR_CLASS_NAME = "RtxRecordThenErrorFG728"
OWNED_VETO_CLASS_NAME = "RtxOwnedVetoFG728"

VALUE_REJECT_MESSAGE = "rtx_value_rejected_728"
REASON_A = "rtx_reason_a_728"
REASON_B = "rtx_reason_b_728"
RUNTIME_MESSAGE = "rtx_runtime_boom_728"
RUNTIME_TYPE_NAME = "RuntimeError"
ESCALATE_MESSAGE = "rtx_escalated_728"
OWNED_REASON = "rtx_owned_veto_reason_728"
VALUE_REJECTION_STAGE = "value_rejection"
VALUE_REJECTION_TYPE_NAME = "PropertyValueRejection"

T = TypeVar("T")

# A factory handing back the throwaway class and a reader for its window observation, so no drive types the class.
_RtxFactory = Callable[[], tuple[type[FeatureGroup], Callable[[], bool]]]


def _capture(call: Callable[[], T]) -> tuple[T | None, str | None]:
    """Run call, returning (value, None) or (None, 'Type: message'). No traceback is retained."""
    try:
        return call(), None
    except Exception as exc:  # noqa: BLE001  (red-phase probe: an escape is the fact under test)
        return None, f"{type(exc).__name__}: {exc}"


def _exception_text(exc: BaseException | None) -> str | None:
    """Type and message of a contained exception, or None. Reads no traceback."""
    return None if exc is None else f"{type(exc).__name__}: {exc}"


def _single(filter_feature_name: str) -> SingleFilter:
    """A minimal EQUAL filter on one feature name."""
    return SingleFilter(filter_feature_name, FilterType.EQUAL, {"value": 1})


def _messages(caplog: pytest.LogCaptureFixture, level: int) -> tuple[str, ...]:
    """Formatted messages GlobalFilter logged at exactly that level."""
    records = [record for record in caplog.records if record.name == GF_LOGGER_NAME and record.levelno == level]
    return tuple(record.getMessage() for record in records)


def _stage_reason(stage: str) -> str:
    """The stage-specific reason text one stage-recording probe stores."""
    return f"rtx_stage_reason_728_{stage}"


def _window_not_observed() -> bool:
    """Reader for probes that never record: their tests assert nothing about the window."""
    return False


def _make_value_rejection_fg() -> tuple[type[FeatureGroup], Callable[[], bool]]:
    """A throwaway group whose hook raises PropertyValueRejection for the filter feature."""
    # Class objects are cyclic; collect leftovers from earlier tests before defining a twin.
    gc.collect()

    class RtxValueRejectionFG728(FeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {HOST_FEATURE}

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: DataAccessCollection | None = None,
        ) -> bool:
            if str(feature_name) == FILTER_FEATURE:
                raise PropertyValueRejection(VALUE_REJECT_MESSAGE)
            return str(feature_name) in cls.feature_names_supported()

    return RtxValueRejectionFG728, _window_not_observed


def _make_recorded_decline_fg() -> tuple[type[FeatureGroup], Callable[[], bool]]:
    """A throwaway group whose hook records a rejection for the filter feature and returns False."""
    gc.collect()

    class RtxRecordedDeclineFG728(FeatureGroup):
        window_active: ClassVar[bool] = False

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {HOST_FEATURE}

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: DataAccessCollection | None = None,
        ) -> bool:
            if str(feature_name) == FILTER_FEATURE:
                record_match_rejection(cls.__name__, REASON_A)
                cls.window_active = cls.__name__ in match_rejection_owners()
                return False
            return str(feature_name) in cls.feature_names_supported()

    return RtxRecordedDeclineFG728, lambda: RtxRecordedDeclineFG728.window_active


def _make_record_then_value_raise_fg() -> tuple[type[FeatureGroup], Callable[[], bool]]:
    """A throwaway group whose hook records reason A, then raises PropertyValueRejection(B)."""
    gc.collect()

    class RtxRecordThenValueRaiseFG728(FeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {HOST_FEATURE}

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: DataAccessCollection | None = None,
        ) -> bool:
            if str(feature_name) == FILTER_FEATURE:
                record_match_rejection(cls.__name__, REASON_A)
                raise PropertyValueRejection(REASON_B)
            return str(feature_name) in cls.feature_names_supported()

    return RtxRecordThenValueRaiseFG728, _window_not_observed


def _make_stage_decline_fg(stage: str) -> tuple[type[FeatureGroup], Callable[[], bool]]:
    """A throwaway group whose hook records with the caller's stage for the filter feature and returns False."""
    gc.collect()

    class RtxStageDeclineFG728(FeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {HOST_FEATURE}

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: DataAccessCollection | None = None,
        ) -> bool:
            if str(feature_name) == FILTER_FEATURE:
                record_match_rejection(cls.__name__, _stage_reason(stage), stage=stage)
                return False
            return str(feature_name) in cls.feature_names_supported()

    return RtxStageDeclineFG728, _window_not_observed


def _make_record_then_error_fg() -> tuple[type[FeatureGroup], Callable[[], bool]]:
    """A throwaway group whose hook records reason A, then raises a plain RuntimeError."""
    gc.collect()

    class RtxRecordThenErrorFG728(FeatureGroup):
        window_active: ClassVar[bool] = False

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {HOST_FEATURE}

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: DataAccessCollection | None = None,
        ) -> bool:
            if str(feature_name) == FILTER_FEATURE:
                record_match_rejection(cls.__name__, REASON_A)
                cls.window_active = cls.__name__ in match_rejection_owners()
                raise RuntimeError(RUNTIME_MESSAGE)
            return str(feature_name) in cls.feature_names_supported()

    return RtxRecordThenErrorFG728, lambda: RtxRecordThenErrorFG728.window_active


def _make_owned_veto_fg() -> tuple[type[FeatureGroup], Callable[[], bool]]:
    """A throwaway group supporting the filter name whose hook records an owned veto, then defers to super."""
    gc.collect()

    class RtxOwnedVetoFG728(FeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {HOST_FEATURE, FILTER_FEATURE}

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: DataAccessCollection | None = None,
        ) -> bool:
            record_match_rejection(cls.__name__, OWNED_REASON, stage=INPUT_DATA_OWNED_STAGE)
            return super().match_feature_group_criteria(feature_name, options, data_access_collection)

    return RtxOwnedVetoFG728, _window_not_observed


def _make_record_but_match_fg() -> tuple[type[FeatureGroup], Callable[[], bool]]:
    """A throwaway group whose hook records a rejection for the filter feature and still returns True."""
    gc.collect()

    class RtxRecordButMatchFG728(FeatureGroup):
        window_active: ClassVar[bool] = False

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {HOST_FEATURE}

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: DataAccessCollection | None = None,
        ) -> bool:
            if str(feature_name) == FILTER_FEATURE:
                record_match_rejection(cls.__name__, REASON_A)
                cls.window_active = cls.__name__ in match_rejection_owners()
                return True
            return str(feature_name) in cls.feature_names_supported()

    return RtxRecordButMatchFG728, lambda: RtxRecordButMatchFG728.window_active


def _make_escalating_rejection_fg(marker: BaseException) -> tuple[type[FeatureGroup], Callable[[], bool]]:
    """A throwaway group whose hook records reason A, then raises the caller's own marked exception object."""
    gc.collect()

    class RtxEscalatingRejectionFG728(FeatureGroup):
        window_active: ClassVar[bool] = False

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {HOST_FEATURE}

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: DataAccessCollection | None = None,
        ) -> bool:
            record_match_rejection(cls.__name__, REASON_A)
            cls.window_active = cls.__name__ in match_rejection_owners()
            raise marker

    return RtxEscalatingRejectionFG728, lambda: RtxEscalatingRejectionFG728.window_active


def _make_plain_error_fg() -> tuple[type[FeatureGroup], Callable[[], bool]]:
    """A throwaway group whose hook raises a plain RuntimeError for the filter feature, recording nothing."""
    gc.collect()

    class RtxPlainErrorFG728(FeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {HOST_FEATURE}

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: DataAccessCollection | None = None,
        ) -> bool:
            if str(feature_name) == FILTER_FEATURE:
                raise RuntimeError(RUNTIME_MESSAGE)
            return str(feature_name) in cls.feature_names_supported()

    return RtxPlainErrorFG728, _window_not_observed


@dataclass(frozen=True)
class _RtxCriteriaSnapshot:
    """Plain-data readout of one criteria call. Holds no class and no exception object."""

    is_false: bool
    is_true: bool
    escaped: str | None
    entries: tuple[tuple[str, str], ...]  # (feature group class name, filter feature name), sorted
    reasons: tuple[str, ...]  # reason text per entry, aligned with entries
    warnings: tuple[str, ...]
    debugs: tuple[str, ...]
    window_active: bool


def _drive_criteria(make: _RtxFactory, caplog: pytest.LogCaptureFixture) -> _RtxCriteriaSnapshot:
    """Call criteria once on a fresh GlobalFilter; the finally unbinds every name that pins the class."""
    caplog.clear()
    fg, read_window = make()
    global_filter = GlobalFilter()
    items: list[tuple[Any, Any]] = []
    try:
        with caplog.at_level(logging.DEBUG, logger=GF_LOGGER_NAME):
            value, escaped = _capture(partial(global_filter.criteria, fg, _single(FILTER_FEATURE), None))
        items = sorted(global_filter.dropped_filters.items(), key=lambda item: str(item[0]))
        return _RtxCriteriaSnapshot(
            is_false=value is False,
            is_true=value is True,
            escaped=escaped,
            entries=tuple((str(key[0].get_class_name()), str(key[1])) for key, _ in items),
            reasons=tuple(str(reason) for _, reason in items),
            warnings=_messages(caplog, logging.WARNING),
            debugs=_messages(caplog, logging.DEBUG),
            window_active=read_window(),
        )
    finally:
        del fg, read_window, global_filter, items
        gc.collect()


@dataclass(frozen=True)
class _CanonicalSnapshot:
    """Plain-data readout of one evaluate() pass. Holds no class and no Elimination object."""

    escaped: str | None
    identified: tuple[str, ...]
    eliminations: tuple[tuple[str, str, str], ...]


def _canonical_snapshot(result: EvaluationResult | None, escaped: str | None) -> _CanonicalSnapshot:
    """Fold one evaluate() outcome to plain data."""
    if result is None:
        return _CanonicalSnapshot(escaped=escaped, identified=(), eliminations=())
    return _CanonicalSnapshot(
        escaped=escaped,
        identified=tuple(sorted(g.get_class_name() for g in result.identified)),
        eliminations=tuple(
            sorted((g.get_class_name(), str(e.stage), str(e.reason)) for g, e in result.eliminations.items())
        ),
    )


def _drive_canonical(make: _RtxFactory) -> _CanonicalSnapshot:
    """Evaluate the filter feature against the probe alone and fold the eliminations to plain tuples."""
    fg, read_window = make()
    plugins: FeatureGroupEnvironmentMapping = {fg: {PythonDictFramework}}
    result = None
    try:
        result, escaped = _capture(partial(IdentifyFeatureGroupClass.evaluate, Feature(FILTER_FEATURE), plugins, None))
        snapshot = _canonical_snapshot(result, escaped)
        del result
        result = None
        return snapshot
    finally:
        del fg, read_window, plugins, result
        gc.collect()


class TestValueRejectionIsATypedDrop:
    """A PropertyValueRejection out of the matcher is a typed verdict drop, not a contained defect."""

    def test_criteria_contains_it_and_stores_exactly_the_rejection_text(self, caplog: pytest.LogCaptureFixture) -> None:
        """The drop holds str(exc) itself, not the 'raised PropertyValueRejection: ...' defect wrapper."""
        snapshot = _drive_criteria(_make_value_rejection_fg, caplog)

        assert snapshot.escaped is None, f"the raise must not cross GlobalFilter.criteria: {snapshot.escaped}"
        assert snapshot.is_false, "a rejected value is a non-match for that filter"
        assert snapshot.entries == ((VALUE_REJECTION_CLASS_NAME, FILTER_FEATURE),), (
            f"exactly one drop, keyed by group and filter feature, got: {snapshot.entries}"
        )
        assert snapshot.reasons == (VALUE_REJECT_MESSAGE,), (
            f"the drop must hold exactly str(exc), no 'raised' prefix, got: {snapshot.reasons}"
        )

    def test_the_drop_logs_debug_and_never_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """A value rejection is the candidate's verdict, so it reports at DEBUG like the resolution seam."""
        snapshot = _drive_criteria(_make_value_rejection_fg, caplog)

        assert snapshot.escaped is None, f"the raise must not cross GlobalFilter.criteria: {snapshot.escaped}"
        assert snapshot.warnings == (), f"a value rejection is a verdict, not a defect, got: {snapshot.warnings}"
        assert len(snapshot.debugs) == 1, f"the drop must still be visible at DEBUG, got: {snapshot.debugs}"

    def test_the_canonical_seam_reports_the_same_reason(self, caplog: pytest.LogCaptureFixture) -> None:
        """Reason parity: one raise message, one reason text, on both seams."""
        filter_side = _drive_criteria(_make_value_rejection_fg, caplog)
        canonical = _drive_canonical(_make_value_rejection_fg)

        assert canonical.escaped is None, f"nothing may cross evaluate: {canonical.escaped}"
        assert canonical.identified == (), f"a rejected value must win nothing, got: {canonical.identified}"
        assert len(canonical.eliminations) == 1, f"exactly one elimination, got: {canonical.eliminations}"
        name, stage, reason = canonical.eliminations[0]
        assert name == VALUE_REJECTION_CLASS_NAME, f"the elimination must name the candidate, got: {name}"
        assert stage == VALUE_REJECTION_STAGE, f"a value rejection owns the stage, got: {stage}"
        assert filter_side.reasons == (reason,), (
            f"both seams must store one reason text for one raise, got {filter_side.reasons} vs {reason!r}"
        )


class TestRecordedRejectionIsATypedDrop:
    """A rejection recorded through the shared window lands in dropped_filters as its own reason."""

    def test_a_recorded_decline_lands_in_dropped_filters(self, caplog: pytest.LogCaptureFixture) -> None:
        """record_match_rejection plus a False return is a typed drop, reported at DEBUG."""
        snapshot = _drive_criteria(_make_recorded_decline_fg, caplog)

        assert snapshot.escaped is None, f"nothing may cross GlobalFilter.criteria: {snapshot.escaped}"
        assert snapshot.is_false, "a recorded decline is a non-match for that filter"
        assert snapshot.entries == ((RECORDED_DECLINE_CLASS_NAME, FILTER_FEATURE),), (
            f"a recorded decline is a typed drop, got: {snapshot.entries}"
        )
        assert snapshot.reasons == (REASON_A,), f"the drop must hold the recorded reason, got: {snapshot.reasons}"
        assert snapshot.warnings == (), f"a recorded decline is a verdict, not a defect, got: {snapshot.warnings}"
        assert len(snapshot.debugs) == 1, f"the drop must still be visible at DEBUG, got: {snapshot.debugs}"

    @pytest.mark.parametrize("stage", [INPUT_DATA_STAGE, INPUT_DATA_OWNED_STAGE])
    def test_both_input_data_stages_flow_through(self, stage: str, caplog: pytest.LogCaptureFixture) -> None:
        """A rejection recorded under either input-data stage still surfaces as a typed drop."""
        snapshot = _drive_criteria(partial(_make_stage_decline_fg, stage), caplog)

        assert snapshot.escaped is None, f"nothing may cross GlobalFilter.criteria: {snapshot.escaped}"
        assert snapshot.is_false, "a recorded decline is a non-match for that filter"
        assert snapshot.entries == ((STAGE_DECLINE_CLASS_NAME, FILTER_FEATURE),), (
            f"a stage-recorded decline is a typed drop, got: {snapshot.entries}"
        )
        assert snapshot.reasons == (_stage_reason(stage),), (
            f"the drop must hold the stage-recorded reason, got: {snapshot.reasons}"
        )


class TestHarvestPrecedence:
    """The first recorded reason wins over a later value rejection; a matcher error outranks both."""

    def test_the_first_recorded_reason_outranks_a_later_value_rejection(self, caplog: pytest.LogCaptureFixture) -> None:
        snapshot = _drive_criteria(_make_record_then_value_raise_fg, caplog)

        assert snapshot.escaped is None, f"the raise must not cross GlobalFilter.criteria: {snapshot.escaped}"
        assert snapshot.is_false, "the rejection is still a non-match for that filter"
        assert snapshot.entries == ((RECORD_THEN_VALUE_CLASS_NAME, FILTER_FEATURE),), (
            f"exactly one drop, keyed by group and filter feature, got: {snapshot.entries}"
        )
        assert snapshot.reasons == (REASON_A,), (
            f"the FIRST recorded reason wins the drop, not the raise and not a wrapper, got: {snapshot.reasons}"
        )

    def test_a_matcher_error_outranks_the_recorded_reason_and_still_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A plain raise is the candidate's own defect: contained-raise text, WARNING policy unchanged."""
        snapshot = _drive_criteria(_make_record_then_error_fg, caplog)

        assert snapshot.escaped is None, f"the raise must not cross GlobalFilter.criteria: {snapshot.escaped}"
        assert snapshot.is_false, "a raising hook is a non-match for that filter"
        assert snapshot.window_active, "the seam must give the matcher an active rejection window"
        assert snapshot.entries == ((RECORD_THEN_ERROR_CLASS_NAME, FILTER_FEATURE),), (
            f"exactly one drop, keyed by group and filter feature, got: {snapshot.entries}"
        )
        reason = snapshot.reasons[0]
        assert RUNTIME_TYPE_NAME in reason, f"the reason must name the exception type: {reason}"
        assert RUNTIME_MESSAGE in reason, f"the reason must carry the raise message: {reason}"
        assert REASON_A not in reason, f"the contained raise outranks the recorded reason: {reason}"
        assert len(snapshot.warnings) == 1, f"a matcher defect must still warn once, got: {snapshot.warnings}"


class TestMatchAndEscalationLeaveNoDrop:
    """A match and a marked abort both leave the ledger empty, recorded rejection or not."""

    def test_a_recording_matcher_that_matches_attaches_without_a_drop(self, caplog: pytest.LogCaptureFixture) -> None:
        """Harvest only on non-match: a recorded reason must not detach a filter the matcher accepted."""
        snapshot = _drive_criteria(_make_record_but_match_fg, caplog)

        assert snapshot.escaped is None, f"nothing may cross GlobalFilter.criteria: {snapshot.escaped}"
        assert snapshot.window_active, "the seam must give the matcher an active rejection window"
        assert snapshot.is_true, "a True return is a match, whatever was recorded before it"
        assert snapshot.entries == (), f"a match is never a drop, got: {snapshot.entries}"
        assert snapshot.warnings == (), f"a match must not warn, got: {snapshot.warnings}"

    def test_an_escalated_value_rejection_still_escapes_and_records_no_drop(self) -> None:
        """An escalate_match_abort-marked raise crosses the seam as the SAME object and is not a drop."""
        marker = escalate_match_abort(PropertyValueRejection(ESCALATE_MESSAGE))
        fg, read_window = _make_escalating_rejection_fg(marker)
        global_filter = GlobalFilter()
        caught: BaseException | None = None
        try:
            global_filter.criteria(fg, _single(FILTER_FEATURE), None)
        except BaseException as exc:  # noqa: BLE001  (the escape itself is the fact under test)
            caught = exc
        is_marker = caught is marker
        type_name = None if caught is None else type(caught).__name__
        message = None if caught is None else str(caught)
        entry_count = len(global_filter.dropped_filters)
        window_active = read_window()
        # Drop the retained traceback: its frames pin the throwaway class through the hook's `cls`.
        marker.__traceback__ = None
        del fg, read_window, global_filter, caught
        gc.collect()

        assert is_marker, f"the marked exception itself must escape, got: {type_name}: {message}"
        assert type_name == VALUE_REJECTION_TYPE_NAME
        assert message == ESCALATE_MESSAGE
        assert entry_count == 0, f"a propagating abort is not a drop, got {entry_count} entries"
        assert window_active, "the seam must give the matcher an active rejection window"


class TestOwnedVetoParity:
    """The default hook's owned-veto gate sees the filter seam's active window."""

    def test_the_default_hooks_owned_veto_gate_sees_the_filter_seams_window(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An owned-stage recording before super() must veto a name the group otherwise supports."""
        snapshot = _drive_criteria(_make_owned_veto_fg, caplog)

        assert snapshot.escaped is None, f"nothing may cross GlobalFilter.criteria: {snapshot.escaped}"
        assert snapshot.is_false, "the owned veto must gate the default hook's name rules"
        assert snapshot.entries == ((OWNED_VETO_CLASS_NAME, FILTER_FEATURE),), (
            f"the veto is a typed drop, keyed by group and filter feature, got: {snapshot.entries}"
        )
        assert snapshot.reasons == (OWNED_REASON,), f"the drop must hold the veto's reason, got: {snapshot.reasons}"
        assert snapshot.warnings == (), f"an owned veto is a verdict, not a defect, got: {snapshot.warnings}"


@dataclass(frozen=True)
class _ProbeSnapshot:
    """Plain-data readout of one probe_match_criteria call. Holds no class and no exception object."""

    matched: bool
    matcher_error: str | None  # 'Type: message' of the contained plain raise, or None
    value_rejection: str | None  # 'Type: message' of the contained PropertyValueRejection, or None
    has_rejection: bool  # outcome.rejection is a MatchRejection
    rejection_reason: str | None
    rejection_stage: str | None
    window_after_is_none: bool


def _drive_probe(make: _RtxFactory) -> _ProbeSnapshot:
    """Probe one throwaway matcher; the import lives here so a missing probe fails only these tests."""
    from mloda.core.abstract_plugins.components.match_hook import probe_match_criteria

    fg, read_window = make()
    try:
        outcome = probe_match_criteria(fg, FILTER_FEATURE, Options(), None)
        snapshot = _ProbeSnapshot(
            matched=outcome.matched,
            matcher_error=_exception_text(outcome.matcher_error),
            value_rejection=_exception_text(outcome.value_rejection),
            has_rejection=isinstance(outcome.rejection, MatchRejection),
            rejection_reason=None if outcome.rejection is None else str(outcome.rejection.reason),
            rejection_stage=None if outcome.rejection is None else str(outcome.rejection.stage),
            window_after_is_none=MATCH_REJECTION_REASONS.get() is None,
        )
        del outcome
        return snapshot
    finally:
        del fg, read_window
        gc.collect()


class TestCriteriaProbeOutcome:
    """The shared probe as a unit: one call, one structured outcome, window reset on exit."""

    def test_the_shared_probe_api_exists(self) -> None:
        """Both seams will import the probe and its frozen outcome type from match_hook."""
        from mloda.core.abstract_plugins.components.match_hook import CriteriaProbeOutcome, probe_match_criteria

        assert callable(probe_match_criteria), "probe_match_criteria must be the shared callable probe"
        assert is_dataclass(CriteriaProbeOutcome), "CriteriaProbeOutcome must be a dataclass"

    def test_a_plain_matcher_error_is_probed_as_matcher_error(self) -> None:
        """A plain raise with no recording: matcher_error set, no value rejection, no match."""
        snapshot = _drive_probe(_make_plain_error_fg)

        assert snapshot.matched is False, "a raising matcher is a non-match"
        assert snapshot.value_rejection is None, f"a plain raise is no value rejection: {snapshot.value_rejection}"
        assert snapshot.matcher_error == f"{RUNTIME_TYPE_NAME}: {RUNTIME_MESSAGE}", (
            f"the contained raise must come back as matcher_error, got: {snapshot.matcher_error}"
        )

    def test_a_value_rejection_is_probed_with_a_harvested_rejection(self) -> None:
        """A PropertyValueRejection raise: value_rejection set and harvested as a MatchRejection."""
        snapshot = _drive_probe(_make_value_rejection_fg)

        assert snapshot.matched is False, "a rejecting matcher is a non-match"
        assert snapshot.matcher_error is None, f"a value rejection is no matcher defect: {snapshot.matcher_error}"
        assert snapshot.value_rejection == f"{VALUE_REJECTION_TYPE_NAME}: {VALUE_REJECT_MESSAGE}", (
            f"the contained rejection must come back as value_rejection, got: {snapshot.value_rejection}"
        )
        assert snapshot.has_rejection, "the raise must be harvested into outcome.rejection"
        assert snapshot.rejection_reason == VALUE_REJECT_MESSAGE, (
            f"the harvested reason must be str(exc), got: {snapshot.rejection_reason}"
        )
        assert snapshot.rejection_stage == VALUE_REJECTION_STAGE, (
            f"a value rejection owns the harvested stage, got: {snapshot.rejection_stage}"
        )

    def test_an_owned_stage_recording_keeps_its_stage(self) -> None:
        snapshot = _drive_probe(partial(_make_stage_decline_fg, INPUT_DATA_OWNED_STAGE))

        assert snapshot.matched is False, "a declining matcher is a non-match"
        assert snapshot.has_rejection, "the recorded rejection must be harvested on a non-match"
        assert snapshot.rejection_stage == INPUT_DATA_OWNED_STAGE, (
            f"the recorded stage must survive the harvest, got: {snapshot.rejection_stage}"
        )

    def test_a_recording_match_harvests_nothing(self) -> None:
        """Harvest only on non-match: a matching probe hands back no rejection."""
        snapshot = _drive_probe(_make_record_but_match_fg)

        assert snapshot.matched is True, "a True return is a match"
        assert not snapshot.has_rejection, f"a match harvests nothing, got: {snapshot.rejection_reason}"
        assert snapshot.rejection_reason is None
        assert snapshot.rejection_stage is None

    def test_the_window_is_reset_after_the_probe(self) -> None:
        snapshot = _drive_probe(_make_recorded_decline_fg)

        assert snapshot.matched is False, "a declining matcher is a non-match"
        assert snapshot.window_after_is_none, "the probe must reset MATCH_REJECTION_REASONS on exit"
