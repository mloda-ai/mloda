"""Issue #927: both match seams read the hook's return by truthiness and answer with a real bool.

Before the fix, ``identify_matched_filters`` gated ``GlobalFilter.criteria`` on ``is False``, so a hook
returning None attached the filter while the resolution seam called that same None a non-match. Pinned here:
a falsy return is a non-match on both seams and a truthy one a match; ``criteria`` reports a falsy non-bool
once per (group, filter feature) while a literal ``False`` stays quiet; and a return whose ``__bool__`` raises
is contained like a raising hook (#899), rolled back and recorded.

Probe classes live inside factory functions and are dropped before any assert runs, so a failing assert never
pins a throwaway FeatureGroup into its traceback and trips the no-leak fixture in tests/conftest.py.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, TypeVar

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.provider import DataCreator
from mloda.user import Feature, FeatureName, FilterType, GlobalFilter, Options, PluginCollector, SingleFilter, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


GF_LOGGER_NAME = "mloda.core.filter.global_filter"
IFG_LOGGER_NAME = "mloda.core.prepare.identify_feature_group"

PROBE_CLASS_NAME = "NonBoolMatcherFG927"
MUTATING_CLASS_NAME = "MutatingExplodingBoolFG927"
HOST_FEATURE = "nbm_host_feat_927"  # the resolved feature the filters are matched against
FILTER_FEATURE = "nbm_filter_feat_927"  # the hook returns the probed value for this name

BOOL_RAISE_MESSAGE = "boom_927_bool_exploded"
BOOL_RAISE_TYPE_NAME = "RuntimeError"
MATCHER_ERROR_STAGE = "matcher_error"

OPTION_KEY_927 = "nbm_option_key_927"  # written by the hook before its return value explodes
OPTION_VALUE_927 = "written_before_the_bool_exploded_927"

E2E_MAIN = "nbm_main_feat_927"  # requested root feature; must keep resolving
E2E_TARGET = "nbm_target_feat_927"  # never requested directly; only reachable as a matched filter feature

T = TypeVar("T")


class ExplodingBool927:
    """A returned value that is neither True nor False and whose truthiness test raises."""

    def __bool__(self) -> bool:
        raise RuntimeError(BOOL_RAISE_MESSAGE)

    def __repr__(self) -> str:
        # Fixed text: a snapshot must be able to show this value without triggering the raise.
        return "<ExplodingBool927>"


class FalsyBool927:
    """A returned value that is not False and says no only through __bool__: bool() alone sees it."""

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "<FalsyBool927>"


class TruthyBool927:
    """The mirror: a returned value that is not True and says yes only through __bool__."""

    def __bool__(self) -> bool:
        return True

    def __repr__(self) -> str:
        return "<TruthyBool927>"


# Keyed by id so parametrize stays readable; the literal rows are the controls whose verdict never moved.
FALSY_NON_BOOL_RETURNS: dict[str, Any] = {
    "none": None,
    "zero": 0,
    "empty_string": "",
    "empty_list": [],
    "falsy_bool_object": FalsyBool927(),
}
TRUTHY_NON_BOOL_RETURNS: dict[str, Any] = {
    "non_empty_string": "yes",
    "one": 1,
    "non_empty_list": [1],
    "truthy_bool_object": TruthyBool927(),
}
FALSY_RETURNS: dict[str, Any] = {**FALSY_NON_BOOL_RETURNS, "literal_false": False}
TRUTHY_RETURNS: dict[str, Any] = {**TRUTHY_NON_BOOL_RETURNS, "literal_true": True}


def _capture(call: Callable[[], T]) -> tuple[Optional[T], Optional[str]]:
    """Run call, returning (value, None) or (None, 'Type: message'). No traceback is retained."""
    try:
        return call(), None
    except Exception as exc:  # noqa: BLE001  (red-phase probe: an escape is the fact under test)
        return None, f"{type(exc).__name__}: {exc}"


def _shown(value: Any) -> str:
    """Type and repr of a returned value, so an identity assert can report it without a truthiness test."""
    return f"{type(value).__name__} {value!r}"


def _single(filter_feature_name: str) -> SingleFilter:
    """A minimal EQUAL filter on one feature name."""
    return SingleFilter(filter_feature_name, FilterType.EQUAL, {"value": 1})


def _messages(caplog: pytest.LogCaptureFixture, level: int, logger_name: str = GF_LOGGER_NAME) -> tuple[str, ...]:
    """Formatted messages that logger logged at exactly that level."""
    records = [record for record in caplog.records if record.name == logger_name and record.levelno == level]
    return tuple(record.getMessage() for record in records)


def _dropped_entries(global_filter: GlobalFilter) -> tuple[tuple[str, str], ...]:
    """(group class name, filter feature name) per recorded drop, sorted. Holds no class."""
    return tuple(sorted((key[0].get_class_name(), key[1]) for key in global_filter.dropped_filters))


def _make_non_bool_matcher_fg(returned: Any) -> type[FeatureGroup]:
    """A throwaway group whose hook returns the caller's value for FILTER_FEATURE and matches HOST_FEATURE."""
    # Class objects are cyclic; collect leftovers from earlier tests before defining a twin.
    gc.collect()

    class NonBoolMatcherFG927(FeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {HOST_FEATURE}

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: Optional[DataAccessCollection] = None,
        ) -> Any:  # Any, not bool: returning a non-bool out of a `-> bool` hook is the case under test.
            if str(feature_name) == FILTER_FEATURE:
                return returned
            return str(feature_name) in cls.feature_names_supported()

    return NonBoolMatcherFG927


def _make_mutating_exploding_bool_fg() -> type[FeatureGroup]:
    """A throwaway group whose hook WRITES to the shared options and then returns an unreadable value."""
    gc.collect()

    class MutatingExplodingBoolFG927(FeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {HOST_FEATURE}

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: Optional[DataAccessCollection] = None,
        ) -> Any:  # Any, not bool: the write lands, then reading the return value is what raises.
            options.add_to_group(OPTION_KEY_927, OPTION_VALUE_927)
            return ExplodingBool927()

    return MutatingExplodingBoolFG927


@dataclass(frozen=True)
class _CriteriaSnapshot:
    """Plain-data readout of GlobalFilter.criteria calls. Holds no class and no exception object."""

    is_false: bool
    is_true: bool
    shown: str
    escaped: Optional[str]
    entries: tuple[tuple[str, str], ...]
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    debugs: tuple[str, ...]


def _drive_criteria(returned: Any, caplog: pytest.LogCaptureFixture, calls: int = 1) -> _CriteriaSnapshot:
    """Call criteria `calls` times on ONE fresh GlobalFilter; the finally unbinds every name that pins the class."""
    caplog.clear()
    fg = _make_non_bool_matcher_fg(returned)
    global_filter = GlobalFilter()
    try:
        value: Any = None
        escaped: Optional[str] = None
        with caplog.at_level(logging.DEBUG, logger=GF_LOGGER_NAME):
            for _ in range(calls):
                value, escaped = _capture(partial(global_filter.criteria, fg, _single(FILTER_FEATURE), None))
        return _CriteriaSnapshot(
            is_false=value is False,
            is_true=value is True,
            shown=_shown(value),
            escaped=escaped,
            entries=_dropped_entries(global_filter),
            reasons=tuple(global_filter.dropped_filters.values()),
            warnings=_messages(caplog, logging.WARNING),
            debugs=_messages(caplog, logging.DEBUG),
        )
    finally:
        del fg, global_filter
        gc.collect()


@dataclass(frozen=True)
class _MatchedFilterSnapshot:
    """Plain-data readout of one identify_matched_filters call. Holds no class and no filter object."""

    names: tuple[str, ...]
    escaped: Optional[str]
    entries: tuple[tuple[str, str], ...]


def _drive_identify_matched_filters(returned: Any) -> _MatchedFilterSnapshot:
    """Match one registered filter against HOST_FEATURE and read the attached filter names out as text."""
    fg = _make_non_bool_matcher_fg(returned)
    global_filter = GlobalFilter()
    global_filter.add_filter(FILTER_FEATURE, FilterType.EQUAL, {"value": 1})
    matched = None
    try:
        matched, escaped = _capture(partial(global_filter.identify_matched_filters, fg, Feature(HOST_FEATURE), None))
        return _MatchedFilterSnapshot(
            names=() if matched is None else tuple(sorted(single.name for single in matched)),
            escaped=escaped,
            entries=_dropped_entries(global_filter),
        )
    finally:
        del fg, global_filter, matched
        gc.collect()


class _RawCriteriaGlobalFilter927(GlobalFilter):
    """A GlobalFilter answering criteria with a raw value, so only the gate that reads it decides."""

    raw: Any = None

    def criteria(
        self,
        feature_group: type[FeatureGroup],
        filter: SingleFilter,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> Any:  # Any, not bool: bypassing the coercion inside criteria is the point of this double.
        return self.raw


def _drive_matched_filter_gate(raw: Any) -> _MatchedFilterSnapshot:
    """Match with criteria stubbed to a raw value, so the coercion inside criteria cannot mask the gate."""
    # The hook is never consulted: the stub answers before identify_matched_filters would ask it.
    fg = _make_non_bool_matcher_fg(False)
    global_filter = _RawCriteriaGlobalFilter927()
    global_filter.raw = raw
    global_filter.add_filter(FILTER_FEATURE, FilterType.EQUAL, {"value": 1})
    matched = None
    try:
        matched, escaped = _capture(partial(global_filter.identify_matched_filters, fg, Feature(HOST_FEATURE), None))
        return _MatchedFilterSnapshot(
            names=() if matched is None else tuple(sorted(single.name for single in matched)),
            escaped=escaped,
            entries=_dropped_entries(global_filter),
        )
    finally:
        del fg, global_filter, matched
        gc.collect()


@dataclass(frozen=True)
class _ResolutionSnapshot:
    """Plain-data readout of one _filter_feature_group_by_criteria call. Holds no class."""

    is_false: bool
    is_true: bool
    shown: str
    escaped: Optional[str]
    warnings: tuple[str, ...]
    matcher_errors: tuple[tuple[str, str], ...]
    option_keys: tuple[str, ...]


def _drive_resolution(build: Callable[[], type[FeatureGroup]], caplog: pytest.LogCaptureFixture) -> _ResolutionSnapshot:
    """Call the resolution seam directly; the finally unbinds the identifier, which keys reasons by class."""
    caplog.clear()
    fg = build()
    identifier = IdentifyFeatureGroupClass()
    feature = Feature(FILTER_FEATURE)
    try:
        with caplog.at_level(logging.DEBUG, logger=IFG_LOGGER_NAME):
            value, escaped = _capture(partial(identifier._filter_feature_group_by_criteria, fg, feature, None))
        return _ResolutionSnapshot(
            is_false=value is False,
            is_true=value is True,
            shown=_shown(value),
            escaped=escaped,
            warnings=_messages(caplog, logging.WARNING, IFG_LOGGER_NAME),
            matcher_errors=tuple(
                sorted((group.get_class_name(), reason) for group, reason in identifier._matcher_errors.items())
            ),
            option_keys=tuple(sorted(str(key) for key in feature.options.keys())),
        )
    finally:
        del fg, identifier, feature
        gc.collect()


@dataclass(frozen=True)
class _EliminationSnapshot:
    """Plain-data readout of one evaluate() pass. Holds no class and no Elimination object."""

    escaped: Optional[str]
    identified: tuple[str, ...]
    eliminations: tuple[tuple[str, str, str], ...]


def _drive_filter_loop(returned: Any) -> _EliminationSnapshot:
    """Evaluate FILTER_FEATURE against the probe alone and read the near-miss ledger out as text."""
    fg = _make_non_bool_matcher_fg(returned)
    plugins: FeatureGroupEnvironmentMapping = {fg: {PythonDictFramework}}
    result = None
    try:
        result, escaped = _capture(partial(IdentifyFeatureGroupClass.evaluate, Feature(FILTER_FEATURE), plugins, None))
        identified: tuple[str, ...] = ()
        eliminations: tuple[tuple[str, str, str], ...] = ()
        if result is not None:
            identified = tuple(sorted(g.get_class_name() for g in result.identified))
            eliminations = tuple(
                sorted((g.get_class_name(), str(e.stage), str(e.reason)) for g, e in result.eliminations.items())
            )
        del result
        result = None
        return _EliminationSnapshot(escaped=escaped, identified=identified, eliminations=eliminations)
    finally:
        del fg, plugins, result
        gc.collect()


class TestFalsyReturnIsANonMatch:
    """A falsy return is a non-match on both seams, and the answer that comes back is a real bool."""

    @pytest.mark.parametrize("returned_id", sorted(FALSY_RETURNS))
    def test_criteria_returns_exactly_false(self, returned_id: str, caplog: pytest.LogCaptureFixture) -> None:
        """Truthiness decides, and the answer is False itself: a falsy value must not leak out of a bool seam."""
        snapshot = _drive_criteria(FALSY_RETURNS[returned_id], caplog)

        assert snapshot.escaped is None, f"nothing may cross GlobalFilter.criteria: {snapshot.escaped}"
        assert snapshot.is_false, f"a falsy return is a non-match and must come back as False, got: {snapshot.shown}"

    @pytest.mark.parametrize("returned_id", sorted(FALSY_RETURNS))
    def test_identify_matched_filters_attaches_nothing(self, returned_id: str) -> None:
        """The API boundary: a filter whose hook answered falsy must not reach the matched set."""
        snapshot = _drive_identify_matched_filters(FALSY_RETURNS[returned_id])

        assert snapshot.escaped is None, f"nothing may cross identify_matched_filters: {snapshot.escaped}"
        assert snapshot.names == (), f"a falsy return must attach no filter, got: {list(snapshot.names)}"

    def test_a_falsy_return_warns_but_is_not_a_recorded_drop(self, caplog: pytest.LogCaptureFixture) -> None:
        """dropped_filters records contained raises (#899); a falsy non-bool is only reported, never recorded."""
        snapshot = _drive_criteria(FALSY_NON_BOOL_RETURNS["none"], caplog)

        assert snapshot.escaped is None, f"nothing may cross GlobalFilter.criteria: {snapshot.escaped}"
        assert snapshot.entries == (), f"a non-match is not a drop, got: {snapshot.entries}"
        assert len(snapshot.warnings) == 1, f"the detached filter must still be reported, got: {snapshot.warnings}"
        assert snapshot.is_false, f"a falsy return is still a plain non-match, got: {snapshot.shown}"

    @pytest.mark.parametrize("returned_id", sorted(FALSY_RETURNS))
    def test_the_resolution_seam_returns_exactly_false(
        self, returned_id: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Same verdict as the filter seam, and a real bool out of a `-> bool` method."""
        snapshot = _drive_resolution(partial(_make_non_bool_matcher_fg, FALSY_RETURNS[returned_id]), caplog)

        assert snapshot.escaped is None, f"nothing may cross the resolution seam: {snapshot.escaped}"
        assert snapshot.is_false, f"a falsy return must come back as False, not raw, got: {snapshot.shown}"

    def test_a_falsy_return_records_no_matcher_error(self) -> None:
        """Through _filter_loop: declining is the candidate's judgment, not a defect worth a near-miss."""
        snapshot = _drive_filter_loop(FALSY_NON_BOOL_RETURNS["none"])

        assert snapshot.escaped is None, f"nothing may cross evaluate: {snapshot.escaped}"
        assert snapshot.identified == (), f"a falsy return must win nothing, got: {snapshot.identified}"
        errors = [entry for entry in snapshot.eliminations if entry[1] == MATCHER_ERROR_STAGE]
        assert errors == [], f"a non-match must not be recorded as a matcher defect, got: {errors}"


class TestTruthyNonBoolReturnIsAMatch:
    """Truthiness decides in both directions: a truthy non-bool matches, and comes back as True."""

    @pytest.mark.parametrize("returned_id", sorted(TRUTHY_RETURNS))
    def test_criteria_returns_exactly_true_and_the_filter_attaches(
        self, returned_id: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The filter still attaches, and the seam's own answer is True itself."""
        matched = _drive_identify_matched_filters(TRUTHY_RETURNS[returned_id])
        snapshot = _drive_criteria(TRUTHY_RETURNS[returned_id], caplog)

        assert matched.escaped is None, f"nothing may cross identify_matched_filters: {matched.escaped}"
        assert matched.names == (FILTER_FEATURE,), f"a truthy return must attach the filter, got: {matched.names}"
        assert snapshot.escaped is None, f"nothing may cross GlobalFilter.criteria: {snapshot.escaped}"
        assert snapshot.is_true, f"a truthy return is a match and must come back as True, got: {snapshot.shown}"

    @pytest.mark.parametrize("returned_id", sorted(TRUTHY_RETURNS))
    def test_the_resolution_seam_returns_exactly_true(self, returned_id: str, caplog: pytest.LogCaptureFixture) -> None:
        """Same verdict as the filter seam, and a real bool out of a `-> bool` method."""
        snapshot = _drive_resolution(partial(_make_non_bool_matcher_fg, TRUTHY_RETURNS[returned_id]), caplog)

        assert snapshot.escaped is None, f"nothing may cross the resolution seam: {snapshot.escaped}"
        assert snapshot.is_true, f"a truthy return must come back as True, not raw, got: {snapshot.shown}"


class TestFalsyNonBoolReturnIsReportedOnce:
    """The change is not silent: a filter that used to attach is now detached, so its author is told."""

    @pytest.mark.parametrize("returned_id", sorted(FALSY_NON_BOOL_RETURNS))
    def test_it_warns_naming_the_group_the_filter_and_the_fix(
        self, returned_id: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """One WARNING that names both halves of the key and says how to keep the filter attached."""
        snapshot = _drive_criteria(FALSY_NON_BOOL_RETURNS[returned_id], caplog)

        assert snapshot.escaped is None, f"nothing may cross GlobalFilter.criteria: {snapshot.escaped}"
        assert len(snapshot.warnings) == 1, f"a falsy non-bool return must warn exactly once, got: {snapshot.warnings}"
        message = snapshot.warnings[0]
        assert PROBE_CLASS_NAME in message, f"the warning must name the feature group: {message}"
        assert FILTER_FEATURE in message, f"the warning must name the filter feature: {message}"
        assert "True" in message, f"the warning must name the explicit True that keeps the filter: {message}"
        assert "return" in message.lower(), f"the warning must tell the author to return it: {message}"

    def test_a_repeat_for_the_same_key_drops_to_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        """The hook is probed per served feature, so only the first ask of a key is worth a WARNING."""
        snapshot = _drive_criteria(FALSY_NON_BOOL_RETURNS["none"], caplog, calls=2)

        assert snapshot.escaped is None, f"nothing may cross GlobalFilter.criteria: {snapshot.escaped}"
        assert len(snapshot.warnings) == 1, f"two probes of one key must warn once, got: {snapshot.warnings}"
        assert len(snapshot.debugs) == 1, f"the repeat must still be reported, at DEBUG, got: {snapshot.debugs}"
        assert PROBE_CLASS_NAME in snapshot.debugs[0], f"the repeat must name the feature group: {snapshot.debugs[0]}"
        assert FILTER_FEATURE in snapshot.debugs[0], f"the repeat must name the filter feature: {snapshot.debugs[0]}"

    def test_a_literal_false_return_reports_nothing(self, caplog: pytest.LogCaptureFixture) -> None:
        """The control: False is the correct way to say no, and reporting it would drown the signal."""
        snapshot = _drive_criteria(False, caplog)

        assert snapshot.is_false, f"a literal False is the plain non-match, got: {snapshot.shown}"
        assert snapshot.warnings == (), f"saying no correctly must not warn, got: {snapshot.warnings}"
        assert snapshot.debugs == (), f"saying no correctly must not log at all, got: {snapshot.debugs}"
        assert snapshot.entries == (), f"a non-match is not a drop, got: {snapshot.entries}"

    @pytest.mark.parametrize("returned_id", sorted(TRUTHY_RETURNS))
    def test_a_matching_return_reports_nothing(self, returned_id: str, caplog: pytest.LogCaptureFixture) -> None:
        """Only the changed verdict is reported: a truthy return attaches the filter exactly as it did before."""
        snapshot = _drive_criteria(TRUTHY_RETURNS[returned_id], caplog)

        assert snapshot.is_true, f"a truthy return is still the match, got: {snapshot.shown}"
        assert snapshot.warnings == (), f"an unchanged verdict must not warn, got: {snapshot.warnings}"

    @pytest.mark.parametrize("returned_id", sorted(FALSY_NON_BOOL_RETURNS))
    def test_the_resolution_seam_stays_quiet(self, returned_id: str, caplog: pytest.LogCaptureFixture) -> None:
        """That seam probes every candidate for every feature and its verdict did not change, so it says nothing."""
        snapshot = _drive_resolution(partial(_make_non_bool_matcher_fg, FALSY_NON_BOOL_RETURNS[returned_id]), caplog)

        assert snapshot.escaped is None, f"nothing may cross the resolution seam: {snapshot.escaped}"
        assert snapshot.warnings == (), f"the resolution seam must not warn on a falsy return, got: {snapshot.warnings}"


class TestTheMatchedFilterGateReadsTruthiness:
    """The gate in identify_matched_filters, pinned past the coercion inside criteria that would mask it."""

    @pytest.mark.parametrize("returned_id", sorted(FALSY_RETURNS))
    def test_a_raw_falsy_criteria_attaches_nothing(self, returned_id: str) -> None:
        """`is False` would attach the filter for every row but the literal one; the gate must read truthiness."""
        snapshot = _drive_matched_filter_gate(FALSY_RETURNS[returned_id])

        assert snapshot.escaped is None, f"nothing may cross identify_matched_filters: {snapshot.escaped}"
        assert snapshot.names == (), f"a falsy criteria answer must attach no filter, got: {list(snapshot.names)}"

    @pytest.mark.parametrize("returned_id", sorted(TRUTHY_RETURNS))
    def test_a_raw_truthy_criteria_attaches_the_filter(self, returned_id: str) -> None:
        """The mirror, so the gate is pinned as truthiness and not as a blanket refusal."""
        snapshot = _drive_matched_filter_gate(TRUTHY_RETURNS[returned_id])

        assert snapshot.escaped is None, f"nothing may cross identify_matched_filters: {snapshot.escaped}"
        assert snapshot.names == (FILTER_FEATURE,), f"a truthy criteria answer must attach the filter: {snapshot.names}"


class TestRaisingBoolIsContained:
    """Reading the return is itself a plugin call, so a raising __bool__ is contained like a raising hook."""

    def test_criteria_contains_it_and_records_the_drop(self, caplog: pytest.LogCaptureFixture) -> None:
        """This one IS a crash, so unlike a plain falsy non-match it lands in the ledger and warns once."""
        snapshot = _drive_criteria(ExplodingBool927(), caplog)

        assert snapshot.escaped is None, f"the raise must not cross GlobalFilter.criteria: {snapshot.escaped}"
        assert snapshot.is_false, f"an unreadable return is a non-match for that filter, got: {snapshot.shown}"
        assert snapshot.entries == ((PROBE_CLASS_NAME, FILTER_FEATURE),), (
            f"exactly one drop, keyed by group and filter feature, got: {snapshot.entries}"
        )
        assert len(snapshot.warnings) == 1, f"exactly one WARNING must report the drop, got: {snapshot.warnings}"
        message = snapshot.warnings[0]
        assert PROBE_CLASS_NAME in message, f"the warning must name the feature group: {message}"
        assert FILTER_FEATURE in message, f"the warning must name the filter feature: {message}"
        assert BOOL_RAISE_TYPE_NAME in message, f"the warning must name the exception type: {message}"
        assert BOOL_RAISE_MESSAGE in message, f"the warning must carry the raise message: {message}"

    def test_identify_matched_filters_attaches_nothing(self) -> None:
        """An unreadable answer must not widen the filter set at the API boundary either."""
        snapshot = _drive_identify_matched_filters(ExplodingBool927())

        assert snapshot.escaped is None, f"the raise must not cross identify_matched_filters: {snapshot.escaped}"
        assert snapshot.names == (), f"an unreadable return must attach no filter, got: {list(snapshot.names)}"

    def test_the_resolution_seam_contains_it_too(self, caplog: pytest.LogCaptureFixture) -> None:
        """The truthiness test happens inside the seam's own containment, not after it."""
        snapshot = _drive_resolution(partial(_make_non_bool_matcher_fg, ExplodingBool927()), caplog)

        assert snapshot.escaped is None, f"the raise must not cross the resolution seam: {snapshot.escaped}"
        assert snapshot.is_false, f"an unreadable return is a non-match for that candidate, got: {snapshot.shown}"
        assert len(snapshot.matcher_errors) == 1, (
            f"containment must record the candidate's defect, got: {snapshot.matcher_errors}"
        )
        name, reason = snapshot.matcher_errors[0]
        assert name == PROBE_CLASS_NAME, f"the entry must be keyed by the broken candidate, got: {name}"
        assert BOOL_RAISE_TYPE_NAME in reason, f"the reason must name the exception type: {reason}"
        assert BOOL_RAISE_MESSAGE in reason, f"the reason must carry the raise message: {reason}"

    def test_the_filter_loop_records_a_matcher_error_near_miss(self) -> None:
        """An unreadable answer is the candidate's own defect, so it is a near-miss, not a silent non-match."""
        snapshot = _drive_filter_loop(ExplodingBool927())

        assert snapshot.escaped is None, f"the raise must not cross evaluate: {snapshot.escaped}"
        assert snapshot.identified == (), f"a broken candidate must win nothing, got: {snapshot.identified}"
        assert len(snapshot.eliminations) == 1, f"exactly one near-miss, got: {snapshot.eliminations}"
        name, stage, reason = snapshot.eliminations[0]
        assert name == PROBE_CLASS_NAME, f"the near-miss must name the broken candidate, got: {name}"
        assert stage == MATCHER_ERROR_STAGE, f"an unreadable return is a matcher defect, got stage: {stage}"
        assert BOOL_RAISE_TYPE_NAME in reason, f"the reason must name the exception type: {reason}"
        assert BOOL_RAISE_MESSAGE in reason, f"the reason must carry the raise message: {reason}"

    def test_a_write_made_before_the_bool_exploded_is_rolled_back(self, caplog: pytest.LogCaptureFixture) -> None:
        """Containment owns the truthiness test, so its option rollback covers it too (#845 follow-up)."""
        snapshot = _drive_resolution(_make_mutating_exploding_bool_fg, caplog)

        assert snapshot.escaped is None, f"the raise must not cross the resolution seam: {snapshot.escaped}"
        assert snapshot.is_false, f"an unreadable return is a non-match for that candidate, got: {snapshot.shown}"
        assert OPTION_KEY_927 not in snapshot.option_keys, (
            f"the write must not survive an unreadable return, got: {snapshot.option_keys}"
        )
        assert len(snapshot.matcher_errors) == 1, (
            f"containment must record the candidate's defect, got: {snapshot.matcher_errors}"
        )
        assert snapshot.matcher_errors[0][0] == MUTATING_CLASS_NAME, (
            f"the entry must be keyed by the mutating candidate, got: {snapshot.matcher_errors[0][0]}"
        )


def _make_e2e_probe_fg() -> type[FeatureGroup]:
    """A throwaway root group that resolves E2E_MAIN and answers None when matched for the filter feature."""
    gc.collect()

    class NonBoolMatchE2EFG927(FeatureGroup):
        @classmethod
        def input_data(cls) -> DataCreator:
            return DataCreator({E2E_MAIN, E2E_TARGET})

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: Optional[DataAccessCollection] = None,
        ) -> Any:  # Any, not bool: returning None out of a `-> bool` hook is the case under test.
            if str(feature_name) == E2E_TARGET:
                return None
            return str(feature_name) == E2E_MAIN

        @classmethod
        def final_filters(cls) -> bool:
            # The payload is not filterable data, so post-calculation row elimination must not run against it.
            return False

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            payload = {
                "names": sorted(str(f.name) for f in features.features),
                "filter_count": len(features.filters) if features.filters else 0,
            }
            return {str(feature.name): [payload] for feature in features.features}

    return NonBoolMatchE2EFG927


def _single_row(frame: Any, column: str) -> Any:
    """Extract the single payload row, tolerant of columnar dict or list-of-row-dicts results."""
    if isinstance(frame, dict):
        values = list(frame[column])
    else:
        values = [row[column] for row in frame]
    assert len(values) == 1, f"expected exactly one row for {column}, got {values!r}"
    return values[0]


def _run() -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Run E2E_MAIN under a filter whose hook answers None; the escape is text, never an exception object."""
    fg = _make_e2e_probe_fg()
    collector = PluginCollector.enabled_feature_groups({fg})
    global_filter = GlobalFilter()
    global_filter.add_filter(E2E_TARGET, FilterType.EQUAL, {"value": 1})
    results, escaped = _capture(
        partial(
            mloda.run_all,
            [Feature(E2E_MAIN)],
            compute_frameworks={PythonDictFramework},
            plugin_collector=collector,
            global_filter=global_filter,
        )
    )
    del fg, collector
    gc.collect()
    if results is None:
        return None, escaped
    assert len(results) == 1, f"expected exactly one result frame, got: {results!r}"
    payload = _single_row(results[0], E2E_MAIN)
    assert isinstance(payload, dict)
    return payload, escaped


def test_a_none_returning_hook_attaches_no_filter_end_to_end() -> None:
    """Through mloda.run_all: the filter feature must not join the FeatureSet on a falsy match answer."""
    payload, escaped = _run()

    assert escaped is None, f"a falsy match answer must not take the run down: {escaped}"
    assert payload is not None
    assert payload["names"] == [E2E_MAIN], f"the filter feature must not attach: {payload!r}"
    assert payload["filter_count"] == 0, f"no filter may match: {payload!r}"
