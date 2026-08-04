"""Issue #927: both match seams read the hook's return by TRUTHINESS and answer with a real bool.

``GlobalFilter.criteria`` is gated on ``is False`` today, so a hook returning None attaches the filter while
``IdentifyFeatureGroupClass._filter_feature_group_by_criteria`` calls that same None a non-match. Pinned here: a
falsy return (None, 0, "", []) is an ordinary non-match on both seams, exactly ``False``, attaching no filter and
recording no drop; a truthy non-bool is a match, exactly ``True``; and a return whose ``__bool__`` raises is
contained like a raising hook (#899), recorded and logged on the filter path. Probe classes live inside factory
functions and are dropped before any assert runs, so a failing assert never pins a throwaway FeatureGroup into its
traceback and trips the no-leak fixture in tests/conftest.py.
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
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.provider import DataCreator
from mloda.user import Feature, FeatureName, FilterType, GlobalFilter, Options, PluginCollector, SingleFilter, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


GF_LOGGER_NAME = "mloda.core.filter.global_filter"

PROBE_CLASS_NAME = "NonBoolMatcherFG927"
HOST_FEATURE = "nbm_host_feat_927"  # the resolved feature the filters are matched against
FILTER_FEATURE = "nbm_filter_feat_927"  # the hook returns the probed value for this name

BOOL_RAISE_MESSAGE = "boom_927_bool_exploded"
BOOL_RAISE_TYPE_NAME = "RuntimeError"

E2E_MAIN = "nbm_main_feat_927"  # requested root feature; must keep resolving
E2E_TARGET = "nbm_target_feat_927"  # never requested directly; only reachable as a matched filter feature

# Keyed by id so parametrize stays readable and each probed value keeps its own type.
FALSY_RETURNS: dict[str, Any] = {"none": None, "zero": 0, "empty_string": "", "empty_list": []}
TRUTHY_RETURNS: dict[str, Any] = {"non_empty_string": "yes", "one": 1, "non_empty_list": [1]}

T = TypeVar("T")


class ExplodingBool927:
    """A returned value that is neither True nor False and whose truthiness test raises."""

    def __bool__(self) -> bool:
        raise RuntimeError(BOOL_RAISE_MESSAGE)

    def __repr__(self) -> str:
        # Fixed text: a snapshot must be able to show this value without triggering the raise.
        return "<ExplodingBool927>"


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


def _messages(caplog: pytest.LogCaptureFixture, level: int) -> tuple[str, ...]:
    """Formatted messages GlobalFilter logged at exactly that level."""
    records = [record for record in caplog.records if record.name == GF_LOGGER_NAME and record.levelno == level]
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


@dataclass(frozen=True)
class _CriteriaSnapshot:
    """Plain-data readout of one GlobalFilter.criteria call. Holds no class and no exception object."""

    is_false: bool
    is_true: bool
    shown: str
    escaped: Optional[str]
    entries: tuple[tuple[str, str], ...]
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]


def _drive_criteria(returned: Any, caplog: pytest.LogCaptureFixture) -> _CriteriaSnapshot:
    """Call criteria once against a fresh GlobalFilter; the finally unbinds every name that pins the class."""
    caplog.clear()
    fg = _make_non_bool_matcher_fg(returned)
    global_filter = GlobalFilter()
    try:
        with caplog.at_level(logging.WARNING, logger=GF_LOGGER_NAME):
            value, escaped = _capture(partial(global_filter.criteria, fg, _single(FILTER_FEATURE), None))
        return _CriteriaSnapshot(
            is_false=value is False,
            is_true=value is True,
            shown=_shown(value),
            escaped=escaped,
            entries=_dropped_entries(global_filter),
            reasons=tuple(global_filter.dropped_filters.values()),
            warnings=_messages(caplog, logging.WARNING),
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


@dataclass(frozen=True)
class _ResolutionSnapshot:
    """Plain-data readout of one _filter_feature_group_by_criteria call. Holds no class."""

    is_false: bool
    is_true: bool
    shown: str
    escaped: Optional[str]


def _drive_resolution(returned: Any) -> _ResolutionSnapshot:
    """Call the resolution seam directly; the finally unbinds the identifier, which keys reasons by class."""
    fg = _make_non_bool_matcher_fg(returned)
    identifier = IdentifyFeatureGroupClass()
    try:
        value, escaped = _capture(
            partial(identifier._filter_feature_group_by_criteria, fg, Feature(FILTER_FEATURE), None)
        )
        return _ResolutionSnapshot(is_false=value is False, is_true=value is True, shown=_shown(value), escaped=escaped)
    finally:
        del fg, identifier
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

    def test_a_falsy_return_is_not_a_recorded_drop(self, caplog: pytest.LogCaptureFixture) -> None:
        """Unlike a contained raise (#899), an ordinary falsy non-match leaves no ledger entry and no warning."""
        snapshot = _drive_criteria(FALSY_RETURNS["none"], caplog)

        assert snapshot.escaped is None, f"nothing may cross GlobalFilter.criteria: {snapshot.escaped}"
        assert snapshot.entries == (), f"a non-match is not a drop, got: {snapshot.entries}"
        assert snapshot.warnings == (), f"a non-match must not warn, got: {snapshot.warnings}"
        assert snapshot.is_false, f"a falsy return is still a plain non-match, got: {snapshot.shown}"

    @pytest.mark.parametrize("returned_id", sorted(FALSY_RETURNS))
    def test_the_resolution_seam_returns_exactly_false(self, returned_id: str) -> None:
        """Same verdict as the filter seam, and a real bool out of a `-> bool` method."""
        snapshot = _drive_resolution(FALSY_RETURNS[returned_id])

        assert snapshot.escaped is None, f"nothing may cross the resolution seam: {snapshot.escaped}"
        assert snapshot.is_false, f"a falsy return must come back as False, not raw, got: {snapshot.shown}"


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
    def test_the_resolution_seam_returns_exactly_true(self, returned_id: str) -> None:
        """Same verdict as the filter seam, and a real bool out of a `-> bool` method."""
        snapshot = _drive_resolution(TRUTHY_RETURNS[returned_id])

        assert snapshot.escaped is None, f"nothing may cross the resolution seam: {snapshot.escaped}"
        assert snapshot.is_true, f"a truthy return must come back as True, not raw, got: {snapshot.shown}"


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

    def test_the_resolution_seam_contains_it_too(self) -> None:
        """The truthiness test happens inside the seam's own containment, not after it."""
        snapshot = _drive_resolution(ExplodingBool927())

        assert snapshot.escaped is None, f"the raise must not cross the resolution seam: {snapshot.escaped}"
        assert snapshot.is_false, f"an unreadable return is a non-match for that candidate, got: {snapshot.shown}"


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
            # Read features.filters inline instead: the payload is not filterable data, so the
            # framework's own post-calculation row elimination must not run against it.
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
