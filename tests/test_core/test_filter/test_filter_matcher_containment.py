"""Issue #899: a raising match hook during filter matching is contained as a non-match for that filter.

The drop is recorded in ``GlobalFilter.dropped_filters`` and logged; a raise marked with ``escalate_match_abort``
still aborts. Probe classes live inside factory functions and are dropped before any assert runs, so a failing
assert never pins a throwaway FeatureGroup into its traceback and trips the no-leak fixture in tests/conftest.py.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Any, TypeVar, cast

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.utils import escalate_match_abort
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.provider import DataCreator, DefaultOptionKeys, property_spec
from mloda.user import Feature, FeatureName, FilterType, GlobalFilter, Options, PluginCollector, SingleFilter, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


GF_LOGGER_NAME = "mloda.core.filter.global_filter"

RAISE_MESSAGE = "boom_899_filter_matcher_exploded"
RAISE_TYPE_NAME = "RuntimeError"
ESCALATE_MESSAGE = "abort_899_filter_matcher_escalated"

UNIT_CLASS_NAME = "RaisingFilterMatcherFG899"
HOST_FEATURE = "gfc_host_feat_899"  # the resolved feature the filters are matched against
FILTER_FEATURE_RAISING = "gfc_raising_filter_feat_899"  # the hook raises for this name
FILTER_FEATURE_OK = "gfc_ok_filter_feat_899"  # the hook matches this name
FILTER_FEATURE_UNKNOWN = "gfc_unknown_filter_feat_899"  # the hook answers False for this name, without raising

HOSTILE_CLASS_NAME = "HostileFilterMatcherFG899"
HOSTILE_TYPE_NAME = "HostileFilterMatcherError899"
HOSTILE_STR_MESSAGE = "boom_899_hostile_str_raised"

FILTER_FEATURE_IN_FEATURES = "gfc_in_features_filter_feat_884"

E2E_MAIN = "gfc_main_feat_899"  # requested root feature; must keep resolving
E2E_TARGET = "gfc_target_feat_899"  # never requested directly; only reachable as a matched filter feature

T = TypeVar("T")


def _capture(call: Callable[[], T]) -> tuple[T | None, str | None]:
    """Run call, returning (value, None) or (None, 'Type: message'). No traceback is retained."""
    try:
        return call(), None
    except Exception as exc:  # noqa: BLE001  (red-phase probe: an escape is the fact under test)
        return None, f"{type(exc).__name__}: {exc}"


def _single(filter_feature_name: str, options: Options | None = None) -> SingleFilter:
    """A minimal EQUAL filter on one feature name, optionally carrying that filter feature's own options."""
    filter_feature: Feature | str = filter_feature_name if options is None else Feature(filter_feature_name, options)
    return SingleFilter(filter_feature, FilterType.EQUAL, {"value": 1})


def _make_raising_filter_matcher_fg() -> type[FeatureGroup]:
    """A throwaway group whose hook raises for FILTER_FEATURE_RAISING and matches its other names."""
    # Class objects are cyclic; collect leftovers from earlier tests before defining a twin.
    gc.collect()

    class RaisingFilterMatcherFG899(FeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {HOST_FEATURE, FILTER_FEATURE_OK}

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: DataAccessCollection | None = None,
        ) -> bool:
            if str(feature_name) == FILTER_FEATURE_RAISING:
                raise RuntimeError(RAISE_MESSAGE)
            return str(feature_name) in cls.feature_names_supported()

    return RaisingFilterMatcherFG899


def _make_escalating_filter_matcher_fg(marker: BaseException) -> type[FeatureGroup]:
    """A throwaway group whose hook raises the caller's own exception object, so identity is assertable."""
    gc.collect()

    class EscalatingFilterMatcherFG899(FeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {HOST_FEATURE, FILTER_FEATURE_OK}

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: DataAccessCollection | None = None,
        ) -> bool:
            if str(feature_name) == FILTER_FEATURE_RAISING:
                raise marker
            return str(feature_name) in cls.feature_names_supported()

    return EscalatingFilterMatcherFG899


class TestFilterMatcherContainment:
    """A raising match hook is a contained non-match for that filter, logged at WARNING as text."""

    def test_criteria_contains_a_plain_raise(self) -> None:
        """A plain raise out of the hook makes criteria return False instead of crossing the seam."""
        fg = _make_raising_filter_matcher_fg()
        matched, escaped = _capture(partial(GlobalFilter().criteria, fg, _single(FILTER_FEATURE_RAISING), None))
        del fg
        gc.collect()

        assert escaped is None, f"the raise must not cross GlobalFilter.criteria: {escaped}"
        assert matched is False, f"a raising hook is a non-match for that filter, got: {matched!r}"

    def test_criteria_reraises_an_escalated_abort(self) -> None:
        """An escalate_match_abort-marked raise crosses the seam as the SAME object, not a wrapper."""
        marker = escalate_match_abort(RuntimeError(ESCALATE_MESSAGE))
        fg = _make_escalating_filter_matcher_fg(marker)
        caught: BaseException | None = None
        try:
            GlobalFilter().criteria(fg, _single(FILTER_FEATURE_RAISING), None)
        except BaseException as exc:  # noqa: BLE001  (the escape itself is the fact under test)
            caught = exc
        is_marker = caught is marker
        type_name = None if caught is None else type(caught).__name__
        message = None if caught is None else str(caught)
        # Drop the retained traceback: its frames pin the throwaway class through the hook's `cls`.
        marker.__traceback__ = None
        del fg, caught
        gc.collect()

        assert is_marker, f"the marked exception itself must escape, got: {type_name}: {message}"
        assert type_name == RAISE_TYPE_NAME
        assert message == ESCALATE_MESSAGE

    def test_identify_matched_filters_drops_only_the_raising_filter(self) -> None:
        """Containment is per filter: the raising one is dropped, the matching one still comes back."""
        fg = _make_raising_filter_matcher_fg()
        global_filter = GlobalFilter()
        global_filter.add_filter(FILTER_FEATURE_RAISING, FilterType.EQUAL, {"value": 1})
        global_filter.add_filter(FILTER_FEATURE_OK, FilterType.EQUAL, {"value": 2})
        matched, escaped = _capture(
            partial(global_filter.identify_matched_filters, fg, Feature(HOST_FEATURE), None),
        )
        names = [] if matched is None else sorted(single.name for single in matched)
        del fg, matched, global_filter
        gc.collect()

        assert escaped is None, f"the raise must not cross identify_matched_filters: {escaped}"
        assert names == [FILTER_FEATURE_OK], f"only the raising filter may be dropped, got: {names}"

    def test_contained_raise_logs_a_warning_naming_group_filter_and_reason(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The drop is visible at WARNING and readable: group, filter feature, exception type and message."""
        fg = _make_raising_filter_matcher_fg()
        with caplog.at_level(logging.WARNING, logger=GF_LOGGER_NAME):
            _, escaped = _capture(partial(GlobalFilter().criteria, fg, _single(FILTER_FEATURE_RAISING), None))
        records = [record for record in caplog.records if record.name == GF_LOGGER_NAME]
        messages = [record.getMessage() for record in records]
        exc_infos = [record.exc_info for record in records]
        del fg
        gc.collect()

        assert escaped is None, f"the raise must not cross GlobalFilter.criteria: {escaped}"
        assert len(messages) == 1, f"exactly one WARNING must report the dropped filter, got: {messages}"
        message = messages[0]
        assert UNIT_CLASS_NAME in message, f"the warning must name the feature group: {message}"
        assert FILTER_FEATURE_RAISING in message, f"the warning must name the filter feature: {message}"
        assert RAISE_TYPE_NAME in message, f"the warning must name the exception type: {message}"
        assert RAISE_MESSAGE in message, f"the warning must carry the raise message: {message}"
        assert exc_infos == [None], "the reason is kept as text: no traceback may be retained on the record"


def _is_dunder(name: str) -> bool:
    """Dunder lookups stay untouched so the interpreter's own machinery keeps working on the double."""
    return name.startswith("__") and name.endswith("__")


class HostileFilterMatcherError899(Exception):
    """Hostile to the except block that reads it: attribute access raises and so does str()."""

    def __getattr__(self, name: str) -> Any:
        if _is_dunder(name):
            raise AttributeError(name)
        raise RuntimeError(f"hostile attribute access for '{name}'")

    def __str__(self) -> str:
        raise RuntimeError(HOSTILE_STR_MESSAGE)


def _make_hostile_filter_matcher_fg() -> type[FeatureGroup]:
    """A throwaway group whose hook raises an exception hostile to every read the except block makes."""
    gc.collect()

    class HostileFilterMatcherFG899(FeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {HOST_FEATURE, FILTER_FEATURE_OK}

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: DataAccessCollection | None = None,
        ) -> bool:
            if str(feature_name) == FILTER_FEATURE_RAISING:
                raise HostileFilterMatcherError899(RAISE_MESSAGE)
            return str(feature_name) in cls.feature_names_supported()

    return HostileFilterMatcherFG899


def _ledger(global_filter: GlobalFilter) -> dict[Any, Any] | None:
    """The instance drop ledger, read via getattr so a missing one asserts readably instead of pinning the class."""
    return cast(dict[Any, Any] | None, getattr(global_filter, "dropped_filters", None))


def _reason_of(recorded: Any) -> Any:
    """Reason of one recorded drop, degrading to the record itself when it carries none."""
    return getattr(recorded, "reason", recorded)


def _messages(caplog: pytest.LogCaptureFixture, level: int) -> tuple[str, ...]:
    """Formatted messages GlobalFilter logged at exactly that level."""
    records = [record for record in caplog.records if record.name == GF_LOGGER_NAME and record.levelno == level]
    return tuple(record.getMessage() for record in records)


def _capture_type_name(call: Callable[[], T]) -> tuple[T | None, str | None]:
    """Run call, returning (value, None) or (None, type name). Reads no message: a double's str() raises."""
    try:
        return call(), None
    except Exception as exc:  # noqa: BLE001  (an escape is the fact under test)
        return None, type(exc).__name__


@dataclass(frozen=True)
class _DropSnapshot:
    """Plain-data readout of one or more criteria calls. Holds no class and no exception object."""

    results: tuple[bool | None, ...]
    escaped: str | None
    has_ledger: bool
    keyed_by_group_and_filter: bool
    entries: tuple[tuple[str, str], ...]  # (feature group class name, filter feature name), sorted
    reasons: tuple[str, ...]  # reason text per entry, aligned with entries
    reason_types: tuple[str, ...]  # type name of each stored reason, so text stays assertable
    warnings: tuple[str, ...]
    debugs: tuple[str, ...]


def _drive_criteria(
    make_fg: Callable[[], type[FeatureGroup]],
    filter_feature_name: str,
    caplog: pytest.LogCaptureFixture,
    calls: int = 1,
    options: Options | None = None,
) -> _DropSnapshot:
    """Call criteria `calls` times against ONE GlobalFilter; the finally unbinds every name that pins the class."""
    caplog.clear()
    fg = make_fg()
    global_filter = GlobalFilter()
    # The engine probes a per-match deepcopy of one declaration, so all `calls` share one ledger key.
    declared = _single(filter_feature_name, options)
    ledger: dict[Any, Any] | None = None
    items: list[tuple[Any, Any]] = []
    try:
        results: list[bool | None] = []
        escaped: str | None = None
        with caplog.at_level(logging.DEBUG, logger=GF_LOGGER_NAME):
            for _ in range(calls):
                value, failure = _capture_type_name(partial(global_filter.criteria, fg, deepcopy(declared), None))
                results.append(value)
                if failure is not None:
                    escaped = failure
                    break
        ledger = _ledger(global_filter)
        items = [] if ledger is None else sorted(ledger.items(), key=lambda item: str(item[0]))
        return _DropSnapshot(
            results=tuple(results),
            escaped=escaped,
            has_ledger=ledger is not None,
            keyed_by_group_and_filter=ledger is not None and (fg, filter_feature_name, declared.uuid) in ledger,
            entries=tuple((str(key[0].get_class_name()), str(key[1])) for key, _ in items),
            reasons=tuple(str(_reason_of(recorded)) for _, recorded in items),
            reason_types=tuple(type(_reason_of(recorded)).__name__ for _, recorded in items),
            warnings=_messages(caplog, logging.WARNING),
            debugs=_messages(caplog, logging.DEBUG),
        )
    finally:
        del fg, global_filter, declared, ledger, items
        gc.collect()


class TestDroppedFilterIsRecorded:
    """A dropped filter widens the result set, so it must leave a machine-readable trace, not just a log line."""

    def test_fresh_global_filter_records_no_drops(self) -> None:
        """The ledger is per instance and starts empty, so any entry in it means a real drop happened."""
        ledger = _ledger(GlobalFilter())

        assert ledger is not None, "GlobalFilter must expose a dropped_filters ledger"
        assert ledger == {}, f"a fresh GlobalFilter has dropped nothing, got: {ledger!r}"

    def test_contained_raise_records_group_filter_and_reason(self, caplog: pytest.LogCaptureFixture) -> None:
        """One entry, keyed by (feature group, filter feature name, filter uuid), carrying the WARNING's reason."""
        snapshot = _drive_criteria(_make_raising_filter_matcher_fg, FILTER_FEATURE_RAISING, caplog)

        assert snapshot.escaped is None, f"the raise must not cross GlobalFilter.criteria: {snapshot.escaped}"
        assert snapshot.has_ledger, "GlobalFilter must expose a dropped_filters ledger"
        assert snapshot.entries == ((UNIT_CLASS_NAME, FILTER_FEATURE_RAISING),), (
            f"exactly one drop, whose key names the group and the filter feature, got: {snapshot.entries}"
        )
        assert snapshot.keyed_by_group_and_filter, "the key must be the group CLASS, not its name or a stand-in"
        assert snapshot.reason_types == ("str",), (
            f"the reason stays text: no exception object may be stored, got: {snapshot.reason_types}"
        )
        reason = snapshot.reasons[0]
        assert RAISE_TYPE_NAME in reason, f"the reason must name the exception type: {reason}"
        assert RAISE_MESSAGE in reason, f"the reason must carry the raise message: {reason}"
        assert len(snapshot.warnings) == 1, f"exactly one WARNING must report the drop, got: {snapshot.warnings}"
        assert reason in snapshot.warnings[0], (
            f"the stored reason must be the one the warning carries: {reason} vs {snapshot.warnings[0]}"
        )

    def test_repeat_drops_of_one_key_warn_once_then_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        """One broken hook may warn only once, deduped on the per-instance ledger, never on module state."""
        snapshot = _drive_criteria(_make_raising_filter_matcher_fg, FILTER_FEATURE_RAISING, caplog, calls=3)

        assert snapshot.escaped is None, f"the raise must not cross GlobalFilter.criteria: {snapshot.escaped}"
        assert snapshot.results == (False, False, False), (
            f"every call is a non-match for that filter, got: {snapshot.results}"
        )
        assert len(snapshot.warnings) == 1, f"only the first drop of a key may warn, got: {snapshot.warnings}"
        assert len(snapshot.debugs) == 2, f"every repeat drop must fall back to DEBUG, got: {snapshot.debugs}"
        assert snapshot.entries == ((UNIT_CLASS_NAME, FILTER_FEATURE_RAISING),), (
            f"repeats must not add entries, got: {snapshot.entries}"
        )
        reason = snapshot.reasons[0]
        assert RAISE_TYPE_NAME in reason, f"a repeat must not rewrite the stored reason: {reason}"
        assert RAISE_MESSAGE in reason, f"a repeat must not rewrite the stored reason: {reason}"

    def test_escalated_abort_records_no_drop(self) -> None:
        """A marked abort propagates instead of dropping a filter, so it must leave the ledger empty."""
        marker = escalate_match_abort(RuntimeError(ESCALATE_MESSAGE))
        fg = _make_escalating_filter_matcher_fg(marker)
        global_filter = GlobalFilter()
        _, escaped = _capture(partial(global_filter.criteria, fg, _single(FILTER_FEATURE_RAISING), None))
        ledger = _ledger(global_filter)
        has_ledger = ledger is not None
        entry_count = 0 if ledger is None else len(ledger)
        # Drop the retained traceback: its frames pin the throwaway class through the hook's `cls`.
        marker.__traceback__ = None
        del fg, global_filter, ledger
        gc.collect()

        assert escaped == f"{RAISE_TYPE_NAME}: {ESCALATE_MESSAGE}"
        assert has_ledger, "GlobalFilter must expose a dropped_filters ledger"
        assert entry_count == 0, f"a propagating abort is not a drop, got {entry_count} entries"

    def test_plain_non_match_records_no_drop(self, caplog: pytest.LogCaptureFixture) -> None:
        """A hook that simply answers False is not a drop: only a contained raise may show up in the ledger."""
        snapshot = _drive_criteria(_make_raising_filter_matcher_fg, FILTER_FEATURE_UNKNOWN, caplog)

        assert snapshot.escaped is None, f"an ordinary non-match raises nothing: {snapshot.escaped}"
        assert snapshot.results == (False,), f"an unsupported filter feature is a non-match, got: {snapshot.results}"
        assert snapshot.has_ledger, "GlobalFilter must expose a dropped_filters ledger"
        assert snapshot.entries == (), f"a non-match is not a drop, got: {snapshot.entries}"
        assert snapshot.warnings == (), f"a non-match must not warn, got: {snapshot.warnings}"


def _make_in_features_mixin_fg() -> type[FeatureGroup]:
    """A throwaway mixin group that matches on its option set, so only the in_features gate can reject a filter."""
    gc.collect()

    class InFeaturesMixinFilterFG884(FeatureChainParserMixin, FeatureGroup):
        PROPERTY_MAPPING = {"operation": property_spec("operation", allowed_values=("op1",), context=True)}

        def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
            return None

    return InFeaturesMixinFilterFG884


class TestUnresolvableInFeaturesIsNotADrop:
    def test_unresolvable_in_features_is_a_non_match_without_a_drop(self, caplog: pytest.LogCaptureFixture) -> None:
        """The value never leaves the matcher, so the seam records no drop and warns about nothing."""
        snapshot = _drive_criteria(
            _make_in_features_mixin_fg,
            FILTER_FEATURE_IN_FEATURES,
            caplog,
            options=Options(context={"operation": "op1", DefaultOptionKeys.in_features: ""}),
        )

        assert snapshot.escaped is None, f"nothing may cross GlobalFilter.criteria: {snapshot.escaped}"
        assert snapshot.results == (False,), f"an unresolvable in_features is a non-match, got: {snapshot.results}"
        assert snapshot.entries == (), f"a non-match is not a drop, got: {snapshot.entries}"
        assert snapshot.warnings == (), f"a non-match must not warn, got: {snapshot.warnings}"


class TestHostileExceptionStaysInsideTheExceptBlock:
    """The except block reads the exception itself, so a hostile double must not escape from in there."""

    def test_hostile_raise_is_contained_recorded_and_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """Same exposure as the resolution seam (#845 R2): a raising __getattr__ and a raising __str__."""
        snapshot = _drive_criteria(_make_hostile_filter_matcher_fg, FILTER_FEATURE_RAISING, caplog)

        assert snapshot.escaped is None, f"nothing may escape the except block itself: {snapshot.escaped}"
        assert snapshot.results == (False,), f"a hostile raise is still a non-match, got: {snapshot.results}"
        assert snapshot.entries == ((HOSTILE_CLASS_NAME, FILTER_FEATURE_RAISING),), (
            f"the drop must still be recorded, got: {snapshot.entries}"
        )
        assert snapshot.reason_types == ("str",), (
            f"the reason stays text even when str() raises, got: {snapshot.reason_types}"
        )
        assert HOSTILE_TYPE_NAME in snapshot.reasons[0], (
            f"the reason must name the exception type, the one read that cannot fail: {snapshot.reasons[0]}"
        )
        assert len(snapshot.warnings) == 1, f"the drop must still be logged once, got: {snapshot.warnings}"
        assert HOSTILE_CLASS_NAME in snapshot.warnings[0], f"the warning must name the group: {snapshot.warnings[0]}"
        assert FILTER_FEATURE_RAISING in snapshot.warnings[0], (
            f"the warning must name the filter feature: {snapshot.warnings[0]}"
        )


def _make_e2e_probe_fg(escalate: bool) -> type[FeatureGroup]:
    """A throwaway root group that resolves E2E_MAIN but raises when matched for the filter feature."""
    gc.collect()

    class FilterMatchRaiserFG899(FeatureGroup):
        @classmethod
        def input_data(cls) -> DataCreator:
            return DataCreator({E2E_MAIN, E2E_TARGET})

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: DataAccessCollection | None = None,
        ) -> bool:
            if str(feature_name) == E2E_TARGET:
                if escalate:
                    raise escalate_match_abort(RuntimeError(ESCALATE_MESSAGE))
                raise RuntimeError(RAISE_MESSAGE)
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

    return FilterMatchRaiserFG899


def _single_row(frame: Any, column: str) -> Any:
    """Extract the single payload row, tolerant of columnar dict or list-of-row-dicts results."""
    if isinstance(frame, dict):
        values = list(frame[column])
    else:
        values = [row[column] for row in frame]
    assert len(values) == 1, f"expected exactly one row for {column}, got {values!r}"
    return values[0]


def _run(escalate: bool) -> tuple[dict[str, Any] | None, str | None]:
    """Run E2E_MAIN under a filter; the escape is text, not pytest.raises, whose ExceptionInfo pins the class."""
    fg = _make_e2e_probe_fg(escalate)
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


def test_contained_raise_does_not_abort_the_run() -> None:
    """A hook raising for the filter feature only drops that filter; the requested feature still returns."""
    payload, escaped = _run(escalate=False)

    assert escaped is None, f"filter matching must not take the whole run down: {escaped}"
    assert payload is not None
    assert payload["names"] == [E2E_MAIN], f"the filter feature must not attach: {payload!r}"
    assert payload["filter_count"] == 0, f"no filter may match: {payload!r}"


def test_escalated_raise_still_aborts_the_run() -> None:
    """An escalate_match_abort-marked raise during filter matching still aborts run_all, message intact."""
    payload, escaped = _run(escalate=True)

    assert payload is None, f"the marked abort must not be contained: {payload!r}"
    assert escaped == f"{RAISE_TYPE_NAME}: {ESCALATE_MESSAGE}"
