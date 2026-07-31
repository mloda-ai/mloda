"""Red-phase pins for issue #899: a raising match hook during FILTER matching must be contained.

``GlobalFilter.criteria`` calls ``match_feature_group_criteria`` uncontained, so one broken hook takes
the whole run down, while the resolution seam (#845, ``IdentifyFeatureGroupClass._filter_feature_group_by_criteria``)
treats the very same raise as a contained non-match. Containment here drops only that filter for that
feature group and logs it at WARNING, deliberately not through the seam's ``contained_raise_log_level``:
this path carries no resolution-failure message, and a silently dropped filter changes results. The
reason stays TEXT, so no retained traceback pins the plugin class. A raise marked with
``escalate_match_abort`` still propagates untouched, exactly as in the resolution seam.

Probe classes live inside factory functions and are dropped before any assert runs, so a failing assert
never pins a throwaway FeatureGroup into its traceback and trips the no-leak fixture in tests/conftest.py.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable
from functools import partial
from typing import Any, Optional, TypeVar

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.utils import escalate_match_abort
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.provider import DataCreator
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

E2E_MAIN = "gfc_main_feat_899"  # requested root feature; must keep resolving
E2E_TARGET = "gfc_target_feat_899"  # never requested directly; only reachable as a matched filter feature

T = TypeVar("T")


def _capture(call: Callable[[], T]) -> tuple[Optional[T], Optional[str]]:
    """Run call, returning (value, None) or (None, 'Type: message'). No traceback is retained."""
    try:
        return call(), None
    except Exception as exc:  # noqa: BLE001  (red-phase probe: an escape is the fact under test)
        return None, f"{type(exc).__name__}: {exc}"


def _single(filter_feature_name: str) -> SingleFilter:
    """A minimal EQUAL filter on one feature name."""
    return SingleFilter(filter_feature_name, FilterType.EQUAL, {"value": 1})


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
            data_access_collection: Optional[DataAccessCollection] = None,
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
            data_access_collection: Optional[DataAccessCollection] = None,
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
            data_access_collection: Optional[DataAccessCollection] = None,
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


def _run(escalate: bool) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Run E2E_MAIN under a global EQUAL filter on E2E_TARGET; return (E2E_MAIN payload, escaped text).

    The escape is captured as text rather than caught by pytest.raises: an ExceptionInfo keeps the
    hook frame alive, and that frame's `cls` pins the throwaway class for the no-leak fixture. `fg`
    and `collector` are deleted from THIS frame before the asserts below for the same reason.
    """
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
