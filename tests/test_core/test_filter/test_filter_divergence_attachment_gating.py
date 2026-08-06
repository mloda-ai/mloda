"""Divergence warnings fire only for attaching filters; unmatched filters are reported after setup.

The fdg_ prefix keeps this module's keys unique.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.provider import DataCreator, PropertySpec
from mloda.user import FilterType, GlobalFilter, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


FDG_KEY = "fdg_key"
FDG_REQ_KEY = "fdg_req_key"
FDG_DEFAULT = "fdg_default_val"
FDG_HOST_VAL = "fdg_host_val"
FDG_FILTER_VAL = "fdg_filter_val"

FDG_HOST = "fdg_host_feature"
FDG_TARGET = "fdg_target_feature"
FDG_NOWHERE = "fdg_nowhere_feature"  # a filter target no feature group ever serves

UNMATCHED_PHRASE = "matched no feature group"

FDG_RERUN_TARGET = "fdg_rerun_target_feature"
FDG_DEDUP_KEY = "fdg_dedup_key"
FDG_OTHER_HOST_VAL = "fdg_other_host_val"


def _rejecting_probe() -> type[FeatureGroup]:
    class FdgRejectingProbeFeatureGroup(FeatureGroup):
        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: Optional[DataAccessCollection] = None,
        ) -> bool:
            return False

    return FdgRejectingProbeFeatureGroup


def _accepting_probe() -> type[FeatureGroup]:
    class FdgAcceptingProbeFeatureGroup(FeatureGroup):
        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: Optional[DataAccessCollection] = None,
        ) -> bool:
            return str(feature_name) == FDG_TARGET

    return FdgAcceptingProbeFeatureGroup


def _engine_probe(served: set[str], mapping: dict[str, PropertySpec] | None = None) -> type[FeatureGroup]:
    """A throwaway root FeatureGroup serving ``served``, echoing one payload row per feature."""

    class FdgEngineProbeFeatureGroup(FeatureGroup):
        PROPERTY_MAPPING = mapping or {}

        @classmethod
        def input_data(cls) -> DataCreator:
            return DataCreator(served)

        @classmethod
        def final_filters(cls) -> bool:
            # The payload is not filterable data, so row elimination must not run against it.
            return False

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return {str(feature.name): [1] for feature in features.features}

    return FdgEngineProbeFeatureGroup


def _warnings_naming(caplog: pytest.LogCaptureFixture, needle: str) -> list[str]:
    """Every warning-level message naming ``needle``; location-agnostic, since the check may move."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING and needle in record.getMessage()
    ]


def test_criteria_rejected_filter_emits_no_divergence_warning(caplog: pytest.LogCaptureFixture) -> None:
    """A filter dropped by the criteria gate never attaches, so its divergence must not be reported."""
    global_filter = GlobalFilter()
    global_filter.add_filter(
        Feature(FDG_TARGET, Options(group={FDG_KEY: FDG_FILTER_VAL})), FilterType.EQUAL, {"value": 1}
    )
    with caplog.at_level(logging.WARNING):
        matched = global_filter.identify_matched_filters(
            _rejecting_probe(), Feature(FDG_HOST, Options(group={FDG_KEY: FDG_HOST_VAL}))
        )

    assert matched == set()
    assert _warnings_naming(caplog, FDG_KEY) == []


def test_attaching_filter_still_reports_its_divergence(caplog: pytest.LogCaptureFixture) -> None:
    """The control: a filter that clears every gate keeps the unchanged divergence message."""
    global_filter = GlobalFilter()
    global_filter.add_filter(
        Feature(FDG_TARGET, Options(group={FDG_KEY: FDG_FILTER_VAL})), FilterType.EQUAL, {"value": 1}
    )
    with caplog.at_level(logging.WARNING):
        matched = global_filter.identify_matched_filters(
            _accepting_probe(), Feature(FDG_HOST, Options(group={FDG_KEY: FDG_HOST_VAL}))
        )

    assert len(matched) == 1
    assert _warnings_naming(caplog, FDG_KEY) == [
        f"Options are not the same. {FDG_KEY} is different. {FDG_FILTER_VAL} != {FDG_HOST_VAL}"
    ]


def test_warn_on_unmatched_filters_names_a_filter_no_group_matched(caplog: pytest.LogCaptureFixture) -> None:
    """A filter every probe rejected is reported once, naming the filter feature."""
    global_filter = GlobalFilter()
    global_filter.add_filter(Feature(FDG_TARGET, Options()), FilterType.EQUAL, {"value": 1})
    global_filter.identify_matched_filters(_rejecting_probe(), Feature(FDG_HOST, Options()))
    with caplog.at_level(logging.WARNING):
        global_filter.warn_on_unmatched_filters()

    messages = _warnings_naming(caplog, FDG_TARGET)
    assert len(messages) == 1, f"exactly one unmatched-filter warning expected, got: {messages!r}"
    assert UNMATCHED_PHRASE in messages[0]


def test_warn_on_unmatched_filters_is_silent_for_a_filter_matched_in_a_later_call(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Matching is tracked across identify calls: one attachment anywhere silences the diagnostic."""
    global_filter = GlobalFilter()
    global_filter.add_filter(Feature(FDG_TARGET, Options()), FilterType.EQUAL, {"value": 1})
    global_filter.identify_matched_filters(_rejecting_probe(), Feature(FDG_HOST, Options()))
    global_filter.identify_matched_filters(_accepting_probe(), Feature(FDG_HOST, Options()))
    with caplog.at_level(logging.WARNING):
        global_filter.warn_on_unmatched_filters()

    assert _warnings_naming(caplog, FDG_TARGET) == []


def _run(
    served: set[str],
    mapping: dict[str, PropertySpec] | None,
    host_options: Options,
    filter_target: str,
    filter_options: Options,
) -> dict[str, Any]:
    """Run FDG_HOST under a global EQUAL filter on ``filter_target``; deletes probe refs in this frame."""
    collector = PluginCollector.enabled_feature_groups({_engine_probe(served, mapping)})
    global_filter = GlobalFilter()
    global_filter.add_filter(Feature(filter_target, filter_options), FilterType.EQUAL, {"value": 1})
    results = mloda.run_all(
        [Feature(FDG_HOST, host_options)],
        compute_frameworks={PythonDictFramework},
        plugin_collector=collector,
        global_filter=global_filter,
    )

    collection_size = len(global_filter.collection)
    frames_repr = repr(list(results))
    del collector, global_filter, results

    return {"collection_size": collection_size, "frames_repr": frames_repr}


def test_required_when_rejected_filter_is_reported_as_unmatched_without_divergence_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A required_when key declared as None reads as absent, so the guard rejects the filter feature
    through the ordinary criteria gate while the host, which supplies the value, matches."""
    mapping = {
        FDG_REQ_KEY: PropertySpec(
            "A group key required whenever the group matches.",
            context=False,
            default=FDG_DEFAULT,
            required_when=lambda _opts: True,
        )
    }
    with caplog.at_level(logging.WARNING):
        observed = _run(
            {FDG_HOST, FDG_TARGET},
            mapping,
            Options(group={FDG_REQ_KEY: FDG_DEFAULT}),
            FDG_TARGET,
            Options(group={FDG_REQ_KEY: None}),
        )

    assert observed["collection_size"] == 0, f"the filter must not attach: {observed!r}"
    # The unmatched report names the key too, in its nearest-miss suffix; only a divergence warning is forbidden.
    divergence = [message for message in _warnings_naming(caplog, FDG_REQ_KEY) if UNMATCHED_PHRASE not in message]
    assert divergence == [], f"a rejected filter must report no divergence, got: {divergence!r}"
    unmatched = [message for message in _warnings_naming(caplog, FDG_TARGET) if UNMATCHED_PHRASE in message]
    assert len(unmatched) == 1, f"exactly one unmatched-filter warning expected, got: {caplog.text!r}"
    assert FDG_REQ_KEY in unmatched[0], f"the report must carry the nearest miss's own reason, got: {unmatched[0]!r}"


def test_engine_reports_a_filter_targeting_a_name_nothing_serves(caplog: pytest.LogCaptureFixture) -> None:
    """End-to-end: a filter no group serves is named exactly once after setup."""
    with caplog.at_level(logging.WARNING):
        observed = _run({FDG_HOST}, None, Options(), FDG_NOWHERE, Options())

    assert observed["collection_size"] == 0
    unmatched = [message for message in _warnings_naming(caplog, FDG_NOWHERE) if UNMATCHED_PHRASE in message]
    assert len(unmatched) == 1, f"exactly one unmatched-filter warning expected, got: {caplog.text!r}"


def test_engine_stays_silent_for_a_filter_that_attaches(caplog: pytest.LogCaptureFixture) -> None:
    """The control: an attaching filter produces zero unmatched-filter warnings."""
    with caplog.at_level(logging.WARNING):
        observed = _run({FDG_HOST, FDG_TARGET}, None, Options(), FDG_TARGET, Options())

    assert observed["collection_size"] >= 1, f"the filter must have attached: {observed!r}"
    assert [message for message in _warnings_naming(caplog, FDG_TARGET) if UNMATCHED_PHRASE in message] == []


def _run_reusing(
    global_filter: GlobalFilter,
    served: set[str],
    mapping: dict[str, PropertySpec] | None = None,
    host_options: Options | None = None,
) -> None:
    """One engine run of FDG_HOST reusing the caller's GlobalFilter; the caller deletes its own
    GlobalFilter reference before asserting, since the collection pins attached probe classes."""
    collector = PluginCollector.enabled_feature_groups({_engine_probe(served, mapping)})
    results = mloda.run_all(
        [Feature(FDG_HOST, host_options or Options())],
        compute_frameworks={PythonDictFramework},
        plugin_collector=collector,
        global_filter=global_filter,
    )
    del collector, results


def test_unmatched_tracking_is_scoped_to_a_single_engine_run(caplog: pytest.LogCaptureFixture) -> None:
    """Reusing one GlobalFilter: a match in run 1 must not silence run 2's unmatched warning."""
    global_filter = GlobalFilter()
    global_filter.add_filter(Feature(FDG_RERUN_TARGET, Options()), FilterType.EQUAL, {"value": 1})
    with caplog.at_level(logging.WARNING):
        _run_reusing(global_filter, {FDG_HOST, FDG_RERUN_TARGET})
    first = [message for message in _warnings_naming(caplog, FDG_RERUN_TARGET) if UNMATCHED_PHRASE in message]
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _run_reusing(global_filter, {FDG_HOST})
    del global_filter

    assert first == [], f"the filter attached in run 1, so run 1 must stay silent: {first!r}"
    second = [message for message in _warnings_naming(caplog, FDG_RERUN_TARGET) if UNMATCHED_PHRASE in message]
    assert len(second) == 1, f"run 2 matched nothing, exactly one warning expected, got: {second!r}"


def test_identical_divergence_message_is_emitted_once_per_setup(caplog: pytest.LogCaptureFixture) -> None:
    """Three attachments rendering one byte-identical divergence message must warn once."""
    global_filter = GlobalFilter()
    global_filter.add_filter(
        Feature(FDG_TARGET, Options(group={FDG_DEDUP_KEY: FDG_FILTER_VAL})), FilterType.EQUAL, {"value": 1}
    )
    probe = _accepting_probe()
    with caplog.at_level(logging.WARNING):
        for host in ("fdg_dedup_host_a", "fdg_dedup_host_b", "fdg_dedup_host_c"):
            global_filter.identify_matched_filters(probe, Feature(host, Options(group={FDG_DEDUP_KEY: FDG_HOST_VAL})))
    del probe, global_filter

    assert _warnings_naming(caplog, FDG_DEDUP_KEY) == [
        f"Options are not the same. {FDG_DEDUP_KEY} is different. {FDG_FILTER_VAL} != {FDG_HOST_VAL}"
    ]


def test_divergence_dedup_keeps_distinct_messages(caplog: pytest.LogCaptureFixture) -> None:
    """The control: two hosts rendering two different messages must both be reported."""
    global_filter = GlobalFilter()
    global_filter.add_filter(
        Feature(FDG_TARGET, Options(group={FDG_DEDUP_KEY: FDG_FILTER_VAL})), FilterType.EQUAL, {"value": 1}
    )
    probe = _accepting_probe()
    with caplog.at_level(logging.WARNING):
        global_filter.identify_matched_filters(
            probe, Feature("fdg_dedup_host_a", Options(group={FDG_DEDUP_KEY: FDG_HOST_VAL}))
        )
        global_filter.identify_matched_filters(
            probe, Feature("fdg_dedup_host_b", Options(group={FDG_DEDUP_KEY: FDG_OTHER_HOST_VAL}))
        )
    del probe, global_filter

    assert sorted(_warnings_naming(caplog, FDG_DEDUP_KEY)) == sorted(
        [
            f"Options are not the same. {FDG_DEDUP_KEY} is different. {FDG_FILTER_VAL} != {FDG_HOST_VAL}",
            f"Options are not the same. {FDG_DEDUP_KEY} is different. {FDG_FILTER_VAL} != {FDG_OTHER_HOST_VAL}",
        ]
    )


def test_divergence_dedup_is_scoped_to_a_single_engine_run(caplog: pytest.LogCaptureFixture) -> None:
    """Reusing one GlobalFilter: each engine run re-emits an identical divergence message once."""
    mapping = {FDG_DEDUP_KEY: PropertySpec("A group key the filter feature declares differently.", context=False)}
    host_options = Options(group={FDG_DEDUP_KEY: FDG_HOST_VAL})
    global_filter = GlobalFilter()
    global_filter.add_filter(
        Feature(FDG_TARGET, Options(group={FDG_DEDUP_KEY: FDG_FILTER_VAL})), FilterType.EQUAL, {"value": 1}
    )
    with caplog.at_level(logging.WARNING):
        _run_reusing(global_filter, {FDG_HOST, FDG_TARGET}, mapping, host_options)
    first = _warnings_naming(caplog, FDG_DEDUP_KEY)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _run_reusing(global_filter, {FDG_HOST, FDG_TARGET}, mapping, host_options)
    del global_filter

    expected = f"Options are not the same. {FDG_DEDUP_KEY} is different. {FDG_FILTER_VAL} != {FDG_HOST_VAL}"
    assert first == [expected], f"run 1 must emit the divergence message once: {first!r}"
    second = _warnings_naming(caplog, FDG_DEDUP_KEY)
    assert second == [expected], f"run 2 must emit the divergence message once again: {second!r}"
