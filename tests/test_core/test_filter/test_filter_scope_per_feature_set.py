"""Issue #928: filters are probed per feature but attached per FeatureSet, so a decline is silent.

Two features of one group that differ only in a plain context option share one FeatureSet. When the
group declines the filter while probing feature A and accepts it while probing feature B, the shared
set still receives the filter, so A is filtered too. The scope stays per FeatureSet; the divergence
must become observable as a WARNING, and moving the deciding option to the group split removes it.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Any, Optional

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.provider import DataCreator
from mloda.user import Feature, FeatureName, FilterType, GlobalFilter, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


EP_LOGGER_NAME = "mloda.core.prepare.execution_plan"

# The fsp_ prefix keeps every feature name unique to this module.
FSP_DECLINE = "fsp_decline_feat_928"  # probing this feature makes the group decline the filter
FSP_ACCEPT = "fsp_accept_feat_928"  # probing this feature makes the group accept the filter
FSP_FILTER = "fsp_filter_feat_928"  # the filter feature whose match hook answers per probing feature
FSP_MODE_KEY = "fsp_mode_928"  # "a" declines, "b" accepts

ALL_FSP_NAMES = (FSP_DECLINE, FSP_ACCEPT, FSP_FILTER)


def _sentinel(features: FeatureSet) -> dict[str, list[int]]:
    """One row per served feature carrying the number of filters the set was handed (-1 for None)."""
    delivered = -1 if features.filters is None else len(features.filters)
    return {str(feature.name): [delivered] for feature in features.features}


def _make_fg() -> type[FeatureGroup]:
    """A throwaway root group whose filter-feature match depends on the mode carried by the probing feature."""
    gc.collect()

    class FspScopeFG928(FeatureGroup):
        @classmethod
        def input_data(cls) -> DataCreator:
            return DataCreator({FSP_DECLINE, FSP_ACCEPT, FSP_FILTER})

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: Optional[DataAccessCollection] = None,
        ) -> bool:
            if str(feature_name) == FSP_FILTER:
                # identify_matched_filters enriched the filter feature with the probing feature's options.
                return bool(options.get(FSP_MODE_KEY) == "b")
            return str(feature_name) in {FSP_DECLINE, FSP_ACCEPT}

        @classmethod
        def final_filters(cls) -> bool:
            # The payload is a sentinel, not filterable data: report features.filters inline instead.
            return False

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return _sentinel(features)

    return FspScopeFG928


def _filter_on_fsp() -> GlobalFilter:
    """A GlobalFilter carrying one filter on the filter feature."""
    global_filter = GlobalFilter()
    global_filter.add_filter(Feature(FSP_FILTER), FilterType.EQUAL, {"value": 1})
    return global_filter


def _options(mode: str, in_group: bool) -> Options:
    """The deciding option, either as a plain context option or as a group option."""
    return Options(group={FSP_MODE_KEY: mode}) if in_group else Options(context={FSP_MODE_KEY: mode})


@dataclass(frozen=True)
class _RunSnapshot:
    """Plain-data readout of one run: sentinels, divergence warnings, probe counts. Holds no class."""

    sentinels: dict[str, int]
    warnings: tuple[str, ...]
    probe_counts: Optional[dict[str, int]]


def _sentinels(results: list[Any], columns: tuple[str, ...]) -> dict[str, int]:
    """Map each requested column to the sentinel reported by the frame carrying it."""
    found = {column: frame[column][0] for frame in results for column in columns if column in frame}
    missing = [column for column in columns if column not in found]
    assert not missing, f"no result frame carries {missing}: {results!r}"
    return found


def _fsp_warnings(caplog: pytest.LogCaptureFixture) -> tuple[str, ...]:
    """WARNINGs the execution plan emitted about this module's feature names."""
    return tuple(
        record.getMessage()
        for record in caplog.records
        if record.name == EP_LOGGER_NAME
        and record.levelno == logging.WARNING
        and any(name in record.getMessage() for name in ALL_FSP_NAMES)
    )


def _probe_counts(global_filter: GlobalFilter) -> Optional[dict[str, int]]:
    """Per probed feature name, how many filters that probe matched. None when the API does not exist."""
    probes = getattr(global_filter, "probes", None)
    if probes is None:
        return None
    # The key holds the feature group class: read the name only, never bind the class.
    return {str(key[1]): len(matched) for key, matched in probes.items()}


def _run(mode_in_group: bool, caplog: pytest.LogCaptureFixture) -> _RunSnapshot:
    """Run both features in one session and collect sentinels, warnings and probes as plain data.

    Every object referencing the throwaway feature group is dropped before returning, so a failing
    assert in the caller cannot pin it into a traceback and trip the no-leak fixture.
    """
    caplog.clear()
    fg = _make_fg()
    collector = PluginCollector.enabled_feature_groups({fg})
    global_filter = _filter_on_fsp()

    with caplog.at_level(logging.WARNING, logger=EP_LOGGER_NAME):
        results = mloda.run_all(
            [
                Feature(FSP_DECLINE, _options("a", mode_in_group)),
                Feature(FSP_ACCEPT, _options("b", mode_in_group)),
            ],
            compute_frameworks={PythonDictFramework},
            plugin_collector=collector,
            global_filter=global_filter,
        )

    snapshot = _RunSnapshot(
        sentinels=_sentinels(results, (FSP_DECLINE, FSP_ACCEPT)),
        warnings=_fsp_warnings(caplog),
        probe_counts=_probe_counts(global_filter),
    )
    # Records carry args that can hold the throwaway class; drop them with everything else.
    caplog.clear()
    del fg, collector, global_filter, results
    gc.collect()
    return snapshot


def test_the_declining_feature_is_filtered_anyway(caplog: pytest.LogCaptureFixture) -> None:
    """Characterization: the shared FeatureSet gets the filter, so the declining feature is filtered too."""
    snapshot = _run(mode_in_group=False, caplog=caplog)

    assert snapshot.sentinels[FSP_ACCEPT] == 1, f"the accepting feature must get its one filter: {snapshot!r}"
    assert snapshot.sentinels[FSP_DECLINE] == 1, (
        f"the decline does not suppress the filter on the shared feature set: {snapshot!r}"
    )


def test_the_divergence_is_reported_as_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    """The silent widening must be observable: one WARNING naming the declining feature and the filter."""
    snapshot = _run(mode_in_group=False, caplog=caplog)

    assert len(snapshot.warnings) == 1, f"exactly one WARNING must report the divergence, got: {snapshot.warnings}"
    message = snapshot.warnings[0]
    assert FSP_DECLINE in message, f"the warning must name the declining feature: {message}"
    assert FSP_FILTER in message, f"the warning must name the filter feature: {message}"


def test_every_probe_is_recorded_including_the_empty_one(caplog: pytest.LogCaptureFixture) -> None:
    """The decline is only detectable if empty probe results are recorded, not just the matches."""
    snapshot = _run(mode_in_group=False, caplog=caplog)

    assert snapshot.probe_counts is not None, "GlobalFilter must expose a probes ledger"
    assert snapshot.probe_counts.get(FSP_DECLINE) == 0, f"the declining probe must be recorded as empty: {snapshot!r}"
    assert snapshot.probe_counts.get(FSP_ACCEPT) == 1, f"the accepting probe must record its filter: {snapshot!r}"


def test_a_group_option_splits_the_sets_and_spares_the_declining_feature(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The documented remedy: as a group option the mode splits the feature sets, so only one is filtered."""
    snapshot = _run(mode_in_group=True, caplog=caplog)

    assert snapshot.sentinels[FSP_ACCEPT] == 1, f"the accepting feature set must get its one filter: {snapshot!r}"
    assert snapshot.sentinels[FSP_DECLINE] == 0, (
        f"the declining feature set must get an empty filter set, not the filter: {snapshot!r}"
    )


def test_the_uniform_case_warns_about_nothing(caplog: pytest.LogCaptureFixture) -> None:
    """Every feature of a set matched the same filters, so there is no divergence to report."""
    snapshot = _run(mode_in_group=True, caplog=caplog)

    assert snapshot.warnings == (), f"a uniform feature set must not warn: {snapshot.warnings}"
