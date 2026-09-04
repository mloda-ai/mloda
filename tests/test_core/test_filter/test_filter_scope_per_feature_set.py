"""Filters are probed per feature but attached per FeatureSet, so a decline is silently widened.
The scope stays per FeatureSet; the divergence must be reported as a WARNING, once, per probed feature.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.provider import DataCreator
from mloda.user import Feature, FeatureName, FilterType, GlobalFilter, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


EP_LOGGER_NAME = "mloda.core.prepare.execution_plan"

# The fsp_ prefix keeps every feature name unique to this module.
FSP_DECLINE = "fsp_decline_feat"  # probing this feature makes the group decline the filter
FSP_ACCEPT = "fsp_accept_feat"  # probing this feature makes the group accept the filter
FSP_FILTER = "fsp_filter_feat"  # the filter feature whose match hook answers per probing feature
FSP_SAME = "fsp_same_feat"  # requested twice, once per mode: one name, two features, one set
FSP_UNIFORM_A = "fsp_uniform_a_feat"  # two different names, identical options, both accepting
FSP_UNIFORM_B = "fsp_uniform_b_feat"
FSP_MODE_KEY = "fsp_mode"  # "a" declines, "b" accepts

SERVED_FSP_NAMES = (FSP_DECLINE, FSP_ACCEPT, FSP_SAME, FSP_UNIFORM_A, FSP_UNIFORM_B)
ALL_FSP_NAMES = SERVED_FSP_NAMES + (FSP_FILTER,)


def _sentinel(features: FeatureSet) -> dict[str, list[int]]:
    """One row per served feature carrying the number of filters the set was handed (-1 for None)."""
    delivered = -1 if features.filters is None else len(features.filters)
    return {str(feature.name): [delivered] for feature in features.features}


def _make_fg(capture: list[tuple[Feature, ...]] | None = None) -> type[FeatureGroup]:
    """A throwaway root group whose filter-feature match depends on the probing feature's mode."""
    gc.collect()

    class FspScopeFG(FeatureGroup):
        @classmethod
        def input_data(cls) -> DataCreator:
            return DataCreator(set(ALL_FSP_NAMES))

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: DataAccessCollection | None = None,
        ) -> bool:
            if str(feature_name) == FSP_FILTER:
                # identify_matched_filters enriched the filter feature with the probing feature's options.
                return bool(options.get(FSP_MODE_KEY) == "b")
            return str(feature_name) in SERVED_FSP_NAMES

        @classmethod
        def final_filters(cls) -> bool:
            # The payload is a sentinel, not filterable data.
            return False

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            if capture is not None:
                capture.append(tuple(features.features))
            return _sentinel(features)

    return FspScopeFG


def _filter_on_fsp() -> GlobalFilter:
    """A GlobalFilter carrying one filter on the filter feature."""
    global_filter = GlobalFilter()
    global_filter.add_filter(Feature(FSP_FILTER), FilterType.EQUAL, {"value": 1})
    return global_filter


def _options(mode: str, in_group: bool = False) -> Options:
    """The deciding option, either as a plain context option or as a group option."""
    return Options(group={FSP_MODE_KEY: mode}) if in_group else Options(context={FSP_MODE_KEY: mode})


def _mode_pair(in_group: bool) -> list[Feature | str]:
    """The declining and the accepting feature, two names, one mode each."""
    return [Feature(FSP_DECLINE, _options("a", in_group)), Feature(FSP_ACCEPT, _options("b", in_group))]


@dataclass(frozen=True)
class _RunSnapshot:
    """Plain-data readout of one run. Holds no reference to the throwaway group."""

    sentinels: dict[str, int]
    warnings: tuple[str, ...]
    probe_entries: dict[str, tuple[int, ...]]
    set_shapes: tuple[tuple[str, ...], ...]


def _sentinels(results: list[Any], columns: tuple[str, ...]) -> dict[str, int]:
    """Map each requested column to the sentinel reported by the frame carrying it."""
    found = {column: frame[column][0] for frame in results for column in columns if column in frame}
    missing = [column for column in columns if column not in found]
    assert not missing, f"no result frame carries {missing}: {results!r}"
    return found


def _fsp_records(caplog: pytest.LogCaptureFixture) -> tuple[tuple[int, str], ...]:
    """Level and message of every execution-plan record naming one of this module's features."""
    return tuple(
        (record.levelno, record.getMessage())
        for record in caplog.records
        if record.name == EP_LOGGER_NAME and any(name in record.getMessage() for name in ALL_FSP_NAMES)
    )


def _fsp_warnings(caplog: pytest.LogCaptureFixture) -> tuple[str, ...]:
    """WARNINGs the execution plan emitted about this module's feature names."""
    return tuple(message for level, message in _fsp_records(caplog) if level == logging.WARNING)


def _probe_entries(global_filter: GlobalFilter) -> dict[str, tuple[int, ...]]:
    """Per probed feature name, the sorted match counts of its probe entries."""
    entries: dict[str, list[int]] = {}
    # The key holds the feature group class: read the name only, never bind the class.
    for key, matched in global_filter.probes.items():
        entries.setdefault(str(key[1]), []).append(len(matched))
    return {name: tuple(sorted(counts)) for name, counts in entries.items()}


def _set_shapes(captured: list[tuple[Feature, ...]]) -> tuple[tuple[str, ...], ...]:
    """The served feature names of every planned FeatureSet, one sorted tuple per set."""
    served = (tuple(sorted(str(feature.name) for feature in features)) for features in captured)
    return tuple(sorted(served))


def _run(
    features: list[Feature | str],
    columns: tuple[str, ...],
    global_filter: GlobalFilter,
    caplog: pytest.LogCaptureFixture,
) -> _RunSnapshot:
    """Run the features in one session and collect the readout as plain data.

    Everything referencing the throwaway group is dropped before returning, so a failing assert
    cannot pin it into a traceback and trip the no-leak fixture.
    """
    caplog.clear()
    captured: list[tuple[Feature, ...]] = []
    fg = _make_fg(captured)
    collector = PluginCollector.enabled_feature_groups({fg})

    with caplog.at_level(logging.WARNING, logger=EP_LOGGER_NAME):
        results = mloda.run_all(
            features,
            compute_frameworks={PythonDictFramework},
            plugin_collector=collector,
            global_filter=global_filter,
        )

    snapshot = _RunSnapshot(
        sentinels=_sentinels(results, columns),
        warnings=_fsp_warnings(caplog),
        probe_entries=_probe_entries(global_filter),
        set_shapes=_set_shapes(captured),
    )
    # Records carry args that can hold the throwaway class.
    caplog.clear()
    del fg, collector, results, captured
    gc.collect()
    return snapshot


def test_the_declining_feature_is_filtered_anyway(caplog: pytest.LogCaptureFixture) -> None:
    """The shared FeatureSet gets the filter, so the declining feature is filtered too."""
    snapshot = _run(_mode_pair(False), (FSP_DECLINE, FSP_ACCEPT), _filter_on_fsp(), caplog)

    assert snapshot.sentinels[FSP_ACCEPT] == 1, f"the accepting feature must get its one filter: {snapshot!r}"
    assert snapshot.sentinels[FSP_DECLINE] == 1, (
        f"the decline does not suppress the filter on the shared feature set: {snapshot!r}"
    )


def test_the_divergence_is_reported_as_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    """One WARNING names the declining feature and the filter."""
    snapshot = _run(_mode_pair(False), (FSP_DECLINE, FSP_ACCEPT), _filter_on_fsp(), caplog)

    assert len(snapshot.warnings) == 1, f"exactly one WARNING must report the divergence, got: {snapshot.warnings}"
    message = snapshot.warnings[0]
    assert FSP_DECLINE in message, f"the warning must name the declining feature: {message}"
    assert FSP_FILTER in message, f"the warning must name the filter feature: {message}"


def test_every_probe_is_recorded_including_the_empty_one(caplog: pytest.LogCaptureFixture) -> None:
    """Empty probe results are recorded too, not just the matches."""
    snapshot = _run(_mode_pair(False), (FSP_DECLINE, FSP_ACCEPT), _filter_on_fsp(), caplog)

    assert snapshot.probe_entries.get(FSP_DECLINE) == (0,), (
        f"the declining probe must be recorded as empty: {snapshot!r}"
    )
    assert snapshot.probe_entries.get(FSP_ACCEPT) == (1,), f"the accepting probe must record its filter: {snapshot!r}"


def test_two_features_of_one_name_diverge_and_are_reported(caplog: pytest.LogCaptureFixture) -> None:
    """One name requested with two context modes: the declining twin is still reported."""
    same_name: list[Feature | str] = [Feature(FSP_SAME, _options("a")), Feature(FSP_SAME, _options("b"))]
    snapshot = _run(same_name, (FSP_SAME,), _filter_on_fsp(), caplog)

    assert len(snapshot.set_shapes) == 1, f"the two twins must share one feature set: {snapshot.set_shapes}"
    assert snapshot.set_shapes[0].count(FSP_SAME) == 2, (
        f"both twins must be planned into that set: {snapshot.set_shapes}"
    )
    assert len(snapshot.warnings) == 1, (
        f"the declining twin of one name must be reported exactly once: {snapshot.warnings}, probes {snapshot!r}"
    )
    message = snapshot.warnings[0]
    assert FSP_SAME in message, f"the warning must name the diverging feature: {message}"
    assert FSP_FILTER in message, f"the warning must name the filter feature: {message}"


def _cross_run_warnings(caplog: pytest.LogCaptureFixture) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Two runs of one name over a shared GlobalFilter: run 1 accepts, run 2 declines."""
    caplog.clear()
    fg = _make_fg()
    collector = PluginCollector.enabled_feature_groups({fg})
    global_filter = _filter_on_fsp()

    with caplog.at_level(logging.WARNING, logger=EP_LOGGER_NAME):
        mloda.run_all(
            [Feature(FSP_SAME, _options("b"))],
            compute_frameworks={PythonDictFramework},
            plugin_collector=collector,
            global_filter=global_filter,
        )
        first = _fsp_warnings(caplog)
        caplog.clear()
        mloda.run_all(
            [Feature(FSP_SAME, _options("a"))],
            compute_frameworks={PythonDictFramework},
            plugin_collector=collector,
            global_filter=global_filter,
        )
        second = _fsp_warnings(caplog)

    caplog.clear()
    del fg, collector, global_filter
    gc.collect()
    return first, second


def test_a_stale_probe_of_an_earlier_run_does_not_mask_a_later_decline(caplog: pytest.LogCaptureFixture) -> None:
    """The second run warns about the stale filter its earlier collection entry still attaches."""
    first, second = _cross_run_warnings(caplog)

    assert first == (), f"the accepting run must not warn: {first}"
    assert len(second) == 1, f"the declining run must report the stale filter it is handed: {second}"
    assert FSP_SAME in second[0], f"the warning must name the declining feature: {second[0]}"
    assert FSP_FILTER in second[0], f"the warning must name the filter feature: {second[0]}"


def test_a_filterless_run_records_no_probes(caplog: pytest.LogCaptureFixture) -> None:
    """An empty GlobalFilter has nothing to probe, so the ledger stays empty."""
    snapshot = _run(_mode_pair(False), (FSP_DECLINE, FSP_ACCEPT), GlobalFilter(), caplog)

    assert snapshot.sentinels == {FSP_DECLINE: -1, FSP_ACCEPT: -1}, f"no filter can be delivered: {snapshot!r}"
    assert snapshot.probe_entries == {}, f"a run without filters must record no probe: {snapshot.probe_entries}"


def _repeated_report_levels(caplog: pytest.LogCaptureFixture) -> tuple[tuple[int, str], ...]:
    """Divergence reports for one feature planned into two feature sets.

    A genuine two-FeatureSet plan is not reachable within the timeout, so one real run populates the
    GlobalFilter and the plan step is then replayed twice over the diverging set.
    """
    caplog.clear()
    captured: list[tuple[Feature, ...]] = []
    fg = _make_fg(captured)
    collector = PluginCollector.enabled_feature_groups({fg})
    global_filter = _filter_on_fsp()

    with caplog.at_level(logging.DEBUG, logger=EP_LOGGER_NAME):
        mloda.run_all(
            _mode_pair(False),
            compute_frameworks={PythonDictFramework},
            plugin_collector=collector,
            global_filter=global_filter,
        )

        diverging = [features for features in captured if any(str(feature.name) == FSP_DECLINE for feature in features)]
        assert len(diverging) == 1, f"the run must plan exactly one set holding the declining feature: {diverging!r}"

        plan = ExecutionPlan(global_filter=global_filter)
        caplog.clear()
        plan.add_single_filters_to_feature_set(fg, FeatureSet(diverging[0]))
        plan.add_single_filters_to_feature_set(fg, FeatureSet(diverging[0]))
        records = _fsp_records(caplog)

    caplog.clear()
    del fg, collector, global_filter, captured, diverging, plan
    gc.collect()
    return records


def test_the_divergence_report_does_not_repeat(caplog: pytest.LogCaptureFixture) -> None:
    """The same divergence warns once, then drops to DEBUG."""
    records = _repeated_report_levels(caplog)

    levels = tuple(level for level, _ in records)
    assert levels == (logging.WARNING, logging.DEBUG), f"first report WARNING, repeat DEBUG, got: {records}"


def test_a_group_option_splits_the_sets_and_spares_the_declining_feature(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """As a group option the mode splits the feature sets, so only one is filtered."""
    snapshot = _run(_mode_pair(True), (FSP_DECLINE, FSP_ACCEPT), _filter_on_fsp(), caplog)

    assert snapshot.sentinels[FSP_ACCEPT] == 1, f"the accepting feature set must get its one filter: {snapshot!r}"
    assert snapshot.sentinels[FSP_DECLINE] == 0, (
        f"the declining feature set must get an empty filter set, not the filter: {snapshot!r}"
    )
    assert len(snapshot.set_shapes) == 2, f"the group option must split the sets: {snapshot.set_shapes}"
    assert snapshot.warnings == (), f"a single-feature set cannot diverge from itself: {snapshot.warnings}"


def test_the_uniform_case_warns_about_nothing(caplog: pytest.LogCaptureFixture) -> None:
    """Two names with identical options share one set and both match, so nothing is reported."""
    uniform: list[Feature | str] = [Feature(FSP_UNIFORM_A, _options("b")), Feature(FSP_UNIFORM_B, _options("b"))]
    snapshot = _run(uniform, (FSP_UNIFORM_A, FSP_UNIFORM_B), _filter_on_fsp(), caplog)

    assert len(snapshot.set_shapes) == 1, f"both names must share one feature set: {snapshot.set_shapes}"
    assert snapshot.sentinels == {FSP_UNIFORM_A: 1, FSP_UNIFORM_B: 1}, (
        f"both features of the uniform set must get the one filter: {snapshot!r}"
    )
    assert snapshot.warnings == (), f"a uniform feature set must not warn: {snapshot.warnings}"


# End-to-end scope pin through engine._add_filter_feature: real rows, one filter scoped to one of two groups.
FSP_SCOPED_TARGET = "fsp_scoped_target_feat"  # served by the scope-named group; its rows get filtered
FSP_OTHER_TARGET = "fsp_other_target_feat"  # served by the sibling group; its rows must stay complete
FSP_SHARED_FILTER = "fsp_shared_filter_feat"  # both groups can serve the filter feature, only the scope decides
FSP_SCOPED_CLASS_NAME = "FspScopedTargetFG"

FSP_ALL_ROWS = [10, 20, 30]
FSP_KEPT_ROWS = [10, 30]
FSP_FILTER_ROWS = [1, 0, 1]  # equal(value=1) keeps rows 0 and 2


def _target_rows(features: FeatureSet) -> dict[str, list[int]]:
    """Fixed rows per served feature; the shared filter column decides which rows survive."""
    rows = {FSP_SCOPED_TARGET: FSP_ALL_ROWS, FSP_OTHER_TARGET: FSP_ALL_ROWS, FSP_SHARED_FILTER: FSP_FILTER_ROWS}
    return {str(feature.name): list(rows[str(feature.name)]) for feature in features.features}


def _make_scoped_target_pair() -> tuple[type[FeatureGroup], type[FeatureGroup]]:
    """Two throwaway root groups: each serves its own target and can serve the shared filter feature."""
    gc.collect()

    class FspScopedTargetFG(FeatureGroup):
        @classmethod
        def input_data(cls) -> DataCreator:
            return DataCreator({FSP_SCOPED_TARGET, FSP_SHARED_FILTER})

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return _target_rows(features)

    class FspOtherTargetFG(FeatureGroup):
        @classmethod
        def input_data(cls) -> DataCreator:
            return DataCreator({FSP_OTHER_TARGET, FSP_SHARED_FILTER})

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return _target_rows(features)

    return FspScopedTargetFG, FspOtherTargetFG


def _drive_scoped_filter_run() -> dict[str, list[int]]:
    """Run both targets with one filter string-scoped to the first group; plain column data comes back."""
    fg_scoped, fg_other = _make_scoped_target_pair()
    collector = PluginCollector.enabled_feature_groups({fg_scoped, fg_other})
    global_filter = GlobalFilter()
    global_filter.add_filter(
        Feature(FSP_SHARED_FILTER, feature_group=FSP_SCOPED_CLASS_NAME), FilterType.EQUAL, {"value": 1}
    )
    requested: list[Feature | str] = [FSP_SCOPED_TARGET, FSP_OTHER_TARGET]

    results = mloda.run_all(
        requested,
        compute_frameworks={PythonDictFramework},
        plugin_collector=collector,
        global_filter=global_filter,
    )

    columns = {
        column: list(frame[column])
        for frame in results
        for column in (FSP_SCOPED_TARGET, FSP_OTHER_TARGET)
        if column in frame
    }
    del fg_scoped, fg_other, collector, global_filter, results
    gc.collect()
    return columns


def test_a_scoped_filter_filters_only_the_scope_named_groups_output() -> None:
    """Through the public API: the scoped group's rows are eliminated, the sibling group's rows stay complete."""
    columns = _drive_scoped_filter_run()

    assert set(columns) == {FSP_SCOPED_TARGET, FSP_OTHER_TARGET}, f"both targets must come back: {columns}"
    assert columns[FSP_SCOPED_TARGET] == FSP_KEPT_ROWS, (
        f"the group inside the scope must have its rows filtered: {columns}"
    )
    assert columns[FSP_OTHER_TARGET] == FSP_ALL_ROWS, f"the group outside the scope must keep every row: {columns}"
