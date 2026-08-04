"""Sibling features of one FeatureSet may match different filters: the set gets the union,
enrichment-only differences stay silent, and each genuinely unmatched filter warns per feature.
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

# The fsd_ prefix keeps every name unique to this module.
FSD_A = "fsd_feat_a"
FSD_B = "fsd_feat_b"
FSD_SHARED_FILTER = "fsd_filter_shared"  # every probe matches it
FSD_X_FILTER = "fsd_filter_x"  # matched only for mode "a"
FSD_Y_FILTER = "fsd_filter_y"  # matched only for mode "b"
FSD_MODE_KEY = "fsd_mode"
FSD_NOTE_KEY = "fsd_note"  # unrelated context key the match hook never reads

SERVED_FSD_NAMES = (FSD_A, FSD_B)
ALL_FSD_NAMES = SERVED_FSD_NAMES + (FSD_SHARED_FILTER, FSD_X_FILTER, FSD_Y_FILTER)


def _sentinel(features: FeatureSet) -> dict[str, list[int]]:
    """One row per served feature carrying the number of filters the set was handed (-1 for None)."""
    delivered = -1 if features.filters is None else len(features.filters)
    return {str(feature.name): [delivered] for feature in features.features}


def _make_fg(capture: list[tuple[Feature, ...]]) -> type[FeatureGroup]:
    """A throwaway root group whose per-filter matches depend on the probing feature's mode."""
    gc.collect()

    class FsdDivergenceFG(FeatureGroup):
        @classmethod
        def input_data(cls) -> DataCreator:
            return DataCreator(set(ALL_FSD_NAMES))

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: Optional[DataAccessCollection] = None,
        ) -> bool:
            name = str(feature_name)
            if name == FSD_SHARED_FILTER:
                return True
            if name == FSD_X_FILTER:
                return bool(options.get(FSD_MODE_KEY) == "a")
            if name == FSD_Y_FILTER:
                return bool(options.get(FSD_MODE_KEY) == "b")
            return name in SERVED_FSD_NAMES

        @classmethod
        def final_filters(cls) -> bool:
            # The payload is a sentinel, not filterable data.
            return False

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            capture.append(tuple(features.features))
            return _sentinel(features)

    return FsdDivergenceFG


def _filters_on(*names: str) -> GlobalFilter:
    """A GlobalFilter carrying one equality filter per given filter feature name."""
    global_filter = GlobalFilter()
    for name in names:
        global_filter.add_filter(Feature(name), FilterType.EQUAL, {"value": 1})
    return global_filter


def _pair(key: str, value_a: str, value_b: str) -> list[Feature | str]:
    """The two sibling features, identical group options, one context value each."""
    return [Feature(FSD_A, Options(context={key: value_a})), Feature(FSD_B, Options(context={key: value_b}))]


@dataclass(frozen=True)
class _RunSnapshot:
    """Plain-data readout of one run. Holds no reference to the throwaway group."""

    sentinels: dict[str, int]
    warnings: tuple[str, ...]
    set_shapes: tuple[tuple[str, ...], ...]


def _sentinels(results: list[Any], columns: tuple[str, ...]) -> dict[str, int]:
    """Map each requested column to the sentinel reported by the frame carrying it."""
    found = {column: frame[column][0] for frame in results for column in columns if column in frame}
    missing = [column for column in columns if column not in found]
    assert not missing, f"no result frame carries {missing}: {results!r}"
    return found


def _fsd_warnings(caplog: pytest.LogCaptureFixture) -> tuple[str, ...]:
    """WARNINGs the execution plan emitted about this module's feature names."""
    return tuple(
        record.getMessage()
        for record in caplog.records
        if record.name == EP_LOGGER_NAME
        and record.levelno == logging.WARNING
        and any(name in record.getMessage() for name in ALL_FSD_NAMES)
    )


def _set_shapes(captured: list[tuple[Feature, ...]]) -> tuple[tuple[str, ...], ...]:
    """The served feature names of every planned FeatureSet, one sorted tuple per set."""
    served = (tuple(sorted(str(feature.name) for feature in features)) for features in captured)
    return tuple(sorted(served))


def _run(
    features: list[Feature | str],
    global_filter: GlobalFilter,
    caplog: pytest.LogCaptureFixture,
) -> _RunSnapshot:
    """Run in one session, then drop every reference to the throwaway group before returning."""
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
        sentinels=_sentinels(results, SERVED_FSD_NAMES),
        warnings=_fsd_warnings(caplog),
        set_shapes=_set_shapes(captured),
    )
    # Records carry args that can hold the throwaway class.
    caplog.clear()
    del fg, collector, results, captured
    gc.collect()
    return snapshot


def test_enrichment_only_divergence_is_silent(caplog: pytest.LogCaptureFixture) -> None:
    """Differing unrelated context enrichment is no divergence: one filter, no warning."""
    snapshot = _run(_pair(FSD_NOTE_KEY, "x", "y"), _filters_on(FSD_SHARED_FILTER), caplog)

    assert len(snapshot.set_shapes) == 1, f"the siblings must share one feature set: {snapshot.set_shapes}"
    assert {FSD_A, FSD_B} <= set(snapshot.set_shapes[0]), f"both siblings must be in it: {snapshot.set_shapes}"
    assert snapshot.sentinels == {FSD_A: 1, FSD_B: 1}, f"both siblings must see the one filter once: {snapshot!r}"
    assert snapshot.warnings == (), f"an enrichment-only difference must not warn: {snapshot.warnings}"


def test_genuine_divergence_attaches_the_union_and_warns_per_feature(caplog: pytest.LogCaptureFixture) -> None:
    """Each sibling matched one filter, gets both, and is warned about the other one."""
    snapshot = _run(_pair(FSD_MODE_KEY, "a", "b"), _filters_on(FSD_X_FILTER, FSD_Y_FILTER), caplog)

    assert len(snapshot.set_shapes) == 1, f"the siblings must share one feature set: {snapshot.set_shapes}"
    assert {FSD_A, FSD_B} <= set(snapshot.set_shapes[0]), f"both siblings must be in it: {snapshot.set_shapes}"
    assert snapshot.sentinels == {FSD_A: 2, FSD_B: 2}, f"the union of both filters must be attached: {snapshot!r}"
    assert len(snapshot.warnings) == 2, f"one warning per sibling must report its unmatched filter: {snapshot.warnings}"
    warnings_a = [message for message in snapshot.warnings if FSD_A in message]
    warnings_b = [message for message in snapshot.warnings if FSD_B in message]
    assert len(warnings_a) == 1, f"exactly one warning must name {FSD_A}: {snapshot.warnings}"
    assert len(warnings_b) == 1, f"exactly one warning must name {FSD_B}: {snapshot.warnings}"
    assert FSD_Y_FILTER in warnings_a[0], f"the warning for {FSD_A} must name the filter it did not match: {warnings_a}"
    assert FSD_X_FILTER in warnings_b[0], f"the warning for {FSD_B} must name the filter it did not match: {warnings_b}"
    for message in snapshot.warnings:
        assert "still applies" in message, f"the scope warning must explain that the filter still applies: {message}"


def test_the_shared_filter_is_not_reported_as_divergence(caplog: pytest.LogCaptureFixture) -> None:
    """Only the genuinely unmatched filter warns; the shared one stays out of the report."""
    snapshot = _run(_pair(FSD_MODE_KEY, "a", "b"), _filters_on(FSD_SHARED_FILTER, FSD_Y_FILTER), caplog)

    assert len(snapshot.set_shapes) == 1, f"the siblings must share one feature set: {snapshot.set_shapes}"
    assert {FSD_A, FSD_B} <= set(snapshot.set_shapes[0]), f"both siblings must be in it: {snapshot.set_shapes}"
    assert snapshot.sentinels == {FSD_A: 2, FSD_B: 2}, f"both filters must be attached once each: {snapshot!r}"
    assert len(snapshot.warnings) == 1, f"only the excluded filter must be reported, once: {snapshot.warnings}"
    message = snapshot.warnings[0]
    assert FSD_A in message, f"the warning must name the sibling that did not match: {message}"
    assert FSD_Y_FILTER in message, f"the warning must name the genuinely unmatched filter: {message}"
    assert FSD_SHARED_FILTER not in message, f"the shared filter was matched and must not be reported: {message}"
    assert FSD_B not in message, f"the fully matched sibling must not be warned about: {message}"
