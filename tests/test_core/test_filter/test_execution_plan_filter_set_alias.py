"""Pin ``ExecutionPlan.add_single_filters_to_feature_set`` copying the filter set it hands out (#910).

``relevant_filters = single_filters`` aliases the live ``GlobalFilter.collection`` set object into
``FeatureSet.filters`` instead of copying it, so a planned ``FeatureSet`` keeps a live view of a
mutable public set. Preparing a second session against the same ``GlobalFilter`` grows that set and
retroactively changes what the first session's already-planned feature group receives.

This is a separate defect from ``SingleFilter`` aliasing its filter feature: it survives that fix and
needs its own copy at the handover.
"""

from __future__ import annotations

from typing import Any

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.provider import DataCreator
from mloda.user import Feature, FilterType, GlobalFilter, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


# The epa_ prefix keeps every feature name unique to this module.
EPA_ROOT = "epa_root"
EPA_TARGET_A = "epa_target_a"
EPA_TARGET_B = "epa_target_b"


def _make_fg() -> type[FeatureGroup]:
    """A throwaway root feature group serving the host and both filter targets."""

    class EpaHostFeatureGroup(FeatureGroup):
        @classmethod
        def input_data(cls) -> DataCreator:
            return DataCreator({EPA_ROOT, EPA_TARGET_A, EPA_TARGET_B})

        @classmethod
        def final_filters(cls) -> bool:
            # The payload is not filterable data: report features.filters inline instead of running
            # post-calculation row elimination against it.
            return False

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            delivered = len(features.filters) if features.filters else 0
            return {str(feature.name): [delivered] for feature in features.features}

    return EpaHostFeatureGroup


def _delivered(results: list[Any]) -> list[int]:
    """The filter count each frame reports for the host column."""
    return [frame[EPA_ROOT][0] for frame in results if EPA_ROOT in frame]


def _run_two_sessions() -> dict[str, Any]:
    """Prepare and run two sessions against one GlobalFilter, then re-run the first one.

    Every object referencing the throwaway feature group is dropped from this frame before returning,
    so a failing assert in the caller cannot pin it into a traceback and trip the no-leak fixture.
    """
    collector = PluginCollector.enabled_feature_groups({_make_fg()})
    global_filter = GlobalFilter()
    global_filter.add_filter(Feature(EPA_TARGET_A), FilterType.EQUAL, {"value": 1})

    first = mloda.prepare(
        [EPA_ROOT],
        compute_frameworks={PythonDictFramework},
        plugin_collector=collector,
        global_filter=global_filter,
    )
    observed: dict[str, Any] = {"first_run": _delivered(first.run())}

    # A second filter on the same host, planned into a second session only.
    global_filter.add_filter(Feature(EPA_TARGET_B), FilterType.EQUAL, {"value": 2})
    second = mloda.prepare(
        [EPA_ROOT],
        compute_frameworks={PythonDictFramework},
        plugin_collector=collector,
        global_filter=global_filter,
    )
    observed["second_run"] = _delivered(second.run())
    observed["first_rerun"] = _delivered(first.run())

    del collector, global_filter, first, second
    return observed


def test_a_second_session_does_not_change_the_first_session_plan() -> None:
    """The first session keeps the one filter it was prepared with (fails pre-fix: its FeatureSet
    aliases the live collection set, so the second session's filter appears in it retroactively)."""
    observed = _run_two_sessions()
    assert observed["first_run"] == [1], f"the first session must be planned with one filter: {observed!r}"
    assert observed["first_rerun"] == [1], f"the first session's plan must not change: {observed!r}"


def test_the_second_session_sees_both_filters() -> None:
    """Guard, passes pre-fix: the session prepared after the second add_filter plans both filters."""
    observed = _run_two_sessions()
    assert observed["second_run"] == [2], f"the second session must be planned with both filters: {observed!r}"
