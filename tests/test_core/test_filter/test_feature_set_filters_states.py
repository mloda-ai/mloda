"""Pin the states ``FeatureSet.filters`` can be in when ``calculate_feature`` reads it.

``ExecutionPlan.add_single_filters_to_feature_set`` leaves ``filters`` at ``None`` in exactly two cases:
no ``GlobalFilter`` was passed, or that ``GlobalFilter`` recorded no match at all (its ``collection`` is
empty). Otherwise it always calls ``feature_set.add_filters(...)`` with the filters matched for that one
feature set, which is an EMPTY SET when nothing matched for it.

"Another feature group matched" is therefore too narrow a reading of when the empty set shows up:

- A feature group is planned as one feature set per compute framework, option set, data type and
  dependency level, so "matched for me" is per feature set, not per feature group. One group split into
  two feature sets can have one of them matched and the other one handed an empty set.
- ``GlobalFilter.collection`` accumulates across every run that shares the object, so a later run whose
  groups match nothing still sees a non-empty collection and gets empty sets instead of ``None``.

These are characterization tests: they pin behaviour that exists today.
"""

from __future__ import annotations

from typing import Any

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.provider import DataCreator
from mloda.user import (
    DataAccessCollection,
    Feature,
    FeatureName,
    FilterType,
    GlobalFilter,
    Options,
    PluginCollector,
    mloda,
)
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


# The fss_ prefix keeps every feature name unique to this module.
FSS_FILTERED = "fss_filtered"
FSS_MATCHING = "fss_matching"
FSS_UNRELATED = "fss_unrelated"
FSS_NEVER_SERVED = "fss_never_served"
FSS_SPLIT_FILTERED = "fss_split_filtered"
FSS_SPLIT_TENANT_X = "fss_split_tenant_x"
FSS_SPLIT_TENANT_Y = "fss_split_tenant_y"
FSS_TENANT_KEY = "fss_tenant"

# len() cannot separate None from set(), so calculate_feature reports this sentinel for the None state.
NO_FILTER_SET = -1


def _sentinel(features: FeatureSet) -> dict[str, list[int]]:
    """One row per served feature carrying the filter state: -1 for None, else the number of filters."""
    delivered = NO_FILTER_SET if features.filters is None else len(features.filters)
    return {str(feature.name): [delivered] for feature in features.features}


def _sentinels(results: list[Any], columns: tuple[str, ...]) -> dict[str, int]:
    """Map each column to the sentinel reported by the frame carrying it.

    A column no frame carries is a served-feature regression, so it is named here instead of surfacing
    as a bare KeyError at the assert that wanted to report the filter state.
    """
    found = {column: frame[column][0] for frame in results for column in columns if column in frame}
    missing = [column for column in columns if column not in found]
    assert not missing, f"no result frame carries {missing}: {results!r}"
    return found


def _filter_on(filter_feature: str) -> GlobalFilter:
    """A GlobalFilter carrying one filter on the given column."""
    global_filter = GlobalFilter()
    global_filter.add_filter(Feature(filter_feature), FilterType.EQUAL, {"value": 1})
    return global_filter


def _make_fgs() -> tuple[type[FeatureGroup], type[FeatureGroup]]:
    """Two throwaway root groups: one serves the filtered column, the other one does not."""

    class FssMatchingFeatureGroup(FeatureGroup):
        @classmethod
        def input_data(cls) -> DataCreator:
            return DataCreator({FSS_MATCHING, FSS_FILTERED})

        @classmethod
        def final_filters(cls) -> bool:
            # The payload is a sentinel, not filterable data: report features.filters inline instead of
            # running post-calculation row elimination against it.
            return False

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return _sentinel(features)

    class FssUnrelatedFeatureGroup(FeatureGroup):
        @classmethod
        def input_data(cls) -> DataCreator:
            return DataCreator({FSS_UNRELATED})

        @classmethod
        def final_filters(cls) -> bool:
            return False

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return _sentinel(features)

    return FssMatchingFeatureGroup, FssUnrelatedFeatureGroup


def _make_split_fg() -> type[FeatureGroup]:
    """One throwaway root group serving both tenants, but serving the filtered column for tenant x only."""

    class FssSplitFeatureGroup(FeatureGroup):
        @classmethod
        def input_data(cls) -> DataCreator:
            return DataCreator({FSS_SPLIT_TENANT_X, FSS_SPLIT_TENANT_Y, FSS_SPLIT_FILTERED})

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: DataAccessCollection | None = None,
        ) -> bool:
            # The tenant decides whether this group can serve the filter column, so the filter matches
            # the tenant x feature set only.
            if str(feature_name) == FSS_SPLIT_FILTERED:
                return bool(options.get(FSS_TENANT_KEY) == "x")
            return str(feature_name) in {FSS_SPLIT_TENANT_X, FSS_SPLIT_TENANT_Y}

        @classmethod
        def final_filters(cls) -> bool:
            return False

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return _sentinel(features)

    return FssSplitFeatureGroup


def _run(filter_on: str | None) -> dict[str, int]:
    """Run both groups in one session, asking each for its own column, and collect their sentinels.

    Every object referencing the throwaway feature groups is dropped from this frame before returning, so a
    failing assert in the caller cannot pin them into a traceback and trip the no-leak fixture.
    """
    matching, unrelated = _make_fgs()
    collector = PluginCollector.enabled_feature_groups({matching, unrelated})
    global_filter = None if filter_on is None else _filter_on(filter_on)

    results = mloda.run_all(
        [FSS_MATCHING, FSS_UNRELATED],
        compute_frameworks={PythonDictFramework},
        plugin_collector=collector,
        global_filter=global_filter,
    )
    observed = _sentinels(results, (FSS_MATCHING, FSS_UNRELATED))

    del matching, unrelated, collector, global_filter, results
    return observed


def _run_split() -> dict[str, int]:
    """Run one group as two option-split feature sets, with the filter matching the tenant x set only.

    The two collection counts prove that no second feature group is involved in the empty set below.
    """
    split = _make_split_fg()
    collector = PluginCollector.enabled_feature_groups({split})
    global_filter = _filter_on(FSS_SPLIT_FILTERED)

    results = mloda.run_all(
        [
            Feature(FSS_SPLIT_TENANT_X, Options(group={FSS_TENANT_KEY: "x"})),
            Feature(FSS_SPLIT_TENANT_Y, Options(group={FSS_TENANT_KEY: "y"})),
        ],
        compute_frameworks={PythonDictFramework},
        plugin_collector=collector,
        global_filter=global_filter,
    )
    observed = _sentinels(results, (FSS_SPLIT_TENANT_X, FSS_SPLIT_TENANT_Y))
    observed["collection_entries"] = len(global_filter.collection)
    # Counted, never bound: a collection key holds the feature group class and would pin it.
    observed["collection_groups"] = len({group for group, _ in global_filter.collection})

    del split, collector, global_filter, results
    return observed


def _run_two_runs(share_global_filter: bool) -> dict[str, int]:
    """Two separate runs, the second one served by a group that matches no filter.

    With share_global_filter the second run reuses the first run's GlobalFilter, whose collection still
    holds the first run's match.
    """
    matching, unrelated = _make_fgs()

    first_filter = _filter_on(FSS_FILTERED)
    first_collector = PluginCollector.enabled_feature_groups({matching})
    first = mloda.run_all(
        [FSS_MATCHING],
        compute_frameworks={PythonDictFramework},
        plugin_collector=first_collector,
        global_filter=first_filter,
    )

    second_filter = first_filter if share_global_filter else _filter_on(FSS_FILTERED)
    second_collector = PluginCollector.enabled_feature_groups({unrelated})
    second = mloda.run_all(
        [FSS_UNRELATED],
        compute_frameworks={PythonDictFramework},
        plugin_collector=second_collector,
        global_filter=second_filter,
    )

    observed = _sentinels(first, (FSS_MATCHING,))
    observed.update(_sentinels(second, (FSS_UNRELATED,)))

    del matching, unrelated, first_filter, second_filter, first_collector, second_collector, first, second
    return observed


def test_no_global_filter_leaves_filters_none() -> None:
    """Without a GlobalFilter the plan never reaches add_filters, so filters stays None."""
    observed = _run(None)
    assert observed[FSS_MATCHING] == NO_FILTER_SET, f"no GlobalFilter must leave filters None: {observed!r}"
    assert observed[FSS_UNRELATED] == NO_FILTER_SET, f"no GlobalFilter must leave filters None: {observed!r}"


def test_a_global_filter_nothing_matched_leaves_filters_none() -> None:
    """A GlobalFilter no feature group matched leaves an empty collection, so filters stays None."""
    observed = _run(FSS_NEVER_SERVED)
    assert observed[FSS_MATCHING] == NO_FILTER_SET, f"an unmatched GlobalFilter must leave filters None: {observed!r}"
    assert observed[FSS_UNRELATED] == NO_FILTER_SET, f"an unmatched GlobalFilter must leave filters None: {observed!r}"


def test_a_group_matching_no_filter_gets_an_empty_set_when_another_group_matched() -> None:
    """Once any group matched, a group that matched nothing gets set(), not None."""
    observed = _run(FSS_FILTERED)
    assert observed[FSS_UNRELATED] == 0, (
        f"a non-matching group in a run where another group matched must get an empty set, not None: {observed!r}"
    )


def test_the_group_that_matched_gets_its_filters() -> None:
    """The group the filter matched receives the one filter it matched."""
    observed = _run(FSS_FILTERED)
    assert observed[FSS_MATCHING] == 1, f"the matching group must receive its one filter: {observed!r}"


def test_the_unmatched_feature_set_of_one_group_gets_an_empty_set() -> None:
    """A second feature group is not needed: one group's other feature set is enough to get set()."""
    observed = _run_split()
    assert observed["collection_entries"] == 1, f"exactly one match must be recorded: {observed!r}"
    assert observed["collection_groups"] == 1, f"only one feature group may be involved: {observed!r}"
    assert observed[FSS_SPLIT_TENANT_Y] == 0, (
        f"the feature set the filter did not match must get an empty set, not None: {observed!r}"
    )


def test_the_matched_feature_set_of_one_group_gets_its_filter() -> None:
    """The sibling feature set of the same group, the one the filter matched, receives that filter."""
    observed = _run_split()
    assert observed[FSS_SPLIT_TENANT_X] == 1, f"the matched feature set must receive its one filter: {observed!r}"


def test_a_reused_global_filter_keeps_handing_out_empty_sets_in_a_later_run() -> None:
    """The collection persists across runs, so a later run matching nothing still gets set(), not None."""
    observed = _run_two_runs(share_global_filter=True)
    assert observed[FSS_MATCHING] == 1, f"the first run's group must receive its one filter: {observed!r}"
    assert observed[FSS_UNRELATED] == 0, (
        f"a reused GlobalFilter keeps its collection non-empty, so the second run gets an empty set: {observed!r}"
    )


def test_a_fresh_global_filter_in_the_later_run_leaves_filters_none() -> None:
    """Control for the reuse test: the same second run with its own GlobalFilter gets None."""
    observed = _run_two_runs(share_global_filter=False)
    # Without this the control would keep passing once the first run stopped matching, controlling nothing.
    assert observed[FSS_MATCHING] == 1, f"the first run's group must receive its one filter: {observed!r}"
    assert observed[FSS_UNRELATED] == NO_FILTER_SET, (
        f"a fresh GlobalFilter has an empty collection, so the second run must get None: {observed!r}"
    )
