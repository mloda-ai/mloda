"""Regression tests for issue #712: a GlobalFilter on a column that a root FeatureGroup itself outputs."""

from typing import Any, ClassVar

import pytest

from mloda.provider import ANY_OUTPUT_PART, ComputeFramework, FeatureSet, DataCreatorFeatureGroup
from mloda.core.filter.filter_engine import BaseFilterEngine
from mloda.user import (
    Feature,
    Features,
    FilterType,
    GlobalFilter,
    Options,
    SingleFilter,
    ParallelizationMode,
    PluginCollector,
)
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from tests.test_core.test_tooling import MlodaTestRunner


class FilterOwnOutputRetrievalFG(DataCreatorFeatureGroup):
    """Root FeatureGroup that requires context-only options and outputs the filterable distance column."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"retrieval__text", "retrieval__doc_id", "retrieval__distance"}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        opts = features.options
        assert opts is not None
        if opts.context.get("query") is None or opts.context.get("index_dir") is None:
            raise ValueError("FilterOwnOutputRetrievalFG requires options.context['query'] and ['index_dir']")
        return {
            "retrieval__text": ["a", "b", "c"],
            "retrieval__doc_id": [1, 2, 3],
            "retrieval__distance": [0.1, 0.4, 0.9],
        }


class FpartRowsAnyFG(DataCreatorFeatureGroup):
    """Serves only parts of fpart__rows, naming them at run time."""

    OUTPUT_PARTS: ClassVar[tuple[str, ...] | None] = (ANY_OUTPUT_PART,)

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"fpart__rows"}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        names = {str(feature.name) for feature in features.features}
        assert names == {"fpart__rows"}, f"a parts-only group must be handed the base name only, got {names}"
        return {"fpart__rows~s": ["alice", "bob"], "fpart__rows~t": [1, 2]}


class FpartRowsDeclaredFG(FpartRowsAnyFG):
    """Same group declaring its parts."""

    OUTPUT_PARTS = ("s", "t")


class FpartLiteralFG(DataCreatorFeatureGroup):
    """Claims the literal whole name fpart__lit~k."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"fpart__lit~k"}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"fpart__lit~k": ["alice", "bob"]}


class FpartRowsNumericFG(FpartRowsAnyFG):
    """Declares numeric parts, which the default set_feature_name would strip."""

    OUTPUT_PARTS = ("0", "1")

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"fpart__rows~0": ["alice", "bob"], "fpart__rows~1": [1, 2]}


class FpartRowsRenamingFG(FpartRowsAnyFG):
    """Renames the base name in set_feature_name (leaving `~` names untouched) and emits the renamed columns."""

    OUTPUT_PARTS = ("s", "t")

    def set_feature_name(self, config: Options, feature_name: Any) -> Any:
        if "~" in str(feature_name):
            return feature_name
        return type(feature_name)(f"{feature_name}_x")

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"fpart__rows_x~s": ["alice", "bob"], "fpart__rows_x~t": [1, 2]}


def _run_part_filter(group: type[DataCreatorFeatureGroup], requested: str, filtered: str, flight_server: Any) -> Any:
    global_filter = GlobalFilter()
    global_filter.add_filter(filtered, "equal", {"value": "alice"})
    return MlodaTestRunner.run_api(
        Features([Feature(requested)]),
        compute_frameworks=[PythonDictFramework],
        parallelization_modes={ParallelizationMode.SYNC},
        flight_server=flight_server,
        global_filter=global_filter,
        plugin_collector=PluginCollector.enabled_feature_groups({group}),
    )


# (group, filtered/requested part, other part, base name of the returned column)
_PART_CASES = [
    pytest.param(FpartRowsAnyFG, "s", "t", "fpart__rows", id="any_part"),
    pytest.param(FpartRowsDeclaredFG, "s", "t", "fpart__rows", id="declared_parts"),
    pytest.param(FpartRowsNumericFG, "0", "1", "fpart__rows", id="numeric_parts"),
    pytest.param(FpartRowsRenamingFG, "s", "t", "fpart__rows_x", id="renaming_group"),
]


@pytest.mark.parametrize("group, part, other, out_base", _PART_CASES)
def test_filter_on_declared_output_part_filters_rows(
    group: type[DataCreatorFeatureGroup], part: str, other: str, out_base: str, flight_server: Any
) -> None:
    result = _run_part_filter(group, f"fpart__rows~{part}", f"fpart__rows~{part}", flight_server)

    assert len(result.results) == 1
    assert result.results[0] == {f"{out_base}~{part}": ["alice"]}


@pytest.mark.parametrize("group, part, other, out_base", _PART_CASES)
def test_filter_part_differs_from_requested_part(
    group: type[DataCreatorFeatureGroup], part: str, other: str, out_base: str, flight_server: Any
) -> None:
    result = _run_part_filter(group, f"fpart__rows~{other}", f"fpart__rows~{part}", flight_server)

    assert len(result.results) == 1
    assert result.results[0] == {f"{out_base}~{other}": [1]}


def test_filter_on_literal_whole_name_with_tilde_still_applies(flight_server: Any) -> None:
    result = _run_part_filter(FpartLiteralFG, "fpart__lit~k", "fpart__lit~k", flight_server)

    assert len(result.results) == 1
    assert result.results[0] == {"fpart__lit~k": ["alice"]}


_ENABLED = PluginCollector.enabled_feature_groups({FilterOwnOutputRetrievalFG})

_EXPECTED_FILTERED_RESULT: dict[str, list[Any]] = {
    "retrieval__text": ["a", "b"],
    "retrieval__distance": [0.1, 0.4],
}


def _make_context_options() -> Options:
    """Fresh Options per feature: Options instances are mutated per feature."""
    return Options(context={"query": "q", "index_dir": "/idx"})


def test_filter_by_name_on_own_output_column_with_context_options(flight_server: Any) -> None:
    """Failure A: filter declared by name must not strip the consumer context keys into group.

    Currently the filter feature lands in its own FeatureSet whose options hold everything in
    group and an empty context, so calculate_feature raises ValueError. Expected: no crash,
    both requested columns present, rows filtered to distance <= 0.5.
    """
    features = Features(
        [
            Feature("retrieval__text", _make_context_options()),
            Feature("retrieval__distance", _make_context_options()),
        ]
    )

    global_filter = GlobalFilter()
    global_filter.add_filter("retrieval__distance", "max", {"value": 0.5})

    result = MlodaTestRunner.run_api(
        features,
        compute_frameworks=[PythonDictFramework],
        parallelization_modes={ParallelizationMode.SYNC},
        flight_server=flight_server,
        global_filter=global_filter,
        plugin_collector=_ENABLED,
    )

    assert len(result.results) == 1
    assert result.results[0] == _EXPECTED_FILTERED_RESULT


def test_filter_feature_with_matching_options_keeps_requested_column(flight_server: Any) -> None:
    """Failure B: the filter twin must not dedupe away the genuinely requested column.

    retrieval__text is requested BEFORE retrieval__distance so the engine creates the filter
    twin first; the requested retrieval__distance is then deduped against the twin and loses
    its initial_requested_data flag, dropping the column from the result. Expected: both
    requested columns present, rows filtered to distance <= 0.5.
    """
    features = Features(
        [
            Feature("retrieval__text", _make_context_options()),
            Feature("retrieval__distance", _make_context_options()),
        ]
    )

    global_filter = GlobalFilter()
    global_filter.add_filter(Feature("retrieval__distance", _make_context_options()), "max", {"value": 0.5})

    result = MlodaTestRunner.run_api(
        features,
        compute_frameworks=[PythonDictFramework],
        parallelization_modes={ParallelizationMode.SYNC},
        flight_server=flight_server,
        global_filter=global_filter,
        plugin_collector=_ENABLED,
    )

    assert len(result.results) == 1
    assert result.results[0] == _EXPECTED_FILTERED_RESULT


def test_unify_options_preserves_key_category() -> None:
    """unify_options must keep consumer group keys in group and consumer context keys in context."""
    global_filter = GlobalFilter()

    unified = global_filter.unify_options(Options(group={"g": 1}, context={"c": 2}), Options())

    assert "c" not in unified.group, f"context key 'c' must not land in group, got group={unified.group}"
    assert unified.group == {"g": 1}
    assert unified.context == {"c": 2}


@pytest.mark.parametrize(
    "filter_column, expected_applicable",
    [("fpart__base~s", True), ("fpart__base~other", False), ("fpart__base", True)],
)
def test_applicable_filters_matches_requested_output_parts(filter_column: str, expected_applicable: bool) -> None:
    """A filter on `<base>~<requested part>` is applicable; an unrequested part is not."""
    feature = Feature("fpart__base")
    feature.record_part("s")
    feature_set = FeatureSet([feature])
    single_filter = SingleFilter(filter_column, FilterType.MIN, {"value": 1})
    feature_set.filters = {single_filter}
    assert BaseFilterEngine.applicable_filters(feature_set) == ([single_filter] if expected_applicable else [])


def test_applicable_filters_none_filters_gives_empty() -> None:
    feature = Feature("fpart__base")
    feature.record_part("s")
    feature_set = FeatureSet([feature])
    feature_set.filters = None
    assert BaseFilterEngine.applicable_filters(feature_set) == []
