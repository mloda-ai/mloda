"""Regression tests for issue #712: a GlobalFilter on a column that a root FeatureGroup itself outputs."""

from typing import Any

from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, Features, GlobalFilter, Options, ParallelizationMode, PluginCollector
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from tests.test_core.test_tooling import MlodaTestRunner


class FilterOwnOutputRetrievalFG(FeatureGroup):
    """Root FeatureGroup that requires context-only options and outputs the filterable distance column."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"retrieval__text", "retrieval__doc_id", "retrieval__distance"})

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
        compute_frameworks={PythonDictFramework},
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
        compute_frameworks={PythonDictFramework},
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
