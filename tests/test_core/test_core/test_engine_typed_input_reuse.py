"""Typed direct requests may satisfy untyped dependencies without broadening Feature identity."""

from __future__ import annotations

from copy import copy
from typing import Any
from uuid import UUID

import pytest

from mloda.provider import DataCreatorFeatureGroup, FeatureGroup, FeatureSet
from mloda.user import DataType, Feature, FeatureName, GlobalFilter, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from tests.test_core.test_abstract_plugins.name_bound_fixtures import make_root_fg


SOURCE = "typed_reuse_source"
CONSUMER = "typed_reuse_consumer"
LATER_CONSUMER = "typed_reuse_later_consumer"


def _plan_direct_and_dependency(
    direct_type: DataType | None,
    input_type: DataType | None,
    direct_first: bool,
    direct_options: Options | None = None,
    input_options: Options | None = None,
    follow_on: bool = False,
) -> tuple[list[Feature], set[UUID]]:
    source = make_root_fg("TypedReuse", __name__, SOURCE, [1, 2])

    class TypedReuseConsumer(FeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {CONSUMER, LATER_CONSUMER}

        def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature]:
            return {Feature(SOURCE, options=copy(input_options), data_type=input_type)}

    direct = Feature(SOURCE, options=direct_options, data_type=direct_type)
    consumer = Feature(CONSUMER)
    requests = [direct, consumer] if direct_first else [consumer, direct]
    if follow_on:
        requests.append(Feature(LATER_CONSUMER))
    session = mloda.prepare(
        requests,
        compute_frameworks=[PythonDictFramework],
        plugin_collector=PluginCollector.enabled_feature_groups({source, TypedReuseConsumer}),
    )
    assert session.engine is not None
    source_features = session.engine.feature_group_collection[source]
    assert all(feature in source_features for feature in source_features)
    resolved_consumers = session.engine.feature_group_collection[TypedReuseConsumer]
    assert len(resolved_consumers) == (2 if follow_on else 1)
    parent_sets = [set(session.engine.feature_link_parents[feature.uuid]) for feature in resolved_consumers]
    assert all(parents == parent_sets[0] for parents in parent_sets)
    return list(source_features), parent_sets[0]


@pytest.mark.parametrize("direct_first", [True, False], ids=["direct_first", "dependency_first"])
def test_typed_direct_request_is_the_single_parent_of_its_untyped_consumer(direct_first: bool) -> None:
    sources, parents = _plan_direct_and_dependency(DataType.INT64, None, direct_first)

    assert len(sources) == 1
    (survivor,) = sources
    assert survivor.data_type is DataType.INT64
    assert survivor.initial_requested_data
    assert parents == {survivor.uuid}


def test_follow_on_untyped_consumer_reuses_the_promoted_typed_parent() -> None:
    sources, parents = _plan_direct_and_dependency(DataType.INT64, None, direct_first=False, follow_on=True)

    assert len(sources) == 1
    (survivor,) = sources
    assert survivor.data_type is DataType.INT64
    assert survivor.initial_requested_data
    assert parents == {survivor.uuid}


@pytest.mark.parametrize("direct_first", [True, False], ids=["direct_first", "dependency_first"])
@pytest.mark.parametrize(
    ("direct_type", "input_type"),
    [(DataType.INT64, DataType.STRING), (None, DataType.INT64)],
    ids=["different_explicit_types", "typed_input_untyped_request"],
)
def test_a_typed_dependency_does_not_merge_with_a_different_direct_request(
    direct_first: bool, direct_type: DataType | None, input_type: DataType
) -> None:
    sources, parents = _plan_direct_and_dependency(direct_type, input_type, direct_first)

    assert len(sources) == 2
    (direct,) = [feature for feature in sources if feature.initial_requested_data]
    (dependency,) = [feature for feature in sources if not feature.initial_requested_data]
    assert direct.data_type is direct_type
    assert dependency.data_type is input_type
    assert parents == {dependency.uuid}


@pytest.mark.parametrize("direct_first", [True, False], ids=["direct_first", "dependency_first"])
@pytest.mark.parametrize("scope", ["group", "context"])
def test_untyped_input_with_different_options_stays_separate(direct_first: bool, scope: str) -> None:
    direct_options = Options(group={"typed_reuse_key": "direct"})
    input_options = Options(group={"typed_reuse_key": "input"})
    if scope == "context":
        direct_options = Options(context={"typed_reuse_key": "direct"})
        input_options = Options(context={"typed_reuse_key": "input"})
    sources, parents = _plan_direct_and_dependency(DataType.INT64, None, direct_first, direct_options, input_options)

    assert len(sources) == 2
    (direct,) = [feature for feature in sources if feature.initial_requested_data]
    (dependency,) = [feature for feature in sources if not feature.initial_requested_data]
    assert direct.options.get("typed_reuse_key") == "direct"
    assert dependency.options.get("typed_reuse_key") == "input"
    assert direct.data_type is DataType.INT64
    assert dependency.data_type is None
    assert parents == {dependency.uuid}


def test_feature_equality_and_hash_collections_still_distinguish_data_types() -> None:
    features = [Feature(SOURCE), Feature.int64_of(SOURCE), Feature.str_of(SOURCE)]

    for index, first in enumerate(features):
        for second in features[index + 1 :]:
            assert first != second
    assert len(set(features)) == 3
    assert {feature: feature.data_type for feature in features}[Feature.int64_of(SOURCE)] is DataType.INT64


def _run_filtered_sibling_requests() -> tuple[set[str], dict[str, tuple[str, ...]], list[Any]]:
    values = {"typed_filter_source_a": [1, 2], "typed_filter_source_b": [10, 20]}
    inputs = {"typed_filter_a": "typed_filter_source_a", "typed_filter_b": "typed_filter_source_b"}

    class TypedFilterSource(DataCreatorFeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return set(values)

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return {str(feature.name): values[str(feature.name)] for feature in features.features}

    class TypedFilterDerived(FeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return set(inputs)

        def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature]:
            return {Feature(inputs[str(feature_name)])}

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return {str(feature.name): data[inputs[str(feature.name)]] for feature in features.features}

    global_filter = GlobalFilter()
    global_filter.add_filter("typed_filter_b", "min", {"value": 20})
    session = mloda.prepare(
        [Feature("typed_filter_a"), Feature.int64_of("typed_filter_b")],
        compute_frameworks=[PythonDictFramework],
        plugin_collector=PluginCollector.enabled_feature_groups({TypedFilterSource, TypedFilterDerived}),
        global_filter=global_filter,
    )
    plan = session.resolved_plan()
    sources = {name for step in plan if step.feature_group is TypedFilterSource for name in step.feature_names}
    edges = {
        name: dependencies
        for step in plan
        if step.feature_group is TypedFilterDerived
        for name, dependencies in step.input_feature_edges.items()
    }
    return sources, edges, list(session.run())


def test_typed_request_after_filter_placeholder_still_resolves_its_own_inputs() -> None:
    sources, edges, results = _run_filtered_sibling_requests()

    assert sources == {"typed_filter_source_a", "typed_filter_source_b"}
    assert edges == {
        "typed_filter_a": ("typed_filter_source_a",),
        "typed_filter_b": ("typed_filter_source_b",),
    }
    assert results == [{"typed_filter_a": [2], "typed_filter_b": [20]}]
