"""Existing request and input-copy behavior remains valid with run-scoped chain memoization."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from mloda.core.abstract_plugins.components.input_data.match_cache import run_match_cache
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.graph.build_graph import BuildGraph
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.user import DataAccessCollection, Feature, FeatureName, Options, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from tests.test_core.test_prepare.input_chain_helpers import (
    CHOICE,
    ChainB2320,
    ChainLeaf2320,
    ChainStub2320,
    prepare_chain,
    result_for,
)


TYPE_NAME = "type_compat_chain_2320"
TYPE_KEY = "type_value_compat_chain_2320"
DOMAIN = "domain_compat_chain_2320"
OPTIONS_PARENT = "options_parent_compat_chain_2320"
OPTIONS_LEAF = "options_leaf_compat_chain_2320"
MARKER = "provider_options_compat_chain_2320"


class _ChainCacheType2320(ChainStub2320):
    VALUE_TYPE: ClassVar[type[Any]] = list

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == TYPE_NAME and type(options.get(TYPE_KEY)) is cls.VALUE_TYPE


class ChainCacheFirst2320(_ChainCacheType2320):
    pass


class ChainCacheSecond2320(_ChainCacheType2320):
    VALUE_TYPE = tuple


class ChainCacheOptions2320(Options):
    def __init__(self) -> None:
        super().__init__()
        self.chain_marker = MARKER


class ChainCacheOptionsParent2320(ChainStub2320):
    MATCHED_NAMES = frozenset({OPTIONS_PARENT})

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature]:
        return {Feature(OPTIONS_LEAF, options=ChainCacheOptions2320())}


class ChainCacheOptionsLeaf2320(ChainStub2320):
    OBSERVED: ClassVar[list[tuple[type[Options], str | None]]] = []

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        if str(feature_name) != OPTIONS_LEAF:
            return False
        cls.OBSERVED.append((type(options), getattr(options, "chain_marker", None)))
        return isinstance(options, ChainCacheOptions2320) and options.chain_marker == MARKER


@pytest.mark.parametrize("values", [([], ()), ({}, set())], ids=["list-tuple", "dict-set"])
def test_type_sensitive_name_fit_does_not_reuse_another_options_type(
    values: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    first_value, second_value = values
    monkeypatch.setattr(ChainCacheFirst2320, "VALUE_TYPE", type(first_value))
    monkeypatch.setattr(ChainCacheSecond2320, "VALUE_TYPE", type(second_value))
    plugins: FeatureGroupEnvironmentMapping = {
        ChainCacheFirst2320: {PyArrowTable},
        ChainCacheSecond2320: {PyArrowTable},
    }

    with run_match_cache():
        for value, winner in (
            (first_value, ChainCacheFirst2320),
            (second_value, ChainCacheSecond2320),
            (first_value, ChainCacheFirst2320),
        ):
            result = IdentifyFeatureGroupClass.evaluate(
                Feature(TYPE_NAME, options={TYPE_KEY: value}), plugins, None, None
            )
            assert set(result.identified) == {winner}


def test_shared_declaration_keeps_distinct_child_nodes_for_distinct_parent_domains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ChainB2320, "DOMAIN_NAME", DOMAIN)
    monkeypatch.setattr(ChainLeaf2320, "DOMAIN_NAME", DOMAIN)

    session = mloda.prepare(
        [Feature(CHOICE), Feature(CHOICE, domain=DOMAIN)],
        compute_frameworks=[PyArrowTable],
        parallelization_modes={ParallelizationMode.SYNC},
        plugin_collector=PluginCollector.enabled_feature_groups({ChainB2320, ChainLeaf2320}),
    )

    assert session.engine is not None
    engine = session.engine
    consumers = {
        feature.domain.name if feature.domain else None: feature
        for feature in engine.feature_group_collection[ChainB2320]
    }
    inputs = {
        feature.domain.name if feature.domain else None: feature
        for feature in engine.feature_group_collection[ChainLeaf2320]
    }
    assert set(consumers) == set(inputs) == {None, DOMAIN}
    assert len({feature.uuid for feature in inputs.values()}) == 2
    for domain in (None, DOMAIN):
        assert engine.feature_link_parents[consumers[domain].uuid] == {inputs[domain].uuid}

    graph_builder = BuildGraph(engine.feature_link_parents, engine.feature_group_collection)
    graph_builder.build_graph_from_feature_links()
    graph = graph_builder.graph
    assert set(graph.get_nodes()) == {feature.uuid for feature in (*consumers.values(), *inputs.values())}
    assert len(graph.get_nodes()) == 4
    assert set(graph.get_edges()) == {(inputs[domain].uuid, consumers[domain].uuid) for domain in (None, DOMAIN)}


def test_provider_returned_input_options_keeps_its_subclass_and_marker_at_matching() -> None:
    ChainCacheOptionsLeaf2320.OBSERVED.clear()

    session = prepare_chain({ChainCacheOptionsParent2320, ChainCacheOptionsLeaf2320}, feature=Feature(OPTIONS_PARENT))

    assert set(result_for(session.resolution_report(), OPTIONS_PARENT).identified) == {ChainCacheOptionsParent2320}
    assert set(result_for(session.resolution_report(), OPTIONS_LEAF).identified) == {ChainCacheOptionsLeaf2320}
    assert ChainCacheOptionsLeaf2320.OBSERVED
    assert all(observed == (ChainCacheOptions2320, MARKER) for observed in ChainCacheOptionsLeaf2320.OBSERVED)
