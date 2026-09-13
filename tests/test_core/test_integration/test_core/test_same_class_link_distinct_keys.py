"""A same-class link whose two sides join on differently named key columns, run end to end through ``mloda.run_all``."""

from __future__ import annotations

from typing import Any

from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, JoinSpec, Link, Options, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


_SCLKR_ELECTION = {"sclkr_nr": [1, 2, 3, 4], "sclkr_votes": [10, 20, 30, 40]}
_SCLKR_REGION = {"sclkr_code": [2, 3, 4, 5], "sclkr_pop": [200, 300, 400, 500]}
_SCLKR_SOURCE_DATA = {"election": _SCLKR_ELECTION, "region": _SCLKR_REGION}

_SCLKR_REGION2 = {"sclkr_nr": [1, 2, 3], "sclkr_pop2": [100, 200, 300]}
_SCLKR_A = {"sclkr_nr": [1, 2, 3], "sclkr_va": [1, 2, 3]}
_SCLKR_B = {"sclkr_nr": [1, 2, 3], "sclkr_vb": [10, 20, 30]}
_SCLKR_HUB_DATA = {"region2": _SCLKR_REGION2, "a": _SCLKR_A, "b": _SCLKR_B}


class SclkrSourceFG(FeatureGroup):
    """Root FG reading like a real source: a foreign-key column absent from a source raises KeyError."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"sclkr_nr", "sclkr_votes", "sclkr_code", "sclkr_pop"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        source = features.get_options_key("sclkr_source")
        return {name: _SCLKR_SOURCE_DATA[source][name] for name in features.get_all_names()}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class SclkrJoinedFG(FeatureGroup):
    """Consumer needing the election votes and the region population through the same-class link."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {
            Feature("sclkr_votes", options={"sclkr_source": "election"}),
            Feature("sclkr_pop", options={"sclkr_source": "region"}),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        joined = data.to_pydict()
        return {cls.get_class_name(): [v + p for v, p in zip(joined["sclkr_votes"], joined["sclkr_pop"])]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class SclkrHubSourceFG(FeatureGroup):
    """Root FG serving a hub and two spokes, all sharing the join key ``sclkr_nr``."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"sclkr_nr", "sclkr_pop2", "sclkr_va", "sclkr_vb"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        source = features.get_options_key("sclkr_source")
        return {name: _SCLKR_HUB_DATA[source][name] for name in features.get_all_names()}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class SclkrHubJoinedFG(FeatureGroup):
    """Consumer needing one column from the hub and one from each spoke."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {
            Feature("sclkr_pop2", options={"sclkr_source": "region2"}),
            Feature("sclkr_va", options={"sclkr_source": "a"}),
            Feature("sclkr_vb", options={"sclkr_source": "b"}),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        joined = data.to_pydict()
        return {
            cls.get_class_name(): [
                pop2 + va + vb for pop2, va, vb in zip(joined["sclkr_pop2"], joined["sclkr_va"], joined["sclkr_vb"])
            ]
        }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


def test_same_class_link_with_distinct_keys_joins_election_and_region() -> None:
    link = Link.inner(
        JoinSpec(SclkrSourceFG, "sclkr_nr"),
        JoinSpec(SclkrSourceFG, "sclkr_code"),
        left_discriminator={"sclkr_source": "election"},
        right_discriminator={"sclkr_source": "region"},
    )

    results = mloda.run_all(
        [Feature(SclkrJoinedFG.get_class_name())],
        links={link},
        compute_frameworks=["PyArrowTable"],
        plugin_collector=PluginCollector.enabled_feature_groups({SclkrSourceFG, SclkrJoinedFG}),
        parallelization_modes={ParallelizationMode.SYNC},
    )

    values = sorted(results[0].to_pydict()[SclkrJoinedFG.get_class_name()])
    assert values == [220, 330, 440]


def test_hub_and_two_spokes_share_one_key_via_distinct_discriminators() -> None:
    links = {
        Link.inner(
            JoinSpec(SclkrHubSourceFG, "sclkr_nr"),
            JoinSpec(SclkrHubSourceFG, "sclkr_nr"),
            left_discriminator={"sclkr_source": "region2"},
            right_discriminator={"sclkr_source": "a"},
        ),
        Link.inner(
            JoinSpec(SclkrHubSourceFG, "sclkr_nr"),
            JoinSpec(SclkrHubSourceFG, "sclkr_nr"),
            left_discriminator={"sclkr_source": "region2"},
            right_discriminator={"sclkr_source": "b"},
        ),
    }
    assert len(links) == 2

    results = mloda.run_all(
        [Feature(SclkrHubJoinedFG.get_class_name())],
        links=links,
        compute_frameworks=["PyArrowTable"],
        plugin_collector=PluginCollector.enabled_feature_groups({SclkrHubSourceFG, SclkrHubJoinedFG}),
        parallelization_modes={ParallelizationMode.SYNC},
    )

    values = sorted(results[0].to_pydict()[SclkrHubJoinedFG.get_class_name()])
    assert values == [111, 222, 333]
