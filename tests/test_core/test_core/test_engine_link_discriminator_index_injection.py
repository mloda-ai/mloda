"""Discriminator-aware index-key injection for same-class links with distinct join-key column names."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from mloda.core.core.engine import Engine
from mloda.core.core.step.join_step import JoinStep
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Features, Index, JoinSpec, Link, Options, PluginCollector
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


_SCLK_ELECTION = {"sclk_nr": [1, 2, 3, 4], "sclk_votes": [10, 20, 30, 40]}
_SCLK_REGION = {"sclk_code": [2, 3, 4, 5], "sclk_pop": [200, 300, 400, 500]}
_SCLK_DATA = {"election": _SCLK_ELECTION, "region": _SCLK_REGION}


class SclkSourceFG(FeatureGroup):
    """Root FG with no declared index_columns: injection runs through _add_index_feature_from_links."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"sclk_nr", "sclk_votes", "sclk_code", "sclk_pop"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        source = features.get_options_key("sclk_source")
        return {name: _SCLK_DATA[source][name] for name in features.get_all_names()}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class SclkIndexedSourceFG(SclkSourceFG):
    """Same shape, but declares index_columns so injection runs through _process_index_feature."""

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index(("sclk_nr",)), Index(("sclk_code",))]


class SclkOtherFG(FeatureGroup):
    """Distinct class used as the right side of a different-class link."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"sclk_other_key", "sclk_other_val"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"sclk_other_key": [2, 3, 4, 5], "sclk_other_val": [1, 1, 1, 1]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class SclkJoinedFG(FeatureGroup):
    """Consumer used only to build a real execution plan for the join-step count."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {
            Feature("sclk_votes", options={"sclk_source": "election"}),
            Feature("sclk_pop", options={"sclk_source": "region"}),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


def _link_a() -> Link:
    return Link.inner(
        JoinSpec(SclkSourceFG, "sclk_nr"),
        JoinSpec(SclkSourceFG, "sclk_code"),
        left_discriminator={"sclk_source": "election"},
        right_discriminator={"sclk_source": "region"},
    )


def _link_b() -> Link:
    return Link.inner(
        JoinSpec(SclkIndexedSourceFG, "sclk_nr"),
        JoinSpec(SclkIndexedSourceFG, "sclk_code"),
        left_discriminator={"sclk_source": "election"},
        right_discriminator={"sclk_source": "region"},
    )


def _build_engine(features: Features, links: set[Link], enabled: set[type[FeatureGroup]]) -> Engine:
    with patch("mloda.core.core.engine.Engine.create_setup_execution_plan"):
        engine = Engine(
            features, {PyArrowTable}, links, plugin_collector=PluginCollector.enabled_feature_groups(enabled)
        )
        engine.setup_features_recursion(features)
    return engine


def _group_names(engine: Engine, feature_group: type[FeatureGroup], key: str, value: str) -> set[str]:
    return {
        str(feature.name)
        for feature in engine.feature_group_collection[feature_group]
        if feature.options.get(key) == value
    }


def test_same_class_link_injects_only_the_matching_side_key() -> None:
    features = Features(
        [
            Feature("sclk_votes", options={"sclk_source": "election"}),
            Feature("sclk_pop", options={"sclk_source": "region"}),
        ]
    )
    engine = _build_engine(features, {_link_a()}, {SclkSourceFG})

    election = _group_names(engine, SclkSourceFG, "sclk_source", "election")
    region = _group_names(engine, SclkSourceFG, "sclk_source", "region")

    assert election == {"sclk_votes", "sclk_nr"}
    assert region == {"sclk_pop", "sclk_code"}


def test_same_class_link_with_index_columns_injects_only_the_matching_side_key() -> None:
    features = Features(
        [
            Feature("sclk_votes", options={"sclk_source": "election"}),
            Feature("sclk_pop", options={"sclk_source": "region"}),
        ]
    )
    engine = _build_engine(features, {_link_b()}, {SclkIndexedSourceFG})

    election = _group_names(engine, SclkIndexedSourceFG, "sclk_source", "election")
    region = _group_names(engine, SclkIndexedSourceFG, "sclk_source", "region")

    assert election == {"sclk_votes", "sclk_nr"}
    assert region == {"sclk_pop", "sclk_code"}


def test_same_class_link_without_discriminators_injects_both_keys_into_both_batches() -> None:
    link = Link.inner(JoinSpec(SclkSourceFG, "sclk_nr"), JoinSpec(SclkSourceFG, "sclk_code"))
    features = Features(
        [
            Feature("sclk_votes", options={"sclk_source": "election"}),
            Feature("sclk_pop", options={"sclk_source": "region"}),
        ]
    )
    engine = _build_engine(features, {link}, {SclkSourceFG})

    election = _group_names(engine, SclkSourceFG, "sclk_source", "election")
    region = _group_names(engine, SclkSourceFG, "sclk_source", "region")

    assert election == {"sclk_votes", "sclk_nr", "sclk_code"}
    assert region == {"sclk_pop", "sclk_nr", "sclk_code"}


def test_different_class_link_with_discriminators_injects_by_class_only() -> None:
    link = Link.inner(
        JoinSpec(SclkSourceFG, "sclk_nr"),
        JoinSpec(SclkOtherFG, "sclk_other_key"),
        left_discriminator={"sclk_source": "election"},
        right_discriminator={"sclk_source": "region"},
    )
    features = Features(
        [
            Feature("sclk_votes", options={"sclk_source": "election"}),
            Feature("sclk_other_val", options={"sclk_source": "region"}),
        ]
    )
    engine = _build_engine(features, {link}, {SclkSourceFG, SclkOtherFG})

    source_names = {str(feature.name) for feature in engine.feature_group_collection[SclkSourceFG]}
    other_names = {str(feature.name) for feature in engine.feature_group_collection[SclkOtherFG]}

    assert "sclk_nr" in source_names
    assert "sclk_other_key" in other_names


def test_region_side_feature_named_after_the_other_sides_key_still_gets_its_own_key() -> None:
    """sclk_pop is omitted on purpose: if present, its own processing would inject sclk_code anyway and mask the bug."""
    features = Features(
        [
            Feature("sclk_votes", options={"sclk_source": "election"}),
            Feature("sclk_nr", options={"sclk_source": "region"}),
        ]
    )
    engine = _build_engine(features, {_link_a()}, {SclkSourceFG})

    region = _group_names(engine, SclkSourceFG, "sclk_source", "region")

    assert "sclk_code" in region


def test_scenario_a_plans_exactly_one_join_step() -> None:
    engine = Engine(
        Features([Feature(SclkJoinedFG.get_class_name())]),
        {PyArrowTable},
        {_link_a()},
        plugin_collector=PluginCollector.enabled_feature_groups({SclkSourceFG, SclkJoinedFG}),
    )

    join_steps = [step for step in engine.execution_planner if isinstance(step, JoinStep)]
    assert len(join_steps) == 1
