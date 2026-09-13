"""Polymorphic link key injection: Engine._link_sides should match by issubclass, not exact equality."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from mloda.core.core.engine import Engine
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, Features, Index, JoinSpec, Link, PluginCollector
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


class PlkiBaseA(FeatureGroup):
    """Base side A: no input_data, so it never resolves a requested feature on its own."""


class PlkiBaseB(FeatureGroup):
    """Base side B: no input_data, so it never resolves a requested feature on its own."""


class PlkiSubA(PlkiBaseA):
    """Concrete side A; also supports plki_a_key directly."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"plki_a_val", "plki_a_key"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {name: [1] for name in features.get_all_names()}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class PlkiSubA2(PlkiBaseA):
    """Sibling of PlkiSubA: must never receive keys injected for links naming PlkiSubA."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"plki_a2_val"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {name: [1] for name in features.get_all_names()}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class PlkiSubB(PlkiBaseB):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"plki_b_val"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {name: [1] for name in features.get_all_names()}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class PlkiIdxBaseA(FeatureGroup):
    """Base side A declaring index_columns(), inherited (not overridden) by its subclass."""

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index(("plki_ia_key",))]


class PlkiIdxBaseB(FeatureGroup):
    """Base side B declaring index_columns(), inherited (not overridden) by its subclass."""

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index(("plki_ib_key",))]


class PlkiIdxSubA(PlkiIdxBaseA):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"plki_ia_val"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {name: [1] for name in features.get_all_names()}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class PlkiIdxSubB(PlkiIdxBaseB):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"plki_ib_val"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {name: [1] for name in features.get_all_names()}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


def _build_engine(features: Features, links: set[Link], enabled: set[type[FeatureGroup]]) -> Engine:
    with patch("mloda.core.core.engine.Engine.create_setup_execution_plan"):
        engine = Engine(
            features, {PyArrowTable}, links, plugin_collector=PluginCollector.enabled_feature_groups(enabled)
        )
        engine.setup_features_recursion(features)
    return engine


def _all_names(engine: Engine, feature_group: type[FeatureGroup]) -> set[str]:
    return {str(feature.name) for feature in engine.feature_group_collection[feature_group]}


def _group_names(engine: Engine, feature_group: type[FeatureGroup], key: str, value: str) -> set[str]:
    return {
        str(feature.name)
        for feature in engine.feature_group_collection[feature_group]
        if feature.options.get(key) == value
    }


def test_base_class_link_injects_key_into_resolved_subclasses() -> None:
    link = Link.inner(JoinSpec(PlkiBaseA, "plki_a_key"), JoinSpec(PlkiBaseB, "plki_b_key"))
    features = Features([Feature("plki_a_val"), Feature("plki_b_val")])
    engine = _build_engine(features, {link}, {PlkiSubA, PlkiSubB})

    assert _all_names(engine, PlkiSubA) == {"plki_a_val", "plki_a_key"}
    assert _all_names(engine, PlkiSubB) == {"plki_b_val", "plki_b_key"}


def test_base_class_link_with_index_columns_injects_key_into_resolved_subclasses() -> None:
    link = Link.inner(JoinSpec(PlkiIdxBaseA, "plki_ia_key"), JoinSpec(PlkiIdxBaseB, "plki_ib_key"))
    features = Features([Feature("plki_ia_val"), Feature("plki_ib_val")])
    engine = _build_engine(features, {link}, {PlkiIdxSubA, PlkiIdxSubB})

    assert _all_names(engine, PlkiIdxSubA) == {"plki_ia_val", "plki_ia_key"}
    assert _all_names(engine, PlkiIdxSubB) == {"plki_ib_val", "plki_ib_key"}


def test_base_class_self_join_with_discriminators_narrows_per_side() -> None:
    link = Link.inner(
        JoinSpec(PlkiBaseA, "plki_a_key"),
        JoinSpec(PlkiBaseA, "plki_a_key2"),
        left_discriminator={"plki_src": "l"},
        right_discriminator={"plki_src": "r"},
    )
    features = Features(
        [
            Feature("plki_a_val", options={"plki_src": "l"}),
            Feature("plki_a_val", options={"plki_src": "r"}),
        ]
    )
    engine = _build_engine(features, {link}, {PlkiSubA})

    assert _group_names(engine, PlkiSubA, "plki_src", "l") == {"plki_a_val", "plki_a_key"}
    assert _group_names(engine, PlkiSubA, "plki_src", "r") == {"plki_a_val", "plki_a_key2"}


def test_subclass_link_does_not_leak_into_sibling_subclass() -> None:
    link = Link.inner(JoinSpec(PlkiSubA, "plki_a_key"), JoinSpec(PlkiSubB, "plki_b_key"))
    features = Features([Feature("plki_a2_val"), Feature("plki_b_val")])
    engine = _build_engine(features, {link}, {PlkiSubA2, PlkiSubB})

    assert _all_names(engine, PlkiSubA2) == {"plki_a2_val"}


def test_base_link_plus_subclass_link_both_inject_into_subclass_but_not_sibling() -> None:
    link_base = Link.inner(JoinSpec(PlkiBaseA, "plki_a_key"), JoinSpec(PlkiBaseB, "plki_b_key"))
    link_sub = Link.inner(JoinSpec(PlkiSubA, "plki_a_key2"), JoinSpec(PlkiSubB, "plki_b_key2"))
    features = Features([Feature("plki_a_val"), Feature("plki_a2_val")])
    engine = _build_engine(features, {link_base, link_sub}, {PlkiSubA, PlkiSubA2})

    assert _all_names(engine, PlkiSubA) == {"plki_a_val", "plki_a_key", "plki_a_key2"}
    assert _all_names(engine, PlkiSubA2) == {"plki_a2_val", "plki_a_key"}


def test_inheritance_related_link_sides_without_discriminators() -> None:
    link = Link.inner(JoinSpec(PlkiBaseA, "plki_a_key"), JoinSpec(PlkiSubA, "plki_a_key2"))
    features = Features([Feature("plki_a_val"), Feature("plki_a2_val")])
    engine = _build_engine(features, {link}, {PlkiSubA, PlkiSubA2})

    assert _all_names(engine, PlkiSubA) == {"plki_a_val", "plki_a_key", "plki_a_key2"}
    assert _all_names(engine, PlkiSubA2) == {"plki_a2_val", "plki_a_key"}


def test_inheritance_related_link_sides_with_discriminators_narrow_only_dual_match() -> None:
    link = Link.inner(
        JoinSpec(PlkiBaseA, "plki_a_key"),
        JoinSpec(PlkiSubA, "plki_a_key2"),
        left_discriminator={"plki_src": "l"},
        right_discriminator={"plki_src": "r"},
    )
    features = Features(
        [
            Feature("plki_a_val", options={"plki_src": "r"}),
            Feature("plki_a_val", options={"plki_src": "l"}),
            Feature("plki_a2_val", options={"plki_src": "r"}),
        ]
    )
    engine = _build_engine(features, {link}, {PlkiSubA, PlkiSubA2})

    assert _group_names(engine, PlkiSubA, "plki_src", "r") == {"plki_a_val", "plki_a_key2"}
    assert _group_names(engine, PlkiSubA, "plki_src", "l") == {"plki_a_val", "plki_a_key"}
    assert _group_names(engine, PlkiSubA2, "plki_src", "r") == {"plki_a2_val", "plki_a_key"}


def test_requesting_a_link_key_itself_marks_a_key_only_batch() -> None:
    link_base = Link.inner(JoinSpec(PlkiBaseA, "plki_a_key"), JoinSpec(PlkiBaseB, "plki_b_key"))
    link_sub = Link.inner(JoinSpec(PlkiSubA, "plki_a_key2"), JoinSpec(PlkiSubB, "plki_b_key2"))
    features = Features([Feature("plki_a_key"), Feature("plki_b_val")])
    engine = _build_engine(features, {link_base, link_sub}, {PlkiSubA, PlkiSubB})

    assert _all_names(engine, PlkiSubA) == {"plki_a_key"}
    assert _all_names(engine, PlkiSubB) == {"plki_b_val", "plki_b_key", "plki_b_key2"}
