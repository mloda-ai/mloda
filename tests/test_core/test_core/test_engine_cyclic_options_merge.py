"""The default-equivalent merge warning compares declared options, so it must survive cyclic values.

The branch is reached exactly when the feature equality probe matched, which cycle-safe Options
equality newly makes possible for cyclic group values. The provenance classes below cover
merge-order independence for own-key and consumer-attribution bookkeeping, including that a
pre-merge `copy()` of a survivor must not observe a later merge.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from copy import copy
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.core.engine import Engine


def _self_referential_list() -> list[Any]:
    cyclic: list[Any] = []
    cyclic.append(cyclic)
    return cyclic


def _self_referential_dict() -> dict[str, Any]:
    cyclic: dict[str, Any] = {}
    cyclic["self"] = cyclic
    return cyclic


def _intake_engine() -> Engine:
    """An Engine carrying only the intake state add_feature_to_collection reads."""
    engine = Engine.__new__(Engine)
    engine.feature_group_collection = defaultdict(set)
    engine.feature_link_parents = defaultdict(set)
    engine._intake_options_memo = {}
    engine._declared_options_by_uuid = {}
    engine.links = None
    return engine


class TestCyclicOptionsAtIntake:
    def test_merging_two_features_with_cyclic_group_options_does_not_recurse(self) -> None:
        engine = _intake_engine()
        first = Feature(name="x", options=Options(group={"g": _self_referential_list()}))
        second = Feature(name="x", options=Options(group={"g": _self_referential_list()}))

        assert engine.add_feature_to_collection(FeatureGroup, first, None) is True
        assert engine.add_feature_to_collection(FeatureGroup, second, None) is False

    def test_merging_two_features_with_cyclic_context_options_does_not_recurse(self) -> None:
        engine = _intake_engine()
        first = Feature(name="x", options=Options(group={"g": 1}, context={"c": _self_referential_dict()}))
        second = Feature(name="x", options=Options(group={"g": 1}, context={"c": _self_referential_dict()}))

        assert engine.add_feature_to_collection(FeatureGroup, first, None) is True
        assert engine.add_feature_to_collection(FeatureGroup, second, None) is False

    def test_equal_cyclic_declared_options_emit_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        engine = _intake_engine()
        survivor = Feature(name="x", options=Options(group={"g": _self_referential_list()}))
        arriving = Feature(name="x", options=Options(group={"g": _self_referential_list()}))
        engine._declared_options_by_uuid[survivor.uuid] = survivor.options

        with caplog.at_level(logging.WARNING, logger="mloda.core.core.engine"):
            engine._warn_on_default_equivalent_merge(arriving, arriving.options, survivor)

        assert "default-equivalent options" not in caplog.text

    def test_differing_cyclic_declared_options_still_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        def build(marker: int) -> Options:
            cyclic: list[Any] = [marker]
            cyclic.append(cyclic)
            return Options(group={"g": cyclic})

        engine = _intake_engine()
        survivor = Feature(name="x", options=build(1))
        arriving = Feature(name="x", options=build(2))
        engine._declared_options_by_uuid[survivor.uuid] = survivor.options

        with caplog.at_level(logging.WARNING, logger="mloda.core.core.engine"):
            engine._warn_on_default_equivalent_merge(arriving, arriving.options, survivor)

        assert "default-equivalent options" in caplog.text

    def test_differing_acyclic_declared_options_still_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        engine = _intake_engine()
        survivor = Feature(name="x", options=Options(group={"g": 1}))
        arriving = Feature(name="x", options=Options(group={"g": 2}))
        engine._declared_options_by_uuid[survivor.uuid] = survivor.options

        with caplog.at_level(logging.WARNING, logger="mloda.core.core.engine"):
            engine._warn_on_default_equivalent_merge(arriving, arriving.options, survivor)

        assert "default-equivalent options" in caplog.text


class TestOwnKeysMergeOrderIndependence:
    """When two value-equal Feature requests merge in add_feature_to_collection, the surviving
    feature's own-key provenance for a shared context key must be the union of both requesters'
    own declarations, regardless of which request the engine processed first."""

    def _own_feature(self) -> Feature:
        """A Feature whose own declared context includes 'k'."""
        return Feature(name="x", options=Options(group={"g": 1}, context={"k": "v"}))

    def _inherited_feature(self) -> Feature:
        """A Feature whose 'k' is delivered purely via inherit_from from a distinct consumer, not own."""
        consumer_options = Options(context={"k": "v"})
        feature = Feature(name="x", options=Options(group={"g": 1}))
        feature.options.inherit_from(consumer_options, inherit_context_keys=frozenset({"k"}))
        assert "k" not in feature.options.own_context_keys
        return feature

    def test_own_first_inherited_second_survivor_gains_k_as_own(self) -> None:
        engine = _intake_engine()
        own_feature = self._own_feature()
        inherited_feature = self._inherited_feature()

        assert engine.add_feature_to_collection(FeatureGroup, own_feature, None) is True
        inherited_group_keys_before = own_feature.options.inherited_group_keys
        inherited_context_keys_before = own_feature.options.inherited_context_keys
        assert engine.add_feature_to_collection(FeatureGroup, inherited_feature, None) is False

        (survivor,) = engine.feature_group_collection[FeatureGroup]
        assert survivor is own_feature
        assert "k" in survivor.options.own_context_keys
        assert survivor.options.inherited_group_keys == inherited_group_keys_before
        assert survivor.options.inherited_context_keys == inherited_context_keys_before

    def test_inherited_first_own_second_survivor_gains_k_as_own(self) -> None:
        engine = _intake_engine()
        inherited_feature = self._inherited_feature()
        own_feature = self._own_feature()

        assert engine.add_feature_to_collection(FeatureGroup, inherited_feature, None) is True
        inherited_group_keys_before = inherited_feature.options.inherited_group_keys
        inherited_context_keys_before = inherited_feature.options.inherited_context_keys
        assert engine.add_feature_to_collection(FeatureGroup, own_feature, None) is False

        (survivor,) = engine.feature_group_collection[FeatureGroup]
        assert survivor is inherited_feature
        assert "k" in survivor.options.own_context_keys
        assert survivor.options.inherited_group_keys == inherited_group_keys_before
        assert survivor.options.inherited_context_keys == inherited_context_keys_before


class TestConsumerAttributionsMergeOrderIndependence:
    """A merge of two value-equal Feature requests must union both requests' consumer_attributions."""

    @pytest.mark.parametrize("order", ["a-first", "b-first"])
    def test_merge_unions_consumer_attributions_regardless_of_order(self, order: str) -> None:
        engine = _intake_engine()
        a = Feature(name="x", options=Options(group={"g": 1}))
        b = Feature(name="x", options=Options(group={"g": 1}))
        a.add_consumer_attribution("ConsumerA", frozenset({"a"}))
        b.add_consumer_attribution("ConsumerB", frozenset())

        first, second = (a, b) if order == "a-first" else (b, a)

        assert engine.add_feature_to_collection(FeatureGroup, first, None) is True
        assert engine.add_feature_to_collection(FeatureGroup, second, None) is False

        (survivor,) = engine.feature_group_collection[FeatureGroup]
        assert survivor is first
        assert set(survivor.consumer_attributions) == {
            ("ConsumerA", frozenset({"a"})),
            ("ConsumerB", frozenset()),
        }

    def test_copy_taken_before_merge_does_not_see_later_merge(self) -> None:
        """A `copy()` of the survivor made before the merge must not observe the merged-in attribution."""
        engine = _intake_engine()
        a = Feature(name="x", options=Options(group={"g": 1}))
        b = Feature(name="x", options=Options(group={"g": 1}))
        a.add_consumer_attribution("ConsumerA", frozenset({"a"}))
        b.add_consumer_attribution("ConsumerB", frozenset())

        assert engine.add_feature_to_collection(FeatureGroup, a, None) is True
        stored = copy(a)
        assert engine.add_feature_to_collection(FeatureGroup, b, None) is False

        assert stored.consumer_attributions == [("ConsumerA", frozenset({"a"}))]
