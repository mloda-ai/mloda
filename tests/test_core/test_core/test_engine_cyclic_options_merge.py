"""The default-equivalent merge warning compares declared options, so it must survive cyclic values.

The branch is reached exactly when the feature equality probe matched, which cycle-safe Options
equality newly makes possible for cyclic group values.
"""

from __future__ import annotations

import logging
from collections import defaultdict
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
