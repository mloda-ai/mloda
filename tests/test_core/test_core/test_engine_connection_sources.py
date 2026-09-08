"""Engine._resolve_connection_sources: dict[type[ComputeFramework], ConnectionSource] resolved
from the DataAccessCollection at setup, keyed by each planned step's destination framework.
"""

from typing import Any
from unittest.mock import Mock, patch

from mloda.core.abstract_plugins.components.connection_spec import ConnectionSource, ConnectionSpec
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.core.engine import Engine
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.core.step.join_step import JoinStep
from mloda.core.core.step.transform_frame_work_step import TransformFrameworkStep
from mloda.provider import FeatureGroup
from mloda.user import Features

from tests.test_core.test_abstract_plugins.test_abstract_compute_framework import (
    BaseTestComputeFramework1,
    BaseTestComputeFramework2,
)
from tests.test_core.test_abstract_plugins.test_abstract_feature_group import BaseTestFeatureGroup1


class _ConnSourceStepFeatureGroup(FeatureGroup):
    """Module-level, picklable feature group used only to build a FeatureGroupStep."""


class _ConnSourceEngineFramework(ComputeFramework):
    """Module-level ComputeFramework subclass; accepts any live object for connection resolution tests."""

    @classmethod
    def _connection_matches(cls, conn: Any) -> bool:
        return True


def _make_engine() -> Engine:
    """A minimally constructed Engine, real resolution short-circuited by data_access_collection=None."""
    with (
        patch(
            "mloda.core.prepare.accessible_plugins.PreFilterPlugins.resolve_feature_group_compute_framework_limitations"
        ) as mocked_derived_accessible_plugins,
        patch("mloda.core.core.engine.Engine.create_setup_execution_plan"),
    ):
        mocked_derived_accessible_plugins.return_value = {
            BaseTestFeatureGroup1: [BaseTestComputeFramework1, BaseTestComputeFramework2],
        }
        compute_framework = {BaseTestComputeFramework1, BaseTestComputeFramework2}
        features = Features(["BaseTestFeature1"])
        return Engine(features, compute_framework, None)


def _tfs_step_to(cfw_class: type[ComputeFramework]) -> TransformFrameworkStep:
    return TransformFrameworkStep(
        BaseTestComputeFramework1, cfw_class, set(), _ConnSourceStepFeatureGroup, _ConnSourceStepFeatureGroup
    )


def test_connection_sources_wraps_a_registered_live_object() -> None:
    engine = _make_engine()
    live = object()
    engine.data_access_collection = DataAccessCollection(connections={live})
    engine.execution_planner = [_tfs_step_to(_ConnSourceEngineFramework)]  # type: ignore[assignment]

    connection_sources = engine._resolve_connection_sources()

    source = connection_sources[_ConnSourceEngineFramework]
    assert isinstance(source, ConnectionSource)
    assert source.live is live


def test_connection_sources_wraps_a_registered_spec() -> None:
    engine = _make_engine()
    spec = ConnectionSpec(_ConnSourceEngineFramework)
    engine.data_access_collection = DataAccessCollection(connections={spec})
    engine.execution_planner = [_tfs_step_to(_ConnSourceEngineFramework)]  # type: ignore[assignment]

    connection_sources = engine._resolve_connection_sources()

    source = connection_sources[_ConnSourceEngineFramework]
    assert source.spec is spec


def test_engine_exposes_connection_sources_not_tfs_connection_map() -> None:
    engine = _make_engine()

    assert hasattr(engine, "connection_sources")
    assert not hasattr(engine, "tfs_connection_map")


def test_connection_sources_reached_only_through_a_feature_group_step() -> None:
    engine = _make_engine()
    spec = ConnectionSpec(_ConnSourceEngineFramework)
    engine.data_access_collection = DataAccessCollection(connections={spec})
    step = FeatureGroupStep(_ConnSourceStepFeatureGroup, FeatureSet([Feature("f")]), set(), _ConnSourceEngineFramework)
    engine.execution_planner = [step]  # type: ignore[assignment]

    connection_sources = engine._resolve_connection_sources()

    source = connection_sources[_ConnSourceEngineFramework]
    assert source.spec is spec


def test_connection_sources_reached_only_through_a_join_step_destination() -> None:
    engine = _make_engine()
    spec = ConnectionSpec(_ConnSourceEngineFramework)
    engine.data_access_collection = DataAccessCollection(connections={spec})
    step = JoinStep(Mock(), _ConnSourceEngineFramework, BaseTestComputeFramework1, set(), set(), set())
    engine.execution_planner = [step]  # type: ignore[assignment]

    connection_sources = engine._resolve_connection_sources()

    source = connection_sources[_ConnSourceEngineFramework]
    assert source.spec is spec
