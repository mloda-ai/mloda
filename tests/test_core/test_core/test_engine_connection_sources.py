"""Engine._resolve_connection_sources: dict[type[ComputeFramework], ConnectionSource] resolved
from the DataAccessCollection at setup, keyed by TFS destination framework.
"""

from typing import Any
from unittest.mock import Mock, patch

from mloda.core.abstract_plugins.components.connection_spec import ConnectionSource, ConnectionSpec
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.core.engine import Engine
from mloda.user import Features

from tests.test_core.test_abstract_plugins.test_abstract_compute_framework import (
    BaseTestComputeFramework1,
    BaseTestComputeFramework2,
)
from tests.test_core.test_abstract_plugins.test_abstract_feature_group import BaseTestFeatureGroup1


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


def _plan_with_tfs_to(cfw_class: type[ComputeFramework]) -> Any:
    plan = Mock()
    plan.tfs_collection = {"x": Mock(to_framework=cfw_class)}
    return plan


def test_connection_sources_wraps_a_registered_live_object() -> None:
    engine = _make_engine()
    live = object()
    engine.data_access_collection = DataAccessCollection(connections={live})
    engine.execution_planner = _plan_with_tfs_to(_ConnSourceEngineFramework)

    connection_sources = engine._resolve_connection_sources()

    source = connection_sources[_ConnSourceEngineFramework]
    assert isinstance(source, ConnectionSource)
    assert source.live is live


def test_connection_sources_wraps_a_registered_spec() -> None:
    engine = _make_engine()
    spec = ConnectionSpec(_ConnSourceEngineFramework)
    engine.data_access_collection = DataAccessCollection(connections={spec})
    engine.execution_planner = _plan_with_tfs_to(_ConnSourceEngineFramework)

    connection_sources = engine._resolve_connection_sources()

    source = connection_sources[_ConnSourceEngineFramework]
    assert source.spec is spec


def test_engine_exposes_connection_sources_not_tfs_connection_map() -> None:
    engine = _make_engine()

    assert hasattr(engine, "connection_sources")
    assert not hasattr(engine, "tfs_connection_map")
