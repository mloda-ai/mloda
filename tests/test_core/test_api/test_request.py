from typing import Union
from unittest.mock import patch, Mock, MagicMock
import pytest

from mloda.provider import ComputeFramework
from mloda.user import Features
from mloda.user import Feature
from mloda.user import mlodaAPI
from mloda.core.core.engine import Engine
from mloda.core.runtime.run import ExecutionOrchestrator
from mloda.user import Index
from mloda.user import Link, JoinSpec
from tests.test_core.test_setup.test_graph_builder import BaseTestGraphFeatureGroup3
from tests.test_core.test_setup.test_link_resolver import BaseLinkTestFeatureGroup1
from mloda.core.abstract_plugins.components.utils import get_all_subclasses


class TestmlodaAPI:
    @pytest.fixture
    def features(self) -> list[str]:
        return ["some_feature"]

    def init_with_no_params(self, features: list[Union[str, Feature]]) -> mlodaAPI:
        api_request = mlodaAPI(features)

        assert isinstance(api_request.features, Features)
        assert isinstance(api_request.compute_framework, set)
        assert api_request.links is None
        return api_request

    def test_init_with_all_params(self) -> None:
        features = [Feature("BaseTestGraphFeatureGroup3")]

        compute_fws = [fw.get_class_name() for fw in get_all_subclasses(ComputeFramework)]
        links = {
            Link.inner(
                JoinSpec(BaseLinkTestFeatureGroup1, Index(tuple(["Index1"]))),
                JoinSpec(BaseTestGraphFeatureGroup3, Index(tuple(["Index1"]))),
            )
        }

        api_request = mlodaAPI(features, compute_fws, links)  # type: ignore
        assert isinstance(api_request.features, Features)
        assert len(api_request.compute_framework) == len(get_all_subclasses(ComputeFramework))
        assert api_request.links is not None
        assert len(api_request.links) == 1

    def test_setup_engine(self, features: list[Union[str, Feature]]) -> None:
        with patch("mloda.core.core.engine.Engine.create_setup_execution_plan"):
            mloda_api = self.init_with_no_params(features)
            assert isinstance(mloda_api.engine, Engine)

    def test_copy_features_default_behavior(self) -> None:
        """Test that by default, features are deep copied and mutations don't affect original."""
        with patch("mloda.core.core.engine.Engine.create_setup_execution_plan"):
            original_features: list[Union[Feature, str]] = [Feature("test_feature")]
            original_feature = original_features[0]
            assert isinstance(original_feature, Feature)

            # Create mlodaAPI with default copy_features=True
            api_request = mlodaAPI(original_features)

            # Verify the feature in the mlodaAPI is a different object (was deep copied)
            api_feature = list(api_request.features)[0]
            assert api_feature is not original_feature
            assert api_feature.name == original_feature.name

    def test_copy_features_false_no_copy(self) -> None:
        """Test that copy_features=False does not create a copy."""
        with patch("mloda.core.core.engine.Engine.create_setup_execution_plan"):
            original_features: list[Union[Feature, str]] = [Feature("test_feature")]
            original_feature = original_features[0]
            assert isinstance(original_feature, Feature)

            # Create mlodaAPI with copy_features=False
            api_request = mlodaAPI(original_features, copy_features=False)

            # Verify the feature in the mlodaAPI is the same object (was NOT copied)
            api_feature = list(api_request.features)[0]
            assert api_feature is original_feature
            assert api_feature.name == original_feature.name


class TestSetupEngineRunnerReturnsRunner:
    def test_setup_engine_runner_returns_execution_orchestrator(self) -> None:
        """_setup_engine_runner should return an ExecutionOrchestrator instance (not None)."""
        with patch("mloda.core.core.engine.Engine.create_setup_execution_plan"):
            api = mlodaAPI(["some_feature"])

        mock_orchestrator = Mock(spec=ExecutionOrchestrator)
        with patch.object(api.engine, "compute", return_value=mock_orchestrator):
            result = api._setup_engine_runner()

        assert isinstance(result, ExecutionOrchestrator), (
            "_setup_engine_runner must return the ExecutionOrchestrator, not None"
        )


class TestBatchRunReturnsRunner:
    def test_batch_run_returns_execution_orchestrator(self) -> None:
        """_batch_run should return an ExecutionOrchestrator instance (not None)."""
        with patch("mloda.core.core.engine.Engine.create_setup_execution_plan"):
            api = mlodaAPI(["some_feature"])

        mock_orchestrator = Mock(spec=ExecutionOrchestrator)
        with (
            patch.object(api, "_setup_engine_runner", return_value=mock_orchestrator),
            patch.object(api, "_run_engine_computation"),
        ):
            result = api._batch_run()

        assert isinstance(result, ExecutionOrchestrator), "_batch_run must return the ExecutionOrchestrator, not None"


class TestShutdownRunnerManagerDeleted:
    def test_mloda_api_has_no_shutdown_runner_manager_method(self) -> None:
        """mlodaAPI should not have a _shutdown_runner_manager method."""
        assert not hasattr(mlodaAPI, "_shutdown_runner_manager"), (
            "_shutdown_runner_manager should be deleted to avoid double shutdown"
        )
