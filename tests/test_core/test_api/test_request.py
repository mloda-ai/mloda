import sqlite3
from typing import Any
from unittest.mock import patch, Mock
import pytest

from mloda.provider import ComputeFramework, FeatureSet, DataCreatorFeatureGroup
from mloda.user import Features
from mloda.user import DataAccessCollection, Feature, Options, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda.user import mlodaAPI
from mloda.core.api.request import SetupConfigurationError
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

    def init_with_no_params(self, features: list[str | Feature]) -> mlodaAPI:
        api_request = mlodaAPI(features)

        assert isinstance(api_request.features, Features)
        assert isinstance(api_request.compute_framework, set)
        assert api_request.links is None
        return api_request

    def test_init_with_all_params(self) -> None:
        features = [Feature("BaseTestGraphFeatureGroup3")]

        compute_fws = sorted(fw.get_class_name() for fw in get_all_subclasses(ComputeFramework))
        links = {
            Link.inner(
                JoinSpec(BaseLinkTestFeatureGroup1, Index(tuple(["Index1"]))),
                JoinSpec(BaseTestGraphFeatureGroup3, Index(tuple(["Index1"]))),
            )
        }

        api_request = mlodaAPI(features, compute_fws, links)
        assert isinstance(api_request.features, Features)
        assert len(api_request.compute_framework) == len(get_all_subclasses(ComputeFramework))
        assert api_request.links is not None
        assert len(api_request.links) == 1

    def test_setup_engine(self, features: list[str | Feature]) -> None:
        with patch("mloda.core.core.engine.Engine.create_setup_execution_plan"):
            mloda_api = self.init_with_no_params(features)
            assert isinstance(mloda_api.engine, Engine)

    def test_copy_features_default_behavior(self) -> None:
        """Test that by default, features are deep copied and mutations don't affect original."""
        with patch("mloda.core.core.engine.Engine.create_setup_execution_plan"):
            original_features: list[Feature | str] = [Feature("test_feature")]
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
            original_features: list[Feature | str] = [Feature("test_feature")]
            original_feature = original_features[0]
            assert isinstance(original_feature, Feature)

            # Create mlodaAPI with copy_features=False
            api_request = mlodaAPI(original_features, copy_features=False)

            # Verify the feature in the mlodaAPI is the same object (was NOT copied)
            api_feature = list(api_request.features)[0]
            assert api_feature is original_feature
            assert api_feature.name == original_feature.name


class TestBareStringRejected:
    def test_api_rejects_bare_string(self) -> None:
        with pytest.raises(SetupConfigurationError):
            mlodaAPI.prepare("abc")
        with pytest.raises(SetupConfigurationError):
            mlodaAPI.run_all("abc")


class _ReadOnlyArgsError(Exception):
    """Redacting this fails: args is a property without a setter."""

    @property
    def args(self) -> tuple[Any, ...]:  # type: ignore[override]
        return ("a-registered-secret-value",)


class TestCredentialRedactionExit:
    def test_the_scope_resets_even_when_redacting_the_exception_raises(self) -> None:
        import contextvars

        from mloda.core.abstract_plugins.components.credential_scrub import current_registered_secrets
        from mloda.core.api.request import _CredentialRedaction

        def body() -> frozenset[str]:
            with pytest.raises(Exception):
                with _CredentialRedaction(frozenset({"a-registered-secret-value"})):
                    raise _ReadOnlyArgsError()
            return current_registered_secrets()

        assert contextvars.copy_context().run(body) == frozenset()


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


_CONN_SEEN: list[tuple[Any, str | None]] = []


class ConnSugarRoot(DataCreatorFeatureGroup):
    """Sqlite root that records each feature's connection object and bound connection_handle."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"conn_sugar_col"}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {SqliteFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        for feature in features.features:
            _CONN_SEEN.append((feature.connection, feature.options.get("connection_handle")))
        return {"conn_sugar_col": [1, 2]}


def _run_conn_sugar(features: list[Feature], **kwargs: Any) -> dict[int, str | None]:
    _CONN_SEEN.clear()
    result = mloda.run_all(
        features,
        compute_frameworks=[SqliteFramework],
        plugin_collector=PluginCollector.enabled_feature_groups({ConnSugarRoot}),
        parallelization_modes={ParallelizationMode.SYNC},
        **kwargs,
    )
    assert result[0].column("conn_sugar_col").to_pylist() == [1, 2]
    return {id(conn): handle for conn, handle in _CONN_SEEN}


class TestFeatureConnectionIntake:
    @pytest.fixture
    def conns(self) -> Any:
        first = sqlite3.connect(":memory:")
        second = sqlite3.connect(":memory:")
        yield first, second
        first.close()
        second.close()

    def test_features_differing_only_in_connection_each_bind_their_own_in_request_order(self, conns: Any) -> None:
        first, second = conns
        seen = _run_conn_sugar(
            [Feature("conn_sugar_col", connection=first), Feature("conn_sugar_col", connection=second)]
        )
        assert seen == {id(first): "_feature_connection_0", id(second): "_feature_connection_1"}

    def test_features_sharing_one_options_instance_bind_their_own_and_leave_it_untouched(self, conns: Any) -> None:
        first, second = conns
        shared = Options({"k": "v"})
        seen = _run_conn_sugar(
            [
                Feature("conn_sugar_col", options=shared, connection=first),
                Feature("conn_sugar_col", options=shared, connection=second),
            ]
        )
        assert seen == {id(first): "_feature_connection_0", id(second): "_feature_connection_1"}
        assert shared.get("connection_handle") is None

    def test_a_user_named_connection_keeps_its_handle_and_the_users_collection_is_unchanged(self, conns: Any) -> None:
        first, second = conns
        dac = DataAccessCollection(connections={"mine": first})
        seen = _run_conn_sugar(
            [Feature("conn_sugar_col", connection=first), Feature("conn_sugar_col", connection=second)],
            data_access_collection=dac,
        )
        assert seen == {id(first): "mine", id(second): "_feature_connection_1"}
        assert dac.connections == {"mine": first}
        assert dac._auto_handles == set()

    def test_copy_features_true_leaves_the_users_feature_without_a_handle(self, conns: Any) -> None:
        first, _ = conns
        feature = Feature("conn_sugar_col", connection=first)
        seen = _run_conn_sugar([feature])
        assert seen == {id(first): "_feature_connection_0"}
        assert feature.options.get("connection_handle") is None

    def test_copy_features_false_rederives_the_handle_on_every_run(self, conns: Any) -> None:
        first, _ = conns
        feature = Feature("conn_sugar_col", connection=first)
        assert _run_conn_sugar([feature], copy_features=False) == {id(first): "_feature_connection_0"}
        dac = DataAccessCollection(connections={"mine": first})
        assert _run_conn_sugar([feature], copy_features=False, data_access_collection=dac) == {id(first): "mine"}

    def test_names_follow_request_order_not_the_set_order_of_a_user_collection_and_repeat_across_runs(
        self, conns: Any
    ) -> None:
        first, second = conns
        extra = sqlite3.connect(":memory:")
        try:
            features = [Feature("conn_sugar_col", connection=first), Feature("conn_sugar_col", connection=second)]
            expected = {id(first): "_feature_connection_0", id(second): "_feature_connection_1"}
            for _ in range(2):
                dac = DataAccessCollection({second, extra, first})
                assert _run_conn_sugar(features, data_access_collection=dac) == expected
        finally:
            extra.close()


class ConnValueRoot(DataCreatorFeatureGroup):
    """Sqlite root whose value is the connection's user_version, so each connection yields its own frame."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"conn_value_col"}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {SqliteFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        connection = next(iter(features.features)).connection
        return {"conn_value_col": [connection.execute("PRAGMA user_version").fetchone()[0]]}


class TestFrameForConnectionFeature:
    def test_frame_for_a_connection_feature_returns_its_own_frame_with_copy_features(self) -> None:
        first = sqlite3.connect(":memory:")
        second = sqlite3.connect(":memory:")
        try:
            first.execute("PRAGMA user_version = 11")
            second.execute("PRAGMA user_version = 22")
            f1 = Feature("conn_value_col", connection=first)
            f2 = Feature("conn_value_col", connection=second)
            result = mloda.run_all(
                [f1, f2],
                compute_frameworks=[SqliteFramework],
                plugin_collector=PluginCollector.enabled_feature_groups({ConnValueRoot}),
                parallelization_modes={ParallelizationMode.SYNC},
            )
            assert result.frame_for(f1).column("conn_value_col").to_pylist() == [11]
            assert result.frame_for(f2).column("conn_value_col").to_pylist() == [22]
        finally:
            first.close()
            second.close()
