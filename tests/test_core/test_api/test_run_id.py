"""Tests for mlodaAPI.run_id: a UUIDv7 minted once at session construction."""

from unittest.mock import Mock, patch

from mloda.core.runtime.run import ExecutionOrchestrator
from mloda.user import mlodaAPI
from tests.helpers.uuid7_assertions import assert_valid_uuid7


class TestSessionRunIdMintedAtConstruction:
    def test_direct_construction_mints_a_valid_uuid7_run_id(self) -> None:
        with patch("mloda.core.core.engine.Engine.create_setup_execution_plan"):
            session = mlodaAPI(["some_feature"])

        assert isinstance(session.run_id, str)
        assert_valid_uuid7(session.run_id)

    def test_prepare_mints_a_valid_uuid7_run_id(self) -> None:
        with patch("mloda.core.core.engine.Engine.create_setup_execution_plan"):
            session = mlodaAPI.prepare(["some_feature"])

        assert isinstance(session.run_id, str)
        assert_valid_uuid7(session.run_id)


class TestSessionRunIdUniquePerSession:
    def test_two_sessions_get_different_run_ids(self) -> None:
        with patch("mloda.core.core.engine.Engine.create_setup_execution_plan"):
            session_a = mlodaAPI(["some_feature"])
            session_b = mlodaAPI(["some_feature"])

        assert session_a.run_id != session_b.run_id


class TestSessionRunIdStableAcrossRuns:
    def test_run_id_unchanged_after_multiple_run_calls(self) -> None:
        with patch("mloda.core.core.engine.Engine.create_setup_execution_plan"):
            session = mlodaAPI(["some_feature"])

        first_run_id = session.run_id
        mock_orchestrator = Mock(spec=ExecutionOrchestrator)

        with (
            patch.object(session, "_setup_engine_runner", return_value=mock_orchestrator),
            patch.object(session, "_run_engine_computation"),
        ):
            session.run()
            session.run()

        assert session.run_id == first_run_id
