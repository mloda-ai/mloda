"""Tests for mlodaAPI.plan_id/plan_context (minted once per session) and per-run run_id."""

from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest
from mloda.core.abstract_plugins.plan_context import PlanContext

from mloda.core.abstract_plugins.verified_context import verified_context
from mloda.core.runtime.run import ExecutionOrchestrator
from mloda.user import mlodaAPI
from tests.helpers.uuid7_assertions import assert_valid_uuid7


class TestSessionPlanIdMintedAtConstruction:
    def test_direct_construction_mints_a_valid_uuid7_plan_id(self) -> None:
        with patch("mloda.core.core.engine.Engine.create_setup_execution_plan"):
            session = mlodaAPI(["some_feature"])

        assert isinstance(session.plan_id, str)
        assert_valid_uuid7(session.plan_id)

    def test_prepare_mints_a_valid_uuid7_plan_id(self) -> None:
        with patch("mloda.core.core.engine.Engine.create_setup_execution_plan"):
            session = mlodaAPI.prepare(["some_feature"])

        assert isinstance(session.plan_id, str)
        assert_valid_uuid7(session.plan_id)

    def test_session_no_longer_has_a_run_id_attribute(self) -> None:
        with patch("mloda.core.core.engine.Engine.create_setup_execution_plan"):
            session = mlodaAPI(["some_feature"])

        assert not hasattr(session, "run_id")


class TestSessionPlanContext:
    def test_plan_context_carries_plan_id_and_a_utc_created_at(self) -> None:
        before = datetime.now(timezone.utc)
        with patch("mloda.core.core.engine.Engine.create_setup_execution_plan"):
            session = mlodaAPI(["some_feature"])

        assert isinstance(session.plan_context, PlanContext)
        assert session.plan_context.plan_id == session.plan_id
        assert session.plan_context.created_at.tzinfo is not None
        assert before <= session.plan_context.created_at <= datetime.now(timezone.utc)

    def test_plan_context_reads_the_verified_identity_at_construction(self) -> None:
        with patch("mloda.core.core.engine.Engine.create_setup_execution_plan"):
            with verified_context(tenant_id="acme", project_id="proj1", principal="hash123"):
                session = mlodaAPI(["some_feature"])

        ctx = session.plan_context
        assert (ctx.tenant_id, ctx.project_id, ctx.principal) == ("acme", "proj1", "hash123")

    def test_plan_context_identity_is_none_without_a_scope(self) -> None:
        with patch("mloda.core.core.engine.Engine.create_setup_execution_plan"):
            session = mlodaAPI(["some_feature"])

        ctx = session.plan_context
        assert (ctx.tenant_id, ctx.project_id, ctx.principal) == (None, None, None)


class TestSessionPlanIdUniquePerSession:
    def test_two_sessions_get_different_plan_ids(self) -> None:
        with patch("mloda.core.core.engine.Engine.create_setup_execution_plan"):
            session_a = mlodaAPI(["some_feature"])
            session_b = mlodaAPI(["some_feature"])

        assert session_a.plan_id != session_b.plan_id


class TestSessionPlanIdStableAcrossRunsRunIdPerRun:
    def test_plan_id_unchanged_and_run_id_differs_per_run_call(self) -> None:
        with patch("mloda.core.core.engine.Engine.create_setup_execution_plan"):
            session = mlodaAPI(["some_feature"])

        first_plan_id = session.plan_id
        mock_orchestrator = Mock(spec=ExecutionOrchestrator)

        with (
            patch.object(session, "_setup_engine_runner", return_value=mock_orchestrator),
            patch.object(session, "_run_engine_computation") as compute,
        ):
            session.run()
            session.run()

        contexts = [call.kwargs["run_context"] for call in compute.call_args_list]
        assert session.plan_id == first_plan_id
        assert len(contexts) == 2
        assert contexts[0].run_id != contexts[1].run_id
        assert {c.plan_id for c in contexts} == {first_plan_id}

    def test_build_run_context_mints_a_fresh_uuid7_run_id_each_call(self) -> None:
        with patch("mloda.core.core.engine.Engine.create_setup_execution_plan"):
            session = mlodaAPI(["some_feature"])

        first = session._build_run_context(None, None)
        second = session._build_run_context(None, None)

        assert first.run_id is not None and second.run_id is not None
        assert_valid_uuid7(first.run_id)
        assert first.run_id != second.run_id
        assert first.plan_id == second.plan_id == session.plan_id

    @pytest.mark.parametrize("field", ["run_id", "plan_id"])
    def test_run_context_carries_ids_as_strings(self, field: str) -> None:
        with patch("mloda.core.core.engine.Engine.create_setup_execution_plan"):
            session = mlodaAPI(["some_feature"])

        assert isinstance(getattr(session._build_run_context(None, None), field), str)
