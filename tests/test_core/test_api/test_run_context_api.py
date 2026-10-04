"""E2E tests for RunContext plumbing through mlodaAPI, CfwManager, and _build_run_context()."""

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

import pytest

from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.hook_context import HookContext
from mloda.core.abstract_plugins.run_context import RunContext
from mloda.core.abstract_plugins.verified_context import verified_context
from mloda.core.api.request import mlodaAPI
from mloda.core.runtime.worker_manager import WorkerManager
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework

_CARRIER = {"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"}


def _run_context_api_bootstrap() -> None:
    """Proves child_bootstrap threads through without error; SYNC mode never invokes it."""


class _RunContextApiFeatureGroup(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"run_context_api_col"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"run_context_api_col": [1, 2, 3]}


_ENABLED = PluginCollector.enabled_feature_groups({_RunContextApiFeatureGroup})


def _prepare_session() -> mlodaAPI:
    return mloda.prepare(
        [Feature(name="run_context_api_col")],
        compute_frameworks=["PythonDictFramework"],
        plugin_collector=_ENABLED,
        parallelization_modes={ParallelizationMode.SYNC},
    )


class TestSessionRunSetsRunContextOnCfwRegister:
    def test_cfw_register_get_run_context_matches_session_plan_id_carrier_and_bootstrap(self) -> None:
        session = _prepare_session()

        session.run(
            parallelization_modes={ParallelizationMode.SYNC},
            carrier=_CARRIER,
            child_bootstrap=_run_context_api_bootstrap,
        )

        assert session.runner is not None
        got = session.runner.cfw_register.get_run_context()
        assert got.run_id is not None
        assert got.started_at is not None
        assert got == replace(
            RunContext(plan_id=session.plan_id, carrier=_CARRIER, child_bootstrap=_run_context_api_bootstrap),
            run_id=got.run_id,
            started_at=got.started_at,
            plugin_versions=got.plugin_versions,
        )


class TestBuildRunContextMintsRunIdAndCarriesPlanId:
    def test_build_run_context_with_no_carrier_or_bootstrap_carries_ids_and_started_at(self) -> None:
        session = _prepare_session()
        before = datetime.now(timezone.utc)

        built = session._build_run_context(None, None)

        assert built.run_id is not None
        assert built.plan_id == session.plan_id
        assert built.run_id != session.plan_id
        assert built.started_at is not None
        assert before <= built.started_at <= datetime.now(timezone.utc)
        assert built.carrier is None
        assert built.child_bootstrap is None

    def test_each_build_gets_a_new_run_id(self) -> None:
        session = _prepare_session()

        assert session._build_run_context(None, None).run_id != session._build_run_context(None, None).run_id


class TestBuildRunContextPicksUpActiveVerifiedContext:
    def test_build_run_context_carries_tenant_project_principal_from_active_scope(self) -> None:
        session = _prepare_session()

        with verified_context(tenant_id="acme", project_id="proj1", principal="hash123"):
            built = session._build_run_context(None, None)

        assert (built.tenant_id, built.project_id, built.principal) == ("acme", "proj1", "hash123")

    def test_build_run_context_leaves_tenant_project_principal_none_with_no_active_scope(self) -> None:
        session = _prepare_session()

        built = session._build_run_context(None, None)

        assert built.tenant_id is None
        assert built.project_id is None
        assert built.principal is None


class TestBuildRunContextGracefulShutdownTimeout:
    def test_default_graceful_shutdown_timeout_is_two_point_zero(self) -> None:
        session = _prepare_session()

        assert session._build_run_context(None, None).graceful_shutdown_timeout == 2.0

    def test_custom_graceful_shutdown_timeout_round_trips_through_build_run_context(self) -> None:
        session = _prepare_session()

        built = session._build_run_context(None, None, graceful_shutdown_timeout=7.5)

        assert built.graceful_shutdown_timeout == 7.5

    def test_custom_graceful_shutdown_timeout_round_trips_through_session_run(self) -> None:
        session = _prepare_session()

        session.run(parallelization_modes={ParallelizationMode.SYNC}, graceful_shutdown_timeout=9.5)

        assert session.runner is not None
        assert session.runner.cfw_register.get_run_context().graceful_shutdown_timeout == 9.5

    def test_default_graceful_shutdown_timeout_round_trips_through_session_run(self) -> None:
        session = _prepare_session()

        session.run(parallelization_modes={ParallelizationMode.SYNC})

        assert session.runner is not None
        assert session.runner.cfw_register.get_run_context().graceful_shutdown_timeout == 2.0

    @pytest.mark.parametrize("call_site", ["run_all", "stream_all"])
    def test_custom_graceful_shutdown_timeout_reaches_worker_manager_join_all(
        self, call_site: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorded: list[float] = []
        original_join_all = WorkerManager.join_all

        def spy_join_all(self: WorkerManager, graceful_timeout: float = 2.0) -> None:
            recorded.append(graceful_timeout)
            original_join_all(self, graceful_timeout=graceful_timeout)

        monkeypatch.setattr(WorkerManager, "join_all", spy_join_all)

        result = getattr(mloda, call_site)(
            [Feature(name="run_context_api_col")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED,
            parallelization_modes={ParallelizationMode.SYNC},
            graceful_shutdown_timeout=9.5,
        )
        if call_site == "stream_all":
            list(result)

        assert recorded == [9.5]


class TestBatchRunWithoutRunContextMintsAFreshRunId:
    def test_batch_run_without_run_context_mints_a_fresh_run_id_per_call(self) -> None:
        session = _prepare_session()

        session._batch_run({ParallelizationMode.SYNC})
        assert session.runner is not None
        first = session.runner.cfw_register.get_run_context()
        session._batch_run({ParallelizationMode.SYNC})
        second = session.runner.cfw_register.get_run_context()

        assert first.run_id is not None and second.run_id is not None
        assert first.run_id != second.run_id
        assert first.plan_id == second.plan_id == session.plan_id


class TestRunMintsRunIdPerCall:
    def test_run_context_run_id_differs_between_run_calls(self) -> None:
        session = _prepare_session()

        session.run(parallelization_modes={ParallelizationMode.SYNC})
        assert session.runner is not None
        first = session.runner.cfw_register.get_run_context().run_id
        session.run(parallelization_modes={ParallelizationMode.SYNC})
        second = session.runner.cfw_register.get_run_context().run_id

        assert first is not None and second is not None
        assert first != second

    def test_stream_run_mints_its_own_run_id(self) -> None:
        session = _prepare_session()

        session.run(parallelization_modes={ParallelizationMode.SYNC})
        assert session.runner is not None
        first = session.runner.cfw_register.get_run_context().run_id
        list(session.stream_run(parallelization_modes={ParallelizationMode.SYNC}))
        second = session.runner.cfw_register.get_run_context().run_id

        assert first is not None and second is not None
        assert first != second


class _CalculateCapture(Extender):
    def __init__(self) -> None:
        self.captured: list[HookContext] = []

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        context = HookContext.current()
        assert context is not None
        self.captured.append(context)
        return result


def _prepare_capturing_session(extender: Extender) -> mlodaAPI:
    return mloda.prepare(
        [Feature(name="run_context_api_col")],
        compute_frameworks=["PythonDictFramework"],
        plugin_collector=_ENABLED,
        parallelization_modes={ParallelizationMode.SYNC},
        function_extender={extender},
    )


_IDENTITY = {"tenant_id": "acme", "project_id": "proj1", "principal": "hash123"}
_OTHER_IDENTITY = {"tenant_id": "globex", "project_id": "proj2", "principal": "hash999"}


class TestRunInheritsPlanIdentity:
    def test_run_with_no_scope_inherits_the_preparers_identity(self) -> None:
        extender = _CalculateCapture()
        with verified_context(**_IDENTITY):
            session = _prepare_capturing_session(extender)

        session.run(parallelization_modes={ParallelizationMode.SYNC})

        (context,) = extender.captured
        assert (context.tenant_id, context.project_id, context.principal) == ("acme", "proj1", "hash123")
        assert context.plan_id == session.plan_id

    def test_build_run_context_with_no_scope_inherits_plan_identity(self) -> None:
        with verified_context(**_IDENTITY):
            session = _prepare_session()

        built = session._build_run_context(None, None)

        assert (built.tenant_id, built.project_id, built.principal) == ("acme", "proj1", "hash123")

    def test_a_different_scope_at_run_carries_the_run_identity(self) -> None:
        extender = _CalculateCapture()
        with verified_context(**_IDENTITY):
            session = _prepare_capturing_session(extender)

        with verified_context(**_OTHER_IDENTITY):
            session.run(parallelization_modes={ParallelizationMode.SYNC})

        (context,) = extender.captured
        assert (context.tenant_id, context.project_id, context.principal) == ("globex", "proj2", "hash999")
        assert context.plan_id == session.plan_id

    def test_stream_run_with_no_scope_inherits_plan_identity(self) -> None:
        extender = _CalculateCapture()
        with verified_context(**_IDENTITY):
            session = _prepare_capturing_session(extender)

        list(session.stream_run(parallelization_modes={ParallelizationMode.SYNC}))

        (context,) = extender.captured
        assert context.principal == "hash123"

    def test_prepared_without_scope_and_run_with_scope_uses_run_identity(self) -> None:
        extender = _CalculateCapture()
        session = _prepare_capturing_session(extender)

        with verified_context(**_IDENTITY):
            session.run(parallelization_modes={ParallelizationMode.SYNC})

        (context,) = extender.captured
        assert context.principal == "hash123"
