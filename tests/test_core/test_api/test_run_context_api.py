"""E2E tests for RunContext plumbing through mlodaAPI, CfwManager, and _build_run_context()."""

from typing import Any, Optional

from mloda.core.abstract_plugins.run_context import RunContext
from mloda.core.api.request import mlodaAPI
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework

_CARRIER = {"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"}


def _run_context_api_bootstrap() -> None:
    """Proves child_bootstrap threads through without error; SYNC mode never invokes it."""


class _RunContextApiFeatureGroup(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
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
    def test_cfw_register_get_run_context_matches_session_run_id_carrier_and_bootstrap(self) -> None:
        session = _prepare_session()

        session.run(
            parallelization_modes={ParallelizationMode.SYNC},
            carrier=_CARRIER,
            child_bootstrap=_run_context_api_bootstrap,
        )

        assert session.runner is not None
        assert session.runner.cfw_register.get_run_context() == RunContext(
            run_id=session.run_id, carrier=_CARRIER, child_bootstrap=_run_context_api_bootstrap
        )


class TestBuildRunContextDefaultsToSessionRunId:
    def test_build_run_context_with_no_carrier_or_bootstrap_carries_only_run_id(self) -> None:
        session = _prepare_session()

        assert session._build_run_context(None, None) == RunContext(run_id=session.run_id)


class TestBatchRunWithoutRunContextUsesSessionBase:
    def test_batch_run_without_run_context_falls_back_to_session_base_context(self) -> None:
        session = _prepare_session()

        session._batch_run({ParallelizationMode.SYNC})

        assert session.runner is not None
        assert session.runner.cfw_register.get_run_context() == RunContext(run_id=session.run_id)


class TestRunUsesSessionRunIdAttribute:
    def test_run_context_run_id_follows_the_session_run_id_attribute(self) -> None:
        session = _prepare_session()
        session.run_id = "run-context-api-reassigned-run-id"

        session.run(parallelization_modes={ParallelizationMode.SYNC})

        assert session.runner is not None
        assert session.runner.cfw_register.get_run_context().run_id == "run-context-api-reassigned-run-id"
