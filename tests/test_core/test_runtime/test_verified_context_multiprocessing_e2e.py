"""E2E: tenant_id/project_id/principal set via verified_context() reach HookContext inside a
real spawned MULTIPROCESSING worker process.

An Extender's own instance state does not propagate back from a spawned child via the manager
proxy, so the recording extender here writes captured values to a file instead, read back by
the parent process after the run. worker_index/pid are also recorded so the test proves the
values actually crossed the pickle boundary, rather than passing vacuously if MULTIPROCESSING
silently degraded to in-process execution.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import pytest

from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.hook_context import HookContext
from mloda.core.abstract_plugins.verified_context import verified_context
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


class _VerifiedContextMultiprocessingFeatureGroup(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"verified_context_mp_e2e_col"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"verified_context_mp_e2e_col": [1, 2, 3]}


_ENABLED = PluginCollector.enabled_feature_groups({_VerifiedContextMultiprocessingFeatureGroup})


class _VerifiedContextRecordingExtender(Extender):
    """Writes tenant_id/project_id/principal to output_path as JSON."""

    def __init__(self, output_path: Path, priority: int = 100) -> None:
        self.priority = priority
        self._output_path = output_path

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        context = HookContext.current()
        assert context is not None
        self._output_path.write_text(
            json.dumps(
                {
                    "tenant_id": context.tenant_id,
                    "project_id": context.project_id,
                    "principal": context.principal,
                    "worker_index": context.worker_index,
                    "pid": os.getpid(),
                }
            )
        )
        return result


@pytest.mark.timeout(30)
class TestVerifiedContextReachesHookContextUnderMultiprocessing:
    def test_tenant_project_principal_survive_the_pickle_boundary_into_a_spawned_worker(
        self, tmp_path: Path, flight_server: Any
    ) -> None:
        output_path = tmp_path / "verified_context.txt"
        extender = _VerifiedContextRecordingExtender(output_path)
        parent_pid = os.getpid()

        session = mloda.prepare(
            [Feature(name="verified_context_mp_e2e_col")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED,
            parallelization_modes={ParallelizationMode.MULTIPROCESSING},
        )

        with verified_context(tenant_id="acme", project_id="proj1", principal="hash123"):
            session.run(
                parallelization_modes={ParallelizationMode.MULTIPROCESSING},
                function_extender={extender},
                flight_server=flight_server,
            )

        assert output_path.exists(), "extender never observed the feature group's calculate_feature call"

        recorded = json.loads(output_path.read_text())
        assert recorded["tenant_id"] == "acme"
        assert recorded["project_id"] == "proj1"
        assert recorded["principal"] == "hash123"
        # Proves this actually crossed the pickle boundary into a real spawned worker, not an
        # in-process degradation that would write the same JSON regardless.
        assert recorded["worker_index"] is not None
        assert recorded["pid"] != parent_pid
