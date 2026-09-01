"""End-to-end test: worker_index, run_id, and carrier reach HookContext inside a real spawned
MULTIPROCESSING worker process.

Two independent root FeatureGroups each get their own ComputeFramework instance and, under
ParallelizationMode.MULTIPROCESSING, their own spawned worker process. WorkerManager assigns
each worker a zero-based index in creation order (see test_worker_manager_worker_index.py),
multiprocessing_worker.worker() lands that index on the ComputeFramework instance running
inside the child process (see test_multiprocessing_worker.py), and ComputeFramework surfaces it,
together with run_id/carrier, on the HookContext an Extender observes (see
test_hook_context_wiring.py). run_id/carrier are only proven to survive the real pickle boundary
here; test_hook_context_run_id_carrier_e2e.py only covers the already-tested SYNC path.

An Extender's own instance state does not propagate back from a spawned MULTIPROCESSING child
via the manager proxy (each child gets its own pickled copy), so the recording extender here
writes the captured worker_index/run_id/carrier to a file instead, for the parent test process to
read back after the run completes. Two recording extenders share the same function_extender set
(every ComputeFramework instance in a run is constructed with the same set), so each filters by
the feature_group_class on the HookContext it observes to avoid cross-writing the other's file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pytest

from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.hook_context import HookContext
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework

_CARRIER = {"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"}


class _WorkerIndexFeatureGroupOne(FeatureGroup):
    """Root PythonDict FeatureGroup, computed alongside _WorkerIndexFeatureGroupTwo."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"worker_index_e2e_col_one"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"worker_index_e2e_col_one": [1, 2, 3]}


class _WorkerIndexFeatureGroupTwo(FeatureGroup):
    """Second, independent root PythonDict FeatureGroup, computed alongside the first."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"worker_index_e2e_col_two"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"worker_index_e2e_col_two": [4, 5, 6]}


_ENABLED = PluginCollector.enabled_feature_groups({_WorkerIndexFeatureGroupOne, _WorkerIndexFeatureGroupTwo})


class _WorkerIndexRecordingExtender(Extender):
    """Writes HookContext.current().worker_index/run_id/carrier to output_path as JSON, for
    calculate_feature calls whose feature_group_class matches target_feature_group only."""

    def __init__(self, output_path: Path, target_feature_group: type[FeatureGroup], priority: int = 100) -> None:
        self.priority = priority
        self._output_path = output_path
        self._target_feature_group_name = f"{target_feature_group.__module__}.{target_feature_group.__qualname__}"

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        context = HookContext.current()
        assert context is not None
        if context.feature_group_class == self._target_feature_group_name:
            self._output_path.write_text(
                json.dumps({"worker_index": context.worker_index, "run_id": context.run_id, "carrier": context.carrier})
            )
        return result


@pytest.mark.timeout(30)
class TestWorkerIndexReachesHookContextUnderMultiprocessing:
    def test_two_independent_root_feature_groups_each_see_a_worker_index_in_zero_or_one(
        self, tmp_path: Path, flight_server: Any
    ) -> None:
        output_one = tmp_path / "worker_index_one.txt"
        output_two = tmp_path / "worker_index_two.txt"
        extender_one = _WorkerIndexRecordingExtender(output_one, _WorkerIndexFeatureGroupOne)
        extender_two = _WorkerIndexRecordingExtender(output_two, _WorkerIndexFeatureGroupTwo)

        session = mloda.prepare(
            [
                Feature(name="worker_index_e2e_col_one"),
                Feature(name="worker_index_e2e_col_two"),
            ],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED,
            parallelization_modes={ParallelizationMode.MULTIPROCESSING},
        )

        session.run(
            parallelization_modes={ParallelizationMode.MULTIPROCESSING},
            function_extender={extender_one, extender_two},
            flight_server=flight_server,
            carrier=_CARRIER,
        )

        assert output_one.exists(), "extender_one never observed its feature group's calculate_feature call"
        assert output_two.exists(), "extender_two never observed its feature group's calculate_feature call"

        recorded_one = json.loads(output_one.read_text())
        recorded_two = json.loads(output_two.read_text())

        seen_indices = {recorded_one["worker_index"], recorded_two["worker_index"]}
        assert seen_indices == {0, 1}

        # run_id/carrier must survive the real pickle boundary into the spawned child unchanged.
        assert recorded_one["run_id"] == session.run_id
        assert recorded_two["run_id"] == session.run_id
        assert recorded_one["carrier"] == _CARRIER
        assert recorded_two["carrier"] == _CARRIER
