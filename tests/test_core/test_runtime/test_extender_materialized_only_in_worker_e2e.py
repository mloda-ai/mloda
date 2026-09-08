"""E2E pin: a worker-bound Extender's live-handle build must fire only inside the
spawned MULTIPROCESSING worker's own pid, exactly once, never in the parent test process running
session.run(). An Extender's own instance state does not propagate back via the manager proxy, so
the extender records to a file under tmp_path instead of relying on shared identity.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


class _MaterializedOnlyInWorkerFeatureGroup(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"materialized_only_in_worker_e2e_col"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"materialized_only_in_worker_e2e_col": [1, 2, 3]}


_ENABLED = PluginCollector.enabled_feature_groups({_MaterializedOnlyInWorkerFeatureGroup})


class _HandleBuildingExtender(Extender):
    """Guarded, idempotent handle build, mirroring a realistic extender that lazily builds a live
    resource once and never rebuilds it once `handle` is set. If the old double pickle round trip
    materializes this in the parent first, the pickled instance carries a non-None handle into the
    worker, so the worker's own unpickle is silently skipped and the recorded pid stays the parent's."""

    def __init__(self, output_path: Path) -> None:
        self._output_path = output_path
        self.handle: str | None = None
        self.materialization_count = 0

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if self.handle is not None:
            return
        self.materialization_count += 1
        self.handle = f"live-handle-pid-{os.getpid()}"
        self._output_path.write_text(
            json.dumps({"pid": os.getpid(), "materialization_count": self.materialization_count})
        )


@pytest.mark.timeout(30)
class TestExtenderMaterializedOnlyInWorker:
    def test_handle_build_fires_only_in_the_worker_pid_exactly_once(self, tmp_path: Path, flight_server: Any) -> None:
        output_path = tmp_path / "materialized_only_in_worker.json"
        extender = _HandleBuildingExtender(output_path)

        session = mloda.prepare(
            [Feature(name="materialized_only_in_worker_e2e_col")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED,
            parallelization_modes={ParallelizationMode.MULTIPROCESSING},
        )

        session.run(
            parallelization_modes={ParallelizationMode.MULTIPROCESSING},
            function_extender={extender},
            flight_server=flight_server,
        )

        assert output_path.exists(), (
            "the extender was never unpickled in a worker: session.run() never dispatched to a spawned process"
        )
        recorded = json.loads(output_path.read_text())

        assert recorded["pid"] != os.getpid(), (
            "the extender's live handle was built in the parent process; it must be deferred to "
            "the worker's own unpickle (ComputeFramework.__setstate__)"
        )
        assert recorded["materialization_count"] == 1, "the guard against re-materialization did not behave as expected"
