"""End-to-end test: a caller-registered child_bootstrap callable fires exactly once inside a
real spawned MULTIPROCESSING worker process, before any command (feature group run) is
processed.

Context: a spawned child process starts with a blank interpreter and no SDK unless something
installs one, and the library itself must not install anything. Programmatic setups (e.g. an
OTel SDK) need a core seam to run arbitrary bootstrap code once inside a freshly spawned
worker, before that worker processes its first command. mlodaAPI threads a plain, picklable,
no-argument child_bootstrap callable down the same call path function_extender already takes
(run_all/session.run -> _batch_run -> _run_engine_computation -> _enter_runner_context ->
ExecutionOrchestrator.__enter__ -> CfwManager.set_child_bootstrap), and
multiprocessing_worker.worker() invokes it once, before entering its while-True command loop
(see test_multiprocessing_worker.py).

Ordering (bootstrap fires strictly before the first feature group runs, hence before any
Extender hook observes a HookContext) is asserted by having both the bootstrap and a recording
Extender append a line to the SAME file: with a single feature group, and therefore a single
spawned worker process, the two appends cannot interleave with any other process, so the file
content order is a reliable proxy for the in-process ordering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pytest

from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.hook_context import HookContext
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


class _ChildBootstrapFeatureGroup(FeatureGroup):
    """Root PythonDict FeatureGroup, the sole feature group computed in this test's run."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"child_bootstrap_e2e_col"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"child_bootstrap_e2e_col": [1, 2, 3]}


_ENABLED = PluginCollector.enabled_feature_groups({_ChildBootstrapFeatureGroup})


class _BootstrapMarker:
    """Picklable no-argument callable: appends a BOOTSTRAP line to marker_path when called.

    A module-level function closing over a Path is not picklable across the spawn boundary
    (closures aren't); a small callable class with the path as a constructor attribute is.
    """

    def __init__(self, marker_path: Path) -> None:
        self._marker_path = marker_path

    def __call__(self) -> None:
        with open(self._marker_path, "a") as handle:
            handle.write("BOOTSTRAP\n")


class _HookRecordingExtender(Extender):
    """Appends a HOOK line to the same marker file after observing calculate_feature."""

    def __init__(self, marker_path: Path, priority: int = 100) -> None:
        self.priority = priority
        self._marker_path = marker_path

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        context = HookContext.current()
        assert context is not None
        with open(self._marker_path, "a") as handle:
            handle.write("HOOK\n")
        return result


@pytest.mark.timeout(30)
class TestChildBootstrapFiresInsideSpawnedWorkerBeforeFirstCommand:
    def test_bootstrap_marker_written_before_hook_marker(self, tmp_path: Path, flight_server: Any) -> None:
        marker_path = tmp_path / "child_bootstrap_marker.txt"
        bootstrap = _BootstrapMarker(marker_path)
        extender = _HookRecordingExtender(marker_path)

        session = mloda.prepare(
            [Feature(name="child_bootstrap_e2e_col")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED,
            parallelization_modes={ParallelizationMode.MULTIPROCESSING},
        )

        session.run(
            parallelization_modes={ParallelizationMode.MULTIPROCESSING},
            function_extender={extender},
            flight_server=flight_server,
            child_bootstrap=bootstrap,
        )

        assert marker_path.exists(), "child_bootstrap never fired inside the spawned worker"
        lines = marker_path.read_text().splitlines()
        assert lines == ["BOOTSTRAP", "HOOK"]
