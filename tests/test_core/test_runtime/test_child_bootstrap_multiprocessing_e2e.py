"""E2E: child_bootstrap fires exactly once inside a real spawned MULTIPROCESSING worker,
before the first command runs.

Ordering is asserted by having both the bootstrap and a recording Extender append a line to
the same file: with a single feature group (single worker process), append order is a
reliable proxy for in-process ordering.
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
    """Picklable callable (not a closure) that appends a BOOTSTRAP line to marker_path."""

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
