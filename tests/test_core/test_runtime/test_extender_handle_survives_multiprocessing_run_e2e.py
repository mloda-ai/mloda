"""E2E: in a mixed SYNC+MULTIPROCESSING run, a SYNC-only ComputeFramework subclass stays resident
in the parent process and must be built with the caller's own Extender object, live handle intact."""

from __future__ import annotations

import os
import threading
from typing import Any

import pytest

from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


class _SyncOnlyExtenderSurvivalFramework(PythonDictFramework):
    """A PythonDictFramework whose steps can only ever run resident in the parent process."""

    @classmethod
    def supported_parallelization_modes(cls) -> set[ParallelizationMode]:
        return {ParallelizationMode.SYNC}


class _ExtenderHandleSurvivalFeatureGroup(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"extender_handle_survival_e2e_col"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {_SyncOnlyExtenderSurvivalFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"extender_handle_survival_e2e_col": [1, 2, 3]}


_ENABLED = PluginCollector.enabled_feature_groups({_ExtenderHandleSurvivalFeatureGroup})


class _CallSitePidProbeExtender(Extender):
    """Builds a live, unpicklable handle eagerly and strips it in __getstate__; records every
    call's pid and whether the handle was still live at call time."""

    def __init__(self) -> None:
        self.handle: Any = threading.Lock()
        self.call_pids: list[int] = []
        self.handle_was_live_during_call: list[bool] = []

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        self.call_pids.append(os.getpid())
        self.handle_was_live_during_call.append(self.handle is not None)
        return func(*args, **kwargs)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["handle"] = None
        return state


@pytest.mark.timeout(30)
class TestCallersOwnExtenderRunsInParentForASyncOnlyFramework:
    def test_probe_records_the_call_from_the_parent_pid_with_its_handle_still_live(self, flight_server: Any) -> None:
        probe = _CallSitePidProbeExtender()
        live_handle = probe.handle

        session = mloda.prepare(
            [Feature(name="extender_handle_survival_e2e_col")],
            compute_frameworks=["_SyncOnlyExtenderSurvivalFramework"],
            plugin_collector=_ENABLED,
            parallelization_modes={ParallelizationMode.SYNC, ParallelizationMode.MULTIPROCESSING},
        )

        results = session.run(
            parallelization_modes={ParallelizationMode.SYNC, ParallelizationMode.MULTIPROCESSING},
            function_extender={probe},
            flight_server=flight_server,
        )

        # The discriminating assertion: against the original bug the parent ran a stripped copy
        # instead of the caller's own object, so call_pids would stay empty here.
        assert probe.call_pids == [os.getpid()], (
            "the caller's own extender object must be the one invoked in the parent process"
        )
        assert probe.handle_was_live_during_call == [True], (
            "the caller's own extender's live handle must still be intact when it is invoked"
        )
        assert probe.handle is live_handle
        # Not asserting the exact result payload: with a flight_server present, a root feature
        # group step with no downstream consumer always uploads its finished data (regardless of
        # which mode actually ran it), so the transport representation is orthogonal to this test.
        assert len(results) == 1
