"""E2E: an exception raised inside child_bootstrap must surface through run_all's caller,
not just kill the worker process silently."""

from __future__ import annotations

from typing import Any

import pytest

from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


class _ChildBootstrapFailureFeatureGroup(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"child_bootstrap_failure_col"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"child_bootstrap_failure_col": [1, 2, 3]}


_ENABLED = PluginCollector.enabled_feature_groups({_ChildBootstrapFailureFeatureGroup})


class _RaisingBootstrap:
    """Picklable no-argument callable that raises when invoked in the spawned worker."""

    def __call__(self) -> None:
        raise RuntimeError("child bootstrap explosion: otel sdk misconfigured")


@pytest.mark.timeout(30)
class TestChildBootstrapFailureSurfacesThroughRunAll:
    def test_run_all_raises_the_original_bootstrap_exception(self, flight_server: Any) -> None:
        with pytest.raises(RuntimeError, match="otel sdk misconfigured"):
            mloda.run_all(
                [Feature(name="child_bootstrap_failure_col")],
                compute_frameworks=["PythonDictFramework"],
                plugin_collector=_ENABLED,
                parallelization_modes={ParallelizationMode.MULTIPROCESSING},
                child_bootstrap=_RaisingBootstrap(),
                flight_server=flight_server,
            )
