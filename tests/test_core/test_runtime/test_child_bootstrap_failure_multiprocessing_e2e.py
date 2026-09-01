"""End-to-end test: an exception raised inside a caller-registered child_bootstrap callable must
surface through mlodaAPI.run_all's caller, not just kill the worker process silently.

Context: multiprocessing_worker.worker() invokes cfw_register.get_child_bootstrap() once before
its while-True command loop (see test_multiprocessing_worker.py and
test_child_bootstrap_multiprocessing_e2e.py for the happy path). If that callable raises, the
exception must be reported through the SAME channel every other worker failure uses
(cfw_register.set_error(...) followed by _handle_stop_command), so it survives out to run_all's
caller with its original type and message via take_error_exception() (see
test_exception_preservation.py, whose assertion style this test mirrors), instead of a generic
MlodaRunError about a worker process dying unexpectedly.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest

from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


class _ChildBootstrapFailureFeatureGroup(FeatureGroup):
    """Root PythonDict FeatureGroup; its calculate_feature never runs because the bootstrap
    callable must raise before the worker processes its first command."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"child_bootstrap_failure_col"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"child_bootstrap_failure_col": [1, 2, 3]}


_ENABLED = PluginCollector.enabled_feature_groups({_ChildBootstrapFailureFeatureGroup})


class _RaisingBootstrap:
    """Picklable no-argument callable: raises a specific RuntimeError when invoked inside the
    spawned worker process. Mirrors _BootstrapMarker's callable-class pattern in
    test_child_bootstrap_multiprocessing_e2e.py -- no free variables, so it needs no closure and
    pickles cleanly across the spawn boundary."""

    def __call__(self) -> None:
        raise RuntimeError("child bootstrap explosion: otel sdk misconfigured")


@pytest.mark.timeout(30)
class TestChildBootstrapFailureSurfacesThroughRunAll:
    def test_run_all_raises_the_original_bootstrap_exception(self, flight_server: Any) -> None:
        """FAILS today: bootstrap() raises uncaught inside worker(), the spawned process dies,
        and run_all only ever raises a generic MlodaRunError about dead worker processes, with no
        trace of "otel sdk misconfigured" -- reviewers confirmed this empirically.
        """
        with pytest.raises(RuntimeError, match="otel sdk misconfigured"):
            mloda.run_all(
                [Feature(name="child_bootstrap_failure_col")],
                compute_frameworks=["PythonDictFramework"],
                plugin_collector=_ENABLED,
                parallelization_modes={ParallelizationMode.MULTIPROCESSING},
                child_bootstrap=_RaisingBootstrap(),
                flight_server=flight_server,
            )
