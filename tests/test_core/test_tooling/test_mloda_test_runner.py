"""MlodaTestRunner.run_api must populate RunResult.runner with the ExecutionOrchestrator."""

from typing import Any, Optional

from mloda.provider import BaseInputData
from mloda.provider import DataCreator
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.user import Feature
from mloda.user import Features
from mloda.user import ParallelizationMode
from mloda.user import PluginCollector
from mloda.core.runtime.run import ExecutionOrchestrator
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from tests.test_core.test_tooling import MlodaTestRunner, PARALLELIZATION_MODES_ALL


class RunnerFieldRootFeature(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): [1]}


@PARALLELIZATION_MODES_ALL
class TestRunApiRunnerField:
    def test_run_api_populates_runner(self, modes: set[ParallelizationMode], flight_server: Any) -> None:
        features = Features([Feature(name=RunnerFieldRootFeature.get_class_name())])

        result = MlodaTestRunner.run_api(
            features,
            compute_frameworks={PyArrowTable},
            parallelization_modes=modes,
            flight_server=flight_server,
            plugin_collector=PluginCollector.enabled_feature_groups({RunnerFieldRootFeature}),
        )

        assert isinstance(result.runner, ExecutionOrchestrator)
