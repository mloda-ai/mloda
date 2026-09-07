"""MlodaTestRunner.run_api must populate RunResult.runner with the ExecutionOrchestrator."""

from typing import Any

from mloda.provider import BaseInputData
from mloda.provider import DataCreator
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.user import Feature
from mloda.user import Features
from mloda.user import PluginCollector
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from tests.test_core.test_tooling import MlodaTestRunner


class RunnerFieldRootFeature(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): [1]}


def test_run_api_populates_runner() -> None:
    features = Features([Feature(name=RunnerFieldRootFeature.get_class_name())])

    result = MlodaTestRunner.run_api(
        features,
        compute_frameworks={PyArrowTable},
        plugin_collector=PluginCollector.enabled_feature_groups({RunnerFieldRootFeature}),
    )

    assert result.runner is not None
    assert result.runner.get_artifacts() is result.artifacts
