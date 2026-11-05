"""MlodaTestRunner.run_api must populate RunResult.artifacts from the run result."""

from mloda.user import Feature
from mloda.user import Features
from mloda.user import PluginCollector
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from tests.test_core.test_artifacts.test_artifact_prepare_run import PrepareRunArtifactFeature
from tests.test_core.test_tooling import MlodaTestRunner


def test_run_api_populates_artifacts() -> None:
    features = Features([Feature(name=PrepareRunArtifactFeature.get_class_name())])

    result = MlodaTestRunner.run_api(
        features,
        compute_frameworks=[PyArrowTable],
        plugin_collector=PluginCollector.enabled_feature_groups({PrepareRunArtifactFeature}),
    )

    assert result.artifacts == {"PrepareRunArtifactFeature": "trained_model_v1"}
    assert not hasattr(result, "runner")
