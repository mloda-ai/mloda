"""Tests for the MlodaTestRunner helper itself."""

from mloda.user import Feature, Features
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from tests.test_core.test_tooling import MlodaTestRunner


class TestRunApi:
    def test_run_api_populates_runner(self) -> None:
        """run_api returns the ExecutionOrchestrator that produced the results, as documented."""
        features = Features([Feature(name="ApiInputDataTestFeatureGroup_id")])
        api_data = {"TestApiInputData": {"ApiInputDataTestFeatureGroup_id": [12, 2, 3]}}

        result = MlodaTestRunner.run_api(
            features,
            compute_frameworks={PyArrowTable},
            api_data=api_data,
        )

        assert result.runner is not None
        assert result.results == result.runner.get_result()
        assert result.artifacts == result.runner.get_artifacts()
