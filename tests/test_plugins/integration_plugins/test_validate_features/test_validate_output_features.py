import logging
import time
from typing import Any, Dict, List, Optional, Set

import pytest

from mloda.user import Feature
from mloda.user import Features
from mloda.provider import FeatureSet
from mloda.user import ParallelizationMode
from mloda.steward import ExtenderHook, Extender
from tests.test_core.test_tooling import MlodaTestRunner, PARALLELIZATION_MODES_SYNC_THREADING
from tests.test_documentation.test_documentation import DokuExtender
from tests.test_plugins.integration_plugins.test_validate_features.example_validator import (
    BaseValidateOutputFeaturesBase,
    BaseValidateOutputFeaturesBaseNegativePandera,
)

logger = logging.getLogger(__name__)


class BaseValidateOutputFeaturesBaseNegative(BaseValidateOutputFeaturesBase):
    """Negative test case for output features validation."""

    @classmethod
    def validate_output_features(cls, data: Any, features: FeatureSet) -> Optional[bool]:
        """This function should be used to validate the output data."""
        if len(data[cls.get_class_name()]) == 3:
            raise ValueError("Data should not have 3 elements.")
        return True


class ValidateOutputFeatureExtender(DokuExtender):
    def wraps(self) -> Set[ExtenderHook]:
        return {ExtenderHook.VALIDATE_OUTPUT_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        _measured_time = f"Time taken: {time.time() - start}"
        logger.error(_measured_time)
        print(_measured_time)
        return result


@PARALLELIZATION_MODES_SYNC_THREADING
class TestValidateOutputFeatures:
    def get_features(self, feature_list: List[str], options: Dict[str, Any] = {}) -> Features:
        return Features([Feature(name=f_name, options=options, initial_requested_data=True) for f_name in feature_list])

    def test_basic_validate_output_features(self, modes: Set[ParallelizationMode], flight_server: Any) -> None:
        features = self.get_features(["BaseValidateOutputFeaturesBase"])
        MlodaTestRunner.run_api_simple(features, parallelization_modes=modes, flight_server=flight_server)

        features = self.get_features(["BaseValidateOutputFeaturesBaseNegative"])
        with pytest.raises(Exception):
            MlodaTestRunner.run_api_simple(features, parallelization_modes=modes, flight_server=flight_server)

    def test_pandera_validate_output_features(self, modes: Set[ParallelizationMode], flight_server: Any) -> None:
        features = self.get_features(["BaseValidateOutputFeaturesBaseNegativePandera"])

        with pytest.raises(Exception) as excinfo:
            MlodaTestRunner.run_api_simple(features, parallelization_modes=modes, flight_server=flight_server)

        assert "in_range(1, 2)" in str(excinfo.value)
        assert "BaseValidateOutputFeaturesBaseNegativePandera" in str(excinfo.value)

    def test_extender_validate_output_features(
        self, modes: Set[ParallelizationMode], flight_server: Any, caplog: Any
    ) -> None:
        features = self.get_features(["BaseValidateOutputFeaturesBase"])

        MlodaTestRunner.run_api_simple(
            features,
            parallelization_modes=modes,
            flight_server=flight_server,
            function_extender={ValidateOutputFeatureExtender()},
        )

        if ParallelizationMode.MULTIPROCESSING not in modes:
            assert "Time taken" in caplog.text
