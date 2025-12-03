import time
from typing import Any, Dict, List, Optional, Set
import pytest


from mloda_core.abstract_plugins.function_extender import WrapperFunctionEnum, WrapperFunctionExtender
from mloda_core.api.request import mlodaAPI
from mloda_core.runtime.flight.flight_server import FlightServer
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_core.abstract_plugins.components.parallelization_modes import ParallelizationModes
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_collection import Features
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from tests.test_plugins.integration_plugins.test_validate_features.example_validator import (
    BaseValidateOutputFeaturesBase,
    BaseValidateOutputFeaturesBaseNegativePandera,
)
from tests.test_documentation.test_documentation import DokuExtender


import logging

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
    def wraps(self) -> Set[WrapperFunctionEnum]:
        return {WrapperFunctionEnum.VALIDATE_OUTPUT_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        _measured_time = f"Time taken: {time.time() - start}"
        logger.error(_measured_time)
        print(_measured_time)
        return result


@pytest.mark.parametrize(
    "modes",
    [
        ({ParallelizationModes.SYNC}),
        ({ParallelizationModes.THREADING}),
        # ({ParallelizationModes.MULTIPROCESSING}),
    ],
)
class TestValidateOutputFeatures:
    def get_features(self, feature_list: List[str], options: Dict[str, Any] = {}) -> Features:
        return Features([Feature(name=f_name, options=options, initial_requested_data=True) for f_name in feature_list])

    def basic_runner(
        self,
        features: Features,
        parallelization_modes: Set[ParallelizationModes],
        flight_server: Any,
        function_extender: Set[WrapperFunctionExtender] = set(),
    ) -> List[Any]:
        results = mlodaAPI.run_all(
            features, {PyArrowTable}, None, None, parallelization_modes, flight_server, function_extender
        )

        # make sure all datasets are dropped on server
        if ParallelizationModes.MULTIPROCESSING in parallelization_modes:
            flight_infos = FlightServer.list_flight_infos(flight_server.location)
            assert len(flight_infos) == 0

        return results

    def test_basic_validate_output_features(self, modes: Set[ParallelizationModes], flight_server: Any) -> None:
        features = self.get_features(["BaseValidateOutputFeaturesBase"])
        self.basic_runner(features, modes, flight_server)

        features = self.get_features(["BaseValidateOutputFeaturesBaseNegative"])
        with pytest.raises(Exception):
            self.basic_runner(features, modes, flight_server)

    def test_pandera_validate_output_features(self, modes: Set[ParallelizationModes], flight_server: Any) -> None:
        features = self.get_features(["BaseValidateOutputFeaturesBaseNegativePandera"])

        with pytest.raises(Exception) as excinfo:
            self.basic_runner(features, modes, flight_server)

        assert "in_range(1, 2)" in str(excinfo.value)
        assert "BaseValidateOutputFeaturesBaseNegativePandera" in str(excinfo.value)

    def test_extender_validate_output_features(
        self, modes: Set[ParallelizationModes], flight_server: Any, caplog: Any
    ) -> None:
        features = self.get_features(["BaseValidateOutputFeaturesBase"])

        self.basic_runner(features, modes, flight_server, function_extender={ValidateOutputFeatureExtender()})

        if ParallelizationModes.MULTIPROCESSING not in modes:
            assert "Time taken" in caplog.text
