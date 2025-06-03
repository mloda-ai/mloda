import time
from typing import Any, Dict, List, Optional, Set
import pytest

from pandera import Column, Check

from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda_core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.function_extender import WrapperFunctionEnum, WrapperFunctionExtender
from mloda_core.api.request import mlodaAPI
from mloda_core.runtime.flight.flight_server import FlightServer
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyarrowTable
from mloda_core.abstract_plugins.components.parallelization_modes import ParallelizationModes
from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_collection import Features
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from tests.test_plugins.integration_plugins.test_validate_features.example_validator import ExamplePanderaValidator
from tests.test_documentation.test_documentation import DokuExtender

import logging

logger = logging.getLogger(__name__)


class BaseValidateInputFeaturesBase(AbstractFeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): [1, 2, 3]}


class SimpleValidateInputFeatures(AbstractFeatureGroup):
    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): [1, 2, 3]}

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {Feature(name="BaseValidateInputFeaturesBase", options=options)}

    @classmethod
    def validate_input_features(cls, data: Any, features: FeatureSet) -> Optional[bool]:
        """This function should be used to validate the input data."""

        if len(data["BaseValidateInputFeaturesBase"]) == 3:
            raise ValueError("Data should have 3 elements")
        return True


class CustomValidateInputFeatures(AbstractFeatureGroup):
    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): [1, 2, 3]}  # dummy return

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {Feature(name="BaseValidateInputFeaturesBase", options=options)}

    @classmethod
    def validate_input_features(cls, data: Any, features: FeatureSet) -> Optional[bool]:
        """This function should be used to validate the input data."""

        validation_rules = {
            "BaseValidateInputFeaturesBase": Column(int, Check.in_range(1, 2)),
        }

        if features.get_options_key("ExamplePanderaValidator") is not None:
            validator = features.get_options_key("ExamplePanderaValidator")
            if not isinstance(validator, ExamplePanderaValidator):
                raise ValueError("ExamplePanderaValidator should be an instance of ExamplePanderaValidator")
        else:
            validation_log_level = features.get_options_key("ValidationLevel")
            validator = ExamplePanderaValidator(validation_rules, validation_log_level)

        return validator.validate(data)


class ValidateInputFeatureExtender(DokuExtender):
    def wraps(self) -> Set[WrapperFunctionEnum]:
        return {WrapperFunctionEnum.VALIDATE_INPUT_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        logger.error(f"Time taken: {time.time() - start}")
        return result


@pytest.mark.parametrize(
    "modes",
    [
        ({ParallelizationModes.SYNC}),
        ({ParallelizationModes.THREADING}),
        # ({ParallelizationModes.MULTIPROCESSING}),
    ],
)
class TestValidateInputFeatures:
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
            features, {PyarrowTable}, None, None, parallelization_modes, flight_server, function_extender
        )

        # make sure all datasets are dropped on server
        if ParallelizationModes.MULTIPROCESSING in parallelization_modes:
            flight_infos = FlightServer.list_flight_infos(flight_server.location)
            assert len(flight_infos) == 0

        return results

    def test_basic_validate_input_features(self, modes: Set[ParallelizationModes], flight_server: Any) -> None:
        _features = "SimpleValidateInputFeatures"

        features = self.get_features([_features])

        with pytest.raises(Exception):
            self.basic_runner(features, modes, flight_server)

    def test_custom_validate_input_features_error(self, modes: Set[ParallelizationModes], flight_server: Any) -> None:
        _features = "CustomValidateInputFeatures"

        features = self.get_features([_features])

        with pytest.raises(Exception) as excinfo:
            self.basic_runner(features, modes, flight_server)

        assert "in_range(1, 2)" in str(excinfo.value)
        assert "BaseValidateInputFeaturesBase" in str(excinfo.value)
        assert "failure cases: 3" in str(excinfo.value)

    def test_custom_validate_input_features_warning(
        self, modes: Set[ParallelizationModes], flight_server: Any, caplog: Any
    ) -> None:
        _features = "CustomValidateInputFeatures"

        features = self.get_features([_features], {"ValidationLevel": "warning"})

        # Current mp version, logging is broken. This is known and not a focus for now.
        if ParallelizationModes.MULTIPROCESSING not in modes:
            self.basic_runner(features, modes, flight_server)

            assert (
                "pandera.errors.SchemaError: Column 'BaseValidateInputFeaturesBase' failed element-wise validator number 0: in_range(1, 2) failure cases: 3"
                in caplog.text
            )

    def test_custom_validate_input_features_given_example_valiator(
        self, modes: Set[ParallelizationModes], flight_server: Any
    ) -> None:
        _features = "CustomValidateInputFeatures"

        validation_rules = {
            "BaseValidateInputFeaturesBase": Column(int, Check.in_range(1, 2)),
        }
        validator = ExamplePanderaValidator(validation_rules, "error")
        features = self.get_features([_features], {validator.__class__.__name__: validator})

        with pytest.raises(Exception) as excinfo:
            self.basic_runner(features, modes, flight_server)

        assert "in_range(1, 2)" in str(excinfo.value)
        assert "BaseValidateInputFeaturesBase" in str(excinfo.value)
        assert "failure cases: 3" in str(excinfo.value)

    def test_custom_validate_input_features_extender(
        self, modes: Set[ParallelizationModes], flight_server: Any, caplog: Any
    ) -> None:
        _features = "CustomValidateInputFeatures"

        features = self.get_features([_features], {"ValidationLevel": "warning"})

        self.basic_runner(features, modes, flight_server, function_extender={ValidateInputFeatureExtender()})

        # Current mp version, logging is broken. This is known and not a focus for now.
        if ParallelizationModes.MULTIPROCESSING not in modes:
            assert "Time taken" in caplog.text
