import logging
import time
from typing import Any, Dict, List, Optional, Set

import pytest
from pandera import Check, Column

from mloda.provider import FeatureGroup
from mloda.user import Feature
from mloda.user import Features
from mloda.user import FeatureName
from mloda.provider import FeatureSet
from mloda.provider import BaseInputData
from mloda.provider import DataCreator
from mloda.user import Options
from mloda.user import ParallelizationMode
from mloda.steward import ExtenderHook, Extender
from tests.test_core.test_tooling import MlodaTestRunner, PARALLELIZATION_MODES_SYNC_THREADING
from tests.test_plugins.integration_plugins.test_validate_features.example_validator import ExamplePanderaValidator

logger = logging.getLogger(__name__)


class BaseValidateInputFeaturesBase(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): [1, 2, 3]}


class SimpleValidateInputFeatures(FeatureGroup):
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


class CustomValidateInputFeatures(FeatureGroup):
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


class ValidateInputFeatureExtender(Extender):
    def wraps(self) -> Set[ExtenderHook]:
        return {ExtenderHook.VALIDATE_INPUT_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        logger.error(f"Time taken: {time.time() - start}")
        return result


class TrackingExtender(Extender):
    """Extender that tracks execution order for testing multiple extenders."""

    execution_log: List[str] = []

    def __init__(self, name: str, priority: int = 100):
        self.name = name
        self.priority = priority

    def wraps(self) -> Set[ExtenderHook]:
        return {ExtenderHook.VALIDATE_INPUT_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        TrackingExtender.execution_log.append(self.name)
        return func(*args, **kwargs)


@PARALLELIZATION_MODES_SYNC_THREADING
class TestValidateInputFeatures:
    def get_features(self, feature_list: List[str], options: Dict[str, Any] = {}) -> Features:
        return Features([Feature(name=f_name, options=options, initial_requested_data=True) for f_name in feature_list])

    def test_basic_validate_input_features(self, modes: Set[ParallelizationMode], flight_server: Any) -> None:
        _features = "SimpleValidateInputFeatures"

        features = self.get_features([_features])

        with pytest.raises(Exception):
            MlodaTestRunner.run_api_simple(features, parallelization_modes=modes, flight_server=flight_server)

    def test_custom_validate_input_features_error(self, modes: Set[ParallelizationMode], flight_server: Any) -> None:
        _features = "CustomValidateInputFeatures"

        features = self.get_features([_features])

        with pytest.raises(Exception) as excinfo:
            MlodaTestRunner.run_api_simple(features, parallelization_modes=modes, flight_server=flight_server)

        assert "in_range(1, 2)" in str(excinfo.value)
        assert "BaseValidateInputFeaturesBase" in str(excinfo.value)
        assert "failure cases: 3" in str(excinfo.value)

    def test_custom_validate_input_features_warning(
        self, modes: Set[ParallelizationMode], flight_server: Any, caplog: Any
    ) -> None:
        _features = "CustomValidateInputFeatures"

        features = self.get_features([_features], {"ValidationLevel": "warning"})

        # Current mp version, logging is broken. This is known and not a focus for now.
        if ParallelizationMode.MULTIPROCESSING not in modes:
            MlodaTestRunner.run_api_simple(features, parallelization_modes=modes, flight_server=flight_server)

            assert (
                "pandera.errors.SchemaError: Column 'BaseValidateInputFeaturesBase' failed element-wise validator number 0: in_range(1, 2) failure cases: 3"
                in caplog.text
            )

    def test_custom_validate_input_features_given_example_valiator(
        self, modes: Set[ParallelizationMode], flight_server: Any
    ) -> None:
        _features = "CustomValidateInputFeatures"

        validation_rules = {
            "BaseValidateInputFeaturesBase": Column(int, Check.in_range(1, 2)),
        }
        validator = ExamplePanderaValidator(validation_rules, "error")
        features = self.get_features([_features], {validator.__class__.__name__: validator})

        with pytest.raises(Exception) as excinfo:
            MlodaTestRunner.run_api_simple(features, parallelization_modes=modes, flight_server=flight_server)

        assert "in_range(1, 2)" in str(excinfo.value)
        assert "BaseValidateInputFeaturesBase" in str(excinfo.value)
        assert "failure cases: 3" in str(excinfo.value)

    def test_custom_validate_input_features_extender(
        self, modes: Set[ParallelizationMode], flight_server: Any, caplog: Any
    ) -> None:
        _features = "CustomValidateInputFeatures"

        features = self.get_features([_features], {"ValidationLevel": "warning"})

        MlodaTestRunner.run_api_simple(
            features,
            parallelization_modes=modes,
            flight_server=flight_server,
            function_extender={ValidateInputFeatureExtender()},
        )

        # Current mp version, logging is broken. This is known and not a focus for now.
        if ParallelizationMode.MULTIPROCESSING not in modes:
            assert "Time taken" in caplog.text

    def test_multiple_extenders_execute_in_priority_order(
        self, modes: Set[ParallelizationMode], flight_server: Any
    ) -> None:
        """Integration test: multiple extenders are chained and execute in priority order."""
        _features = "CustomValidateInputFeatures"
        features = self.get_features([_features], {"ValidationLevel": "warning"})

        # Clear execution log and create extenders with different priorities
        TrackingExtender.execution_log = []
        extender_low = TrackingExtender("low_priority", priority=10)
        extender_high = TrackingExtender("high_priority", priority=50)

        # Pass extenders in non-priority order to verify sorting works
        MlodaTestRunner.run_api_simple(
            features,
            parallelization_modes=modes,
            flight_server=flight_server,
            function_extender={extender_high, extender_low},
        )

        # Verify both extenders were called in priority order (lower first)
        assert "low_priority" in TrackingExtender.execution_log
        assert "high_priority" in TrackingExtender.execution_log
        assert TrackingExtender.execution_log.index("low_priority") < TrackingExtender.execution_log.index(
            "high_priority"
        )
