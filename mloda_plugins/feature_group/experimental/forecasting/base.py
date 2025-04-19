"""
Base implementation for forecasting feature groups.
"""

from __future__ import annotations

from typing import Any, Optional, Set, Type, Union

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.base_artifact import BaseArtifact
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda_core.abstract_plugins.components.feature_chainer.feature_chainer_parser_configuration import (
    FeatureChainParserConfiguration,
    create_configurable_parser,
)
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys
from mloda_plugins.feature_group.experimental.forecasting.forecasting_artifact import ForecastingArtifact


class ForecastingFeatureGroup(AbstractFeatureGroup):
    # Option keys for forecasting configuration
    ALGORITHM = "algorithm"
    HORIZON = "horizon"
    TIME_UNIT = "time_unit"
    """
    Base class for all forecasting feature groups.

    Forecasting feature groups generate forecasts for time series data using various algorithms.
    They allow you to predict future values based on historical patterns and trends.

    ## Feature Naming Convention

    Forecasting features follow this naming pattern:
    `{algorithm}_forecast_{horizon}{time_unit}__{mloda_source_feature}`

    The source feature (mloda_source_feature) is extracted from the feature name and used
    as input for the forecasting algorithm. Note the double underscore before the source feature.

    Examples:
    - `linear_forecast_7day__sales`: 7-day forecast of sales using linear regression
    - `randomforest_forecast_24hr__energy_consumption`: 24-hour forecast of energy consumption using random forest
    - `svr_forecast_3month__demand`: 3-month forecast of demand using support vector regression

    ## Configuration-Based Creation

    ForecastingFeatureGroup supports configuration-based creation through the
    FeatureChainParserConfiguration mechanism. This allows features to be created
    from options rather than explicit feature names.

    To create a forecasting feature using configuration:

    ```python
    feature = Feature(
        "PlaceHolder",  # Placeholder name, will be replaced
        Options({
            ForecastingFeatureGroup.ALGORITHM: "linear",
            ForecastingFeatureGroup.HORIZON: 7,
            ForecastingFeatureGroup.TIME_UNIT: "day",
            DefaultOptionKeys.mloda_source_feature: "sales"
        })
    )

    # The Engine will automatically parse this into a feature with name "linear_forecast_7day__sales"
    ```

    ## Supported Forecasting Algorithms

    - `linear`: Linear regression
    - `ridge`: Ridge regression
    - `lasso`: Lasso regression
    - `randomforest`: Random Forest regression
    - `gbr`: Gradient Boosting regression
    - `svr`: Support Vector regression
    - `knn`: K-Nearest Neighbors regression

    ## Supported Time Units

    - `second`: Seconds
    - `minute`: Minutes
    - `hour`: Hours
    - `day`: Days
    - `week`: Weeks
    - `month`: Months
    - `year`: Years

    ## Requirements
    - The input data must have a datetime column that can be used for time-based operations
    - By default, the feature group will use DefaultOptionKeys.reference_time (default: "time_filter")
    - You can specify a custom time column by setting the reference_time option in the feature group options
    """

    # Define supported forecasting algorithms
    FORECASTING_ALGORITHMS = {
        "linear": "Linear Regression",
        "ridge": "Ridge Regression",
        "lasso": "Lasso Regression",
        "randomforest": "Random Forest Regression",
        "gbr": "Gradient Boosting Regression",
        "svr": "Support Vector Regression",
        "knn": "K-Nearest Neighbors Regression",
    }

    # Define supported time units (same as TimeWindowFeatureGroup)
    TIME_UNITS = {
        "second": "Seconds",
        "minute": "Minutes",
        "hour": "Hours",
        "day": "Days",
        "week": "Weeks",
        "month": "Months",
        "year": "Years",
    }

    # Define the prefix pattern for this feature group
    PREFIX_PATTERN = r"^([\w]+)_forecast_(\d+)([\w]+)__"

    @staticmethod
    def artifact() -> Type[BaseArtifact] | None:
        """
        Returns the artifact class for this feature group.

        The ForecastingFeatureGroup uses the ForecastingArtifact to store
        trained models and other components needed for forecasting.
        """
        return ForecastingArtifact

    @classmethod
    def get_time_filter_feature(cls, options: Optional[Options] = None) -> str:
        """
        Get the time filter feature name from options or use the default.

        Args:
            options: Optional Options object that may contain a custom time filter feature name

        Returns:
            The time filter feature name to use
        """
        reference_time_key = DefaultOptionKeys.reference_time.value
        if options and options.get(reference_time_key):
            reference_time = options.get(reference_time_key)
            if not isinstance(reference_time, str):
                raise ValueError(
                    f"Invalid reference_time option: {reference_time}. Must be string. Is: {type(reference_time)}."
                )
            return reference_time
        return reference_time_key

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """
        Extract source feature and time filter feature from the feature name.

        Args:
            options: The options for the feature
            feature_name: The name of the feature

        Returns:
            A set containing the source feature and time filter feature
        """
        source_feature = FeatureChainParser.extract_source_feature(feature_name.name, self.PREFIX_PATTERN)
        time_filter_feature = Feature(self.get_time_filter_feature(options))
        return {Feature(source_feature), time_filter_feature}

    @classmethod
    def parse_forecast_prefix(cls, feature_name: str) -> tuple[str, int, str]:
        """
        Parse the forecast prefix into its components.

        Args:
            feature_name: The feature name to parse

        Returns:
            A tuple containing (algorithm, horizon, time_unit)

        Raises:
            ValueError: If the prefix doesn't match the expected pattern
        """
        # Extract the prefix part (everything before the double underscore)
        prefix_end = feature_name.find("__")
        if prefix_end == -1:
            raise ValueError(
                f"Invalid forecast feature name format: {feature_name}. Missing double underscore separator."
            )

        prefix = feature_name[:prefix_end]

        # Parse the prefix components
        parts = prefix.split("_")
        if len(parts) < 3 or parts[1] != "forecast":
            raise ValueError(
                f"Invalid forecast feature name format: {feature_name}. "
                f"Expected format: {{algorithm}}_forecast_{{horizon}}{{time_unit}}__{{mloda_source_feature}}"
            )

        algorithm = parts[0]
        horizon_time = parts[2]

        # Find where the digits end and the time unit begins
        for i, char in enumerate(horizon_time):
            if not char.isdigit():
                break
        else:
            raise ValueError(f"Invalid horizon format: {horizon_time}. Must include time unit.")

        horizon_str = horizon_time[:i]
        time_unit = horizon_time[i:]

        # Validate algorithm
        if algorithm not in cls.FORECASTING_ALGORITHMS:
            raise ValueError(
                f"Unsupported forecasting algorithm: {algorithm}. "
                f"Supported algorithms: {', '.join(cls.FORECASTING_ALGORITHMS.keys())}"
            )

        # Validate time unit
        if time_unit not in cls.TIME_UNITS:
            raise ValueError(f"Unsupported time unit: {time_unit}. Supported units: {', '.join(cls.TIME_UNITS.keys())}")

        # Convert horizon to integer
        try:
            horizon = int(horizon_str)
            if horizon <= 0:
                raise ValueError("Horizon must be positive")
        except ValueError:
            raise ValueError(f"Invalid horizon: {horizon_str}. Must be a positive integer.")

        return algorithm, horizon, time_unit

    @classmethod
    def get_algorithm(cls, feature_name: str) -> str:
        """Extract the forecasting algorithm from the feature name."""
        return cls.parse_forecast_prefix(feature_name)[0]

    @classmethod
    def get_horizon(cls, feature_name: str) -> int:
        """Extract the forecast horizon from the feature name."""
        return cls.parse_forecast_prefix(feature_name)[1]

    @classmethod
    def get_time_unit(cls, feature_name: str) -> str:
        """Extract the time unit from the feature name."""
        return cls.parse_forecast_prefix(feature_name)[2]

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Optional[Any] = None,
    ) -> bool:
        """Check if feature name matches the expected pattern for forecasting features."""
        if isinstance(feature_name, FeatureName):
            feature_name = feature_name.name

        try:
            # First validate that this is a valid feature name for this feature group
            if not FeatureChainParser.validate_feature_name(feature_name, cls.PREFIX_PATTERN):
                return False

            # Then validate the forecasting components
            cls.parse_forecast_prefix(feature_name)
            return True
        except ValueError:
            return False

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """
        Perform forecasting operations.

        Processes all requested features, determining the forecasting algorithm,
        horizon, time unit, and source feature from each feature name.

        If a trained model exists in the artifact, it is used to generate forecasts.
        Otherwise, a new model is trained and saved as an artifact.

        Adds the forecasting results directly to the input data structure.
        """
        time_filter_feature = cls.get_time_filter_feature(features.options)

        cls._check_time_filter_feature_exists(data, time_filter_feature)
        cls._check_time_filter_feature_is_datetime(data, time_filter_feature)

        # Process each requested feature
        for feature_name in features.get_all_names():
            algorithm, horizon, time_unit = cls.parse_forecast_prefix(feature_name)
            mloda_source_feature = FeatureChainParser.extract_source_feature(feature_name, cls.PREFIX_PATTERN)

            cls._check_source_feature_exists(data, mloda_source_feature)

            # Check if we have a trained model in the artifact
            model_artifact = None
            if features.artifact_to_load:
                model_artifact = cls.load_artifact(features)
                if model_artifact is None:
                    raise ValueError("No artifact to load although it was requested.")

            # Perform forecasting
            result, updated_artifact = cls._perform_forecasting(
                data, algorithm, horizon, time_unit, mloda_source_feature, time_filter_feature, model_artifact
            )

            # Save the updated artifact if needed
            if features.artifact_to_save and updated_artifact and not features.artifact_to_load:
                features.save_artifact = updated_artifact

            # Add the result to the data
            data = cls._add_result_to_data(data, feature_name, result)

        return data

    @classmethod
    def _check_time_filter_feature_exists(cls, data: Any, time_filter_feature: str) -> None:
        """
        Check if the time filter feature exists in the data.

        Args:
            data: The input data
            time_filter_feature: The name of the time filter feature

        Raises:
            ValueError: If the time filter feature does not exist in the data
        """
        raise NotImplementedError(f"_check_time_filter_feature_exists not implemented in {cls.__name__}")

    @classmethod
    def _check_time_filter_feature_is_datetime(cls, data: Any, time_filter_feature: str) -> None:
        """
        Check if the time filter feature is a datetime column.

        Args:
            data: The input data
            time_filter_feature: The name of the time filter feature

        Raises:
            ValueError: If the time filter feature is not a datetime column
        """
        raise NotImplementedError(f"_check_time_filter_feature_is_datetime not implemented in {cls.__name__}")

    @classmethod
    def _check_source_feature_exists(cls, data: Any, mloda_source_feature: str) -> None:
        """
        Check if the source feature exists in the data.

        Args:
            data: The input data
            mloda_source_feature: The name of the source feature

        Raises:
            ValueError: If the source feature does not exist in the data
        """
        raise NotImplementedError(f"_check_source_feature_exists not implemented in {cls.__name__}")

    @classmethod
    def _add_result_to_data(cls, data: Any, feature_name: str, result: Any) -> Any:
        """
        Add the result to the data.

        Args:
            data: The input data
            feature_name: The name of the feature to add
            result: The result to add

        Returns:
            The updated data
        """
        raise NotImplementedError(f"_add_result_to_data not implemented in {cls.__name__}")

    @classmethod
    def _perform_forecasting(
        cls,
        data: Any,
        algorithm: str,
        horizon: int,
        time_unit: str,
        mloda_source_feature: str,
        time_filter_feature: str,
        model_artifact: Optional[Any] = None,
    ) -> tuple[Any, Optional[Any]]:
        """
        Method to perform the forecasting. Should be implemented by subclasses.

        Args:
            data: The input data
            algorithm: The forecasting algorithm to use
            horizon: The forecast horizon
            time_unit: The time unit for the horizon
            mloda_source_feature: The name of the source feature
            time_filter_feature: The name of the time filter feature
            model_artifact: Optional artifact containing a trained model

        Returns:
            A tuple containing (forecast_result, updated_artifact)
        """
        raise NotImplementedError(f"_perform_forecasting not implemented in {cls.__name__}")

    @classmethod
    def configurable_feature_chain_parser(cls) -> Optional[Type[FeatureChainParserConfiguration]]:
        """
        Returns the FeatureChainParserConfiguration class for this feature group.

        This method allows the Engine to automatically create features with the correct
        naming convention based on configuration options, rather than requiring explicit
        feature names.

        Returns:
            A configured FeatureChainParserConfiguration class
        """
        return create_configurable_parser(
            parse_keys=[
                cls.ALGORITHM,
                cls.HORIZON,
                cls.TIME_UNIT,
                DefaultOptionKeys.mloda_source_feature,
            ],
            feature_name_template="{algorithm}_forecast_{horizon}{time_unit}__{mloda_source_feature}",
            validation_rules={
                cls.ALGORITHM: lambda x: x in cls.FORECASTING_ALGORITHMS,
                cls.TIME_UNIT: lambda x: x in cls.TIME_UNITS,
                cls.HORIZON: lambda x: (isinstance(x, int) or (isinstance(x, str) and x.isdigit())) and int(x) > 0,
            },
        )
