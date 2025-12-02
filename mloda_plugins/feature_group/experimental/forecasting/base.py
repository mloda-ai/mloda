"""
Base implementation for forecasting feature groups.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, List, Optional, Set, Type, Union

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.base_artifact import BaseArtifact
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_chainer.feature_chain_parser import (
    CHAIN_SEPARATOR,
    FeatureChainParser,
)
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys
from mloda_plugins.feature_group.experimental.forecasting.forecasting_artifact import ForecastingArtifact


class ForecastingFeatureGroup(AbstractFeatureGroup):
    """
    Base class for all forecasting feature groups.

    Forecasting feature groups generate forecasts for time series data using various algorithms.
    They allow you to predict future values based on historical patterns and trends.
    Supports both string-based feature creation and configuration-based creation with proper
    group/context parameter separation.

    ## Feature Creation Methods

    ### 1. String-Based Creation

    Features follow the naming pattern: `{in_features}__{algorithm}_forecast_{horizon}{time_unit}`

    Examples:
    ```python
    features = [
        "sales__linear_forecast_7day",      # 7-day forecast of sales using linear regression
        "energy_consumption__randomforest_forecast_24hr",  # 24-hour forecast using random forest
        "demand__svr_forecast_3month"       # 3-month forecast using support vector regression
    ]
    ```

    ### 2. Configuration-Based Creation

    Uses Options with proper group/context parameter separation:

    ```python
    feature = Feature(
        name="placeholder",
        options=Options(
            context={
                ForecastingFeatureGroup.ALGORITHM: "linear",
                ForecastingFeatureGroup.HORIZON: 7,
                ForecastingFeatureGroup.TIME_UNIT: "day",
                DefaultOptionKeys.in_features: "sales",
            }
        )
    )
    ```

    ## Parameter Classification

    ### Context Parameters (Default)
    These parameters don't affect Feature Group resolution/splitting:
    - `algorithm`: The forecasting algorithm to use
    - `horizon`: The forecast horizon (number of time units)
    - `time_unit`: The time unit for the horizon
    - `in_features`: The source feature to generate forecasts for

    ### Group Parameters
    Currently none for ForecastingFeatureGroup. Parameters that affect Feature Group
    resolution/splitting would be placed here.

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

    # Option keys for forecasting configuration
    ALGORITHM = "algorithm"
    HORIZON = "horizon"
    TIME_UNIT = "time_unit"
    OUTPUT_CONFIDENCE_INTERVALS = "output_confidence_intervals"

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
    PREFIX_PATTERN = r".*__([\w]+)_forecast_(\d+)([\w]+)$"

    # Property mapping for configuration-based features with group/context separation
    PROPERTY_MAPPING = {
        ALGORITHM: {
            **FORECASTING_ALGORITHMS,
            DefaultOptionKeys.mloda_context: True,
            DefaultOptionKeys.mloda_strict_validation: True,
        },
        HORIZON: {
            "explanation": "Forecast horizon (number of time units to predict)",
            DefaultOptionKeys.mloda_context: True,
            DefaultOptionKeys.mloda_strict_validation: True,
            DefaultOptionKeys.mloda_validation_function: lambda x: (
                isinstance(x, int) or (isinstance(x, str) and x.isdigit())
            )
            and int(x) > 0,
        },
        TIME_UNIT: {
            **TIME_UNITS,
            DefaultOptionKeys.mloda_context: True,
            DefaultOptionKeys.mloda_strict_validation: True,
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature to generate forecasts for",
            DefaultOptionKeys.mloda_context: True,
            DefaultOptionKeys.mloda_strict_validation: False,
        },
        OUTPUT_CONFIDENCE_INTERVALS: {
            "explanation": "Whether to output confidence intervals as separate columns using ~lower and ~upper suffix pattern",
            DefaultOptionKeys.mloda_context: True,
            DefaultOptionKeys.mloda_strict_validation: False,
            DefaultOptionKeys.mloda_default: False,  # Default is False (don't output confidence intervals)
            DefaultOptionKeys.mloda_validation_function: lambda value: isinstance(value, bool),
        },
    }

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
        """Extract source feature and time filter feature from either configuration-based options or string parsing."""

        source_feature: str | None = None

        # Try string-based parsing first
        _, source_feature = FeatureChainParser.parse_feature_name(feature_name, [self.PREFIX_PATTERN])
        if source_feature is not None:
            time_filter_feature = Feature(self.get_time_filter_feature(options))
            return {Feature(source_feature), time_filter_feature}

        # Fall back to configuration-based approach
        source_features = options.get_in_features()
        if len(source_features) != 1:
            raise ValueError(
                f"Expected exactly one source feature, but found {len(source_features)}: {source_features}"
            )

        source_feature_obj = next(iter(source_features))
        time_filter_feature = Feature(self.get_time_filter_feature(options))
        return {source_feature_obj, time_filter_feature}

    @classmethod
    def parse_forecast_suffix(cls, feature_name: str) -> tuple[str, int, str]:
        """
        Parse the forecast suffix into its components.

        Args:
            feature_name: The feature name to parse

        Returns:
            A tuple containing (algorithm, horizon, time_unit)

        Raises:
            ValueError: If the suffix doesn't match the expected pattern
        """
        # Extract the suffix part (everything after the double underscore)
        suffix_start = feature_name.find("__")
        if suffix_start == -1:
            raise ValueError(
                f"Invalid forecast feature name format: {feature_name}. Missing double underscore separator."
            )

        suffix = feature_name[suffix_start + 2 :]

        # Parse the suffix components
        parts = suffix.split("_")
        if len(parts) < 3 or parts[1] != "forecast":
            raise ValueError(
                f"Invalid forecast feature name format: {feature_name}. "
                f"Expected format: {{in_features}}__{{algorithm}}_forecast_{{horizon}}{{time_unit}}"
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
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Optional[Any] = None,
    ) -> bool:
        """Check if feature name matches the expected pattern for forecasting features."""

        # Use the unified parser with property mapping for full configuration support
        result = FeatureChainParser.match_configuration_feature_chain_parser(
            feature_name,
            options,
            property_mapping=cls.PROPERTY_MAPPING,
            prefix_patterns=[cls.PREFIX_PATTERN],
        )

        # If it matches and it's a string-based feature, validate with our custom logic
        if result:
            feature_name_str = feature_name.name if isinstance(feature_name, FeatureName) else feature_name

            # Check if this is a string-based feature (contains the pattern)
            if FeatureChainParser.is_chained_feature(feature_name_str):
                try:
                    # Use existing validation logic that validates algorithm, horizon, and time_unit
                    cls.parse_forecast_suffix(feature_name_str)
                except ValueError:
                    # If validation fails, this feature doesn't match
                    return False
        return result

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """
        Perform forecasting operations.

        Processes all requested features, determining the forecasting algorithm,
        horizon, time unit, and source feature from either string parsing or
        configuration-based options.

        If a trained model exists in the artifact, it is used to generate forecasts.
        Otherwise, a new model is trained and saved as an artifact.

        Adds the forecasting results directly to the input data structure.
        """

        _options = None
        for feature in features.features:
            if _options:
                if _options != feature.options:
                    raise ValueError("All features must have the same options.")
            _options = feature.options

        time_filter_feature = cls.get_time_filter_feature(_options)

        cls._check_time_filter_feature_exists(data, time_filter_feature)
        cls._check_time_filter_feature_is_datetime(data, time_filter_feature)

        # Store the original clean data
        original_data = data

        # Collect all results before modifying the data
        results = []

        # Process each requested feature with the original clean data
        for feature in features.features:
            algorithm, horizon, time_unit, in_features = cls._extract_forecasting_parameters(feature)

            # Resolve multi-column features automatically
            # If in_features is "onehot_encoded__product", this discovers
            # ["onehot_encoded__product~0", "onehot_encoded__product~1", ...]
            available_columns = cls._get_available_columns(original_data)
            resolved_columns = cls.resolve_multi_column_feature(in_features, available_columns)

            # Check that resolved columns exist
            cls._check_source_features_exist(original_data, resolved_columns)

            # Check if we have a trained model in the artifact
            model_artifact = None
            if features.artifact_to_load:
                model_artifact = cls.load_artifact(features)
                if model_artifact is None:
                    raise ValueError("No artifact to load although it was requested.")

            # Check if we should output confidence intervals
            output_confidence_intervals = (
                feature.options.get(cls.OUTPUT_CONFIDENCE_INTERVALS)
                if feature.options.get(cls.OUTPUT_CONFIDENCE_INTERVALS) is not None
                else False
            )

            # Perform forecasting using the original clean data
            if output_confidence_intervals:
                # Get forecast, lower bound, and upper bound
                result, lower_bound, upper_bound, updated_artifact = cls._perform_forecasting_with_confidence(
                    original_data, algorithm, horizon, time_unit, resolved_columns, time_filter_feature, model_artifact
                )

                # Save the updated artifact if needed
                if features.artifact_to_save and updated_artifact and not features.artifact_to_load:
                    features.save_artifact = updated_artifact

                # Store the results for later addition (main forecast + confidence bounds)
                results.append((feature.get_name(), result))
                results.append((f"{feature.get_name()}~lower", lower_bound))
                results.append((f"{feature.get_name()}~upper", upper_bound))
            else:
                # Original behavior: only output point forecast
                result, updated_artifact = cls._perform_forecasting(
                    original_data, algorithm, horizon, time_unit, resolved_columns, time_filter_feature, model_artifact
                )

                # Save the updated artifact if needed
                if features.artifact_to_save and updated_artifact and not features.artifact_to_load:
                    features.save_artifact = updated_artifact

                # Store the result for later addition
                results.append((feature.get_name(), result))

        # Add all results to the data at once
        for feature_name, result in results:
            data = cls._add_result_to_data(data, feature_name, result)

        return data

    @classmethod
    def _extract_forecasting_parameters(cls, feature: Feature) -> tuple[str, int, str, str]:
        """
        Extract forecasting parameters from a feature.

        Tries string-based parsing first, falls back to configuration-based approach.

        Args:
            feature: The feature to extract parameters from

        Returns:
            Tuple of (algorithm, horizon, time_unit, source_feature_name)

        Raises:
            ValueError: If parameters cannot be extracted
        """
        # Try string-based parsing first
        feature_name_str = feature.name.name if hasattr(feature.name, "name") else str(feature.name)

        if FeatureChainParser.is_chained_feature(feature_name_str):
            algorithm, horizon, time_unit = cls.parse_forecast_suffix(feature_name_str)

            # Extract source feature name (everything before the last double underscore)
            source_feature_name = feature_name_str.rsplit(CHAIN_SEPARATOR, 1)[0]
            return algorithm, horizon, time_unit, source_feature_name

        # Fall back to configuration-based approach
        source_features = feature.options.get_in_features()
        source_feature = next(iter(source_features))
        source_feature_name = source_feature.get_name()

        algorithm = feature.options.get(cls.ALGORITHM)
        horizon = feature.options.get(cls.HORIZON)
        time_unit = feature.options.get(cls.TIME_UNIT)

        if algorithm is None or horizon is None or time_unit is None or source_feature_name is None:
            raise ValueError(f"Could not extract forecasting parameters from: {feature.name}")

        # Validate parameters
        if algorithm not in cls.FORECASTING_ALGORITHMS:
            raise ValueError(
                f"Unsupported forecasting algorithm: {algorithm}. "
                f"Supported algorithms: {', '.join(cls.FORECASTING_ALGORITHMS.keys())}"
            )

        if time_unit not in cls.TIME_UNITS:
            raise ValueError(f"Unsupported time unit: {time_unit}. Supported units: {', '.join(cls.TIME_UNITS.keys())}")

        # Convert horizon to integer if it's a string
        if isinstance(horizon, str):
            horizon = int(horizon)

        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError(f"Invalid horizon: {horizon}. Must be a positive integer.")

        return algorithm, horizon, time_unit, source_feature_name

    @classmethod
    @abstractmethod
    def _check_time_filter_feature_exists(cls, data: Any, time_filter_feature: str) -> None:
        """
        Check if the time filter feature exists in the data.

        Args:
            data: The input data
            time_filter_feature: The name of the time filter feature

        Raises:
            ValueError: If the time filter feature does not exist in the data
        """
        ...

    @classmethod
    @abstractmethod
    def _check_time_filter_feature_is_datetime(cls, data: Any, time_filter_feature: str) -> None:
        """
        Check if the time filter feature is a datetime column.

        Args:
            data: The input data
            time_filter_feature: The name of the time filter feature

        Raises:
            ValueError: If the time filter feature is not a datetime column
        """
        ...

    @classmethod
    @abstractmethod
    def _get_available_columns(cls, data: Any) -> Set[str]:
        """
        Get the set of available column names from the data.

        Args:
            data: The input data

        Returns:
            Set of column names available in the data
        """
        ...

    @classmethod
    @abstractmethod
    def _check_source_features_exist(cls, data: Any, feature_names: List[str]) -> None:
        """
        Check if the resolved source features exist in the data.

        Args:
            data: The input data
            feature_names: List of resolved feature names (may contain ~N suffixes)

        Raises:
            ValueError: If none of the features exist in the data
        """
        ...

    @classmethod
    @abstractmethod
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
        ...

    @classmethod
    @abstractmethod
    def _perform_forecasting(
        cls,
        data: Any,
        algorithm: str,
        horizon: int,
        time_unit: str,
        in_features: List[str],
        time_filter_feature: str,
        model_artifact: Optional[Any] = None,
    ) -> tuple[Any, Optional[Any]]:
        """
        Method to perform the forecasting. Should be implemented by subclasses.

        Supports both single-column and multi-column forecasting:
        - Single column: [feature_name] - forecasts a single time series
        - Multi-column: [feature~0, feature~1, ...] - forecasts multiple time series

        Args:
            data: The input data
            algorithm: The forecasting algorithm to use
            horizon: The forecast horizon
            time_unit: The time unit for the horizon
            in_features: List of resolved source feature names to forecast
            time_filter_feature: The name of the time filter feature
            model_artifact: Optional artifact containing a trained model

        Returns:
            A tuple containing (forecast_result, updated_artifact)
        """
        ...

    @classmethod
    @abstractmethod
    def _perform_forecasting_with_confidence(
        cls,
        data: Any,
        algorithm: str,
        horizon: int,
        time_unit: str,
        in_features: List[str],
        time_filter_feature: str,
        model_artifact: Optional[Any] = None,
    ) -> tuple[Any, Any, Any, Optional[Any]]:
        """
        Method to perform forecasting and return point forecast plus confidence intervals.

        Should be implemented by subclasses to provide confidence intervals for forecasts.

        Args:
            data: The input data
            algorithm: The forecasting algorithm to use
            horizon: The forecast horizon
            time_unit: The time unit for the horizon
            in_features: List of resolved source feature names to forecast
            time_filter_feature: The name of the time filter feature
            model_artifact: Optional artifact containing a trained model

        Returns:
            A tuple containing (point_forecast, lower_bound, upper_bound, updated_artifact)
            - point_forecast: The point forecast values
            - lower_bound: The lower confidence bound
            - upper_bound: The upper confidence bound
            - updated_artifact: The updated artifact (or None)
        """
        ...
