"""
Base implementation for time window feature groups.
"""

from __future__ import annotations

from typing import Any, Optional, Set, Type, Union

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
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


class TimeWindowFeatureGroup(AbstractFeatureGroup):
    # Option keys for time window configuration
    WINDOW_FUNCTION = "window_function"
    WINDOW_SIZE = "window_size"
    TIME_UNIT = "time_unit"
    """
    Base class for all time window feature groups.

    Time window feature groups calculate rolling window operations over time series data.
    They allow you to compute metrics like moving averages, rolling maximums, or cumulative
    sums over specified time periods.

    ## Feature Naming Convention

    Time window features follow this naming pattern:
    `{window_function}_{window_size}_{time_unit}_window__{mloda_source_feature}`

    The source feature (mloda_source_feature) is extracted from the feature name and used
    as input for the time window operation. Note the double underscore before the source feature.

    Examples:
    - `avg_7_day_window__temperature`: 7-day moving average of temperature
    - `max_3_hour_window__cpu_usage`: 3-hour rolling maximum of CPU usage
    - `sum_30_minute_window__transactions`: 30-minute cumulative sum of transactions

    ## Supported Window Functions

    - `sum`: Sum of values in the window
    - `min`: Minimum value in the window
    - `max`: Maximum value in the window
    - `avg`/`mean`: Average (mean) of values in the window
    - `count`: Count of non-null values in the window
    - `std`: Standard deviation of values in the window
    - `var`: Variance of values in the window
    - `median`: Median value in the window
    - `first`: First value in the window
    - `last`: Last value in the window

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

    # Define supported window functions
    WINDOW_FUNCTIONS = {
        "sum": "Sum of values in window",
        "min": "Minimum value in window",
        "max": "Maximum value in window",
        "avg": "Average (mean) of values in window",
        "mean": "Average (mean) of values in window",
        "count": "Count of non-null values in window",
        "std": "Standard deviation of values in window",
        "var": "Variance of values in window",
        "median": "Median value in window",
        "first": "First value in window",
        "last": "Last value in window",
    }

    # Define supported time units
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
    PREFIX_PATTERN = r"^([\w]+)_(\d+)_([\w]+)_window__"

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        source_feature = FeatureChainParser.extract_source_feature(feature_name.name, self.PREFIX_PATTERN)
        time_filter_feature = Feature(self.get_time_filter_feature(options))
        return {Feature(source_feature), time_filter_feature}

    @classmethod
    def parse_time_window_prefix(cls, feature_name: str) -> tuple[str, int, str]:
        """
        Parse the time window prefix into its components.

        Args:
            feature_name: The feature name to parse

        Returns:
            A tuple containing (window_function, window_size, time_unit)

        Raises:
            ValueError: If the prefix doesn't match the expected pattern
        """
        # Extract the prefix part (everything before the double underscore)
        prefix_end = feature_name.find("__")
        if prefix_end == -1:
            raise ValueError(
                f"Invalid time window feature name format: {feature_name}. Missing double underscore separator."
            )

        prefix = feature_name[:prefix_end]

        # Parse the prefix components
        parts = prefix.split("_")
        if len(parts) != 4 or parts[3] != "window":
            raise ValueError(
                f"Invalid time window feature name format: {feature_name}. "
                f"Expected format: {{window_function}}_{{window_size}}_{{time_unit}}_window__{{mloda_source_feature}}"
            )

        window_function, window_size_str, time_unit = parts[0], parts[1], parts[2]

        # Validate window function
        if window_function not in cls.WINDOW_FUNCTIONS:
            raise ValueError(
                f"Unsupported window function: {window_function}. "
                f"Supported functions: {', '.join(cls.WINDOW_FUNCTIONS.keys())}"
            )

        # Validate time unit
        if time_unit not in cls.TIME_UNITS:
            raise ValueError(f"Unsupported time unit: {time_unit}. Supported units: {', '.join(cls.TIME_UNITS.keys())}")

        # Convert window size to integer
        try:
            window_size = int(window_size_str)
            if window_size <= 0:
                raise ValueError("Window size must be positive")
        except ValueError:
            raise ValueError(f"Invalid window size: {window_size_str}. Must be a positive integer.")

        return window_function, window_size, time_unit

    @classmethod
    def get_window_function(cls, feature_name: str) -> str:
        """Extract the window function from the feature name."""
        return cls.parse_time_window_prefix(feature_name)[0]

    @classmethod
    def get_window_size(cls, feature_name: str) -> int:
        """Extract the window size from the feature name."""
        return cls.parse_time_window_prefix(feature_name)[1]

    @classmethod
    def get_time_unit(cls, feature_name: str) -> str:
        """Extract the time unit from the feature name."""
        return cls.parse_time_window_prefix(feature_name)[2]

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Optional[Any] = None,
    ) -> bool:
        """Check if feature name matches the expected pattern for time window features."""
        if isinstance(feature_name, FeatureName):
            feature_name = feature_name.name

        try:
            # First validate that this is a valid feature name for this feature group
            if not FeatureChainParser.validate_feature_name(feature_name, cls.PREFIX_PATTERN):
                return False

            # Then validate the time window components
            cls.parse_time_window_prefix(feature_name)
            return True
        except ValueError:
            return False

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """
        Perform time window operations.

        Processes all requested features, determining the window function, window size,
        time unit, and source feature from each feature name.

        Adds the time window results directly to the input data structure.
        """
        time_filter_feature = cls.get_time_filter_feature(features.options)

        cls._check_time_filter_feature_exists(data, time_filter_feature)

        cls._check_time_filter_feature_is_datetime(data, time_filter_feature)

        # Process each requested feature
        for feature_name in features.get_all_names():
            window_function, window_size, time_unit = cls.parse_time_window_prefix(feature_name)
            mloda_source_feature = FeatureChainParser.extract_source_feature(feature_name, cls.PREFIX_PATTERN)

            cls._check_source_feature_exists(data, mloda_source_feature)

            result = cls._perform_window_operation(
                data, window_function, window_size, time_unit, mloda_source_feature, time_filter_feature
            )

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
    def _perform_window_operation(
        cls,
        data: Any,
        window_function: str,
        window_size: int,
        time_unit: str,
        mloda_source_feature: str,
        time_filter_feature: Optional[str] = None,
    ) -> Any:
        """
        Method to perform the time window operation. Should be implemented by subclasses.

        Args:
            data: The input data
            window_function: The type of window function to perform
            window_size: The size of the window
            time_unit: The time unit for the window
            mloda_source_feature: The name of the source feature
            time_filter_feature: The name of the time filter feature to use for time-based operations.
                                If None, uses the value from get_time_filter_feature().

        Returns:
            The result of the window operation
        """
        raise NotImplementedError(f"_perform_window_operation not implemented in {cls.__name__}")

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
                cls.WINDOW_FUNCTION,
                cls.WINDOW_SIZE,
                cls.TIME_UNIT,
                DefaultOptionKeys.mloda_source_feature,
            ],
            feature_name_template="{window_function}_{window_size}_{time_unit}_window__{mloda_source_feature}",
            validation_rules={
                cls.WINDOW_FUNCTION: lambda x: x in cls.WINDOW_FUNCTIONS,
                cls.TIME_UNIT: lambda x: x in cls.TIME_UNITS,
                cls.WINDOW_SIZE: lambda x: (isinstance(x, int) or (isinstance(x, str) and x.isdigit())) and int(x) > 0,
            },
        )
