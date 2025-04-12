"""
Base implementation for time window feature groups.
"""

from __future__ import annotations

import re
from typing import Any, Optional, Set, Union

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.options import Options
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


class BaseTimeWindowFeatureGroup(AbstractFeatureGroup):
    """
    Base class for all time window feature groups.

    Time window feature groups calculate rolling window operations over time series data.
    They allow you to compute metrics like moving averages, rolling maximums, or cumulative
    sums over specified time periods.

    ## Feature Naming Convention

    Time window features follow this naming pattern:
    `{window_function}_{window_size}_{time_unit}_window_{source_feature}`

    Examples:
    - `avg_7_day_window_temperature`: 7-day moving average of temperature
    - `max_3_hour_window_cpu_usage`: 3-hour rolling maximum of CPU usage
    - `sum_30_minute_window_transactions`: 30-minute cumulative sum of transactions

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
        if options and options.get(DefaultOptionKeys.reference_time):
            reference_time = options.get(DefaultOptionKeys.reference_time)
            if not isinstance(reference_time, str):
                raise ValueError(
                    f"Invalid reference_time option: {reference_time}. Must be string. Is: {type(reference_time)}."
                )
            return reference_time
        return DefaultOptionKeys.reference_time

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

    FEATURE_NAME_PATTERN = r"^([\w]+)_(\d+)_([\w]+)_window_([\w]+)$"

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        source_feature = self.get_source_feature(feature_name.name)
        time_filter_feature = Feature(self.get_time_filter_feature(options))
        return {Feature(source_feature), time_filter_feature}

    @classmethod
    def parse_feature_name(cls, feature_name: str) -> tuple[str, int, str, str]:
        """
        Parse the feature name into its components.

        Args:
            feature_name: The feature name to parse

        Returns:
            A tuple containing (window_function, window_size, time_unit, source_feature)

        Raises:
            ValueError: If the feature name does not match the expected pattern
        """
        match = re.match(cls.FEATURE_NAME_PATTERN, feature_name)
        if not match:
            raise ValueError(
                f"Invalid time window feature name format: {feature_name}. "
                f"Expected format: {{window_function}}_{{window_size}}_{{time_unit}}_window_{{source_feature}}"
            )

        window_function, window_size_str, time_unit, source_feature = match.groups()

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

        return window_function, window_size, time_unit, source_feature

    @classmethod
    def get_window_function(cls, feature_name: str) -> str:
        """Extract the window function from the feature name."""
        return cls.parse_feature_name(feature_name)[0]

    @classmethod
    def get_window_size(cls, feature_name: str) -> int:
        """Extract the window size from the feature name."""
        return cls.parse_feature_name(feature_name)[1]

    @classmethod
    def get_time_unit(cls, feature_name: str) -> str:
        """Extract the time unit from the feature name."""
        return cls.parse_feature_name(feature_name)[2]

    @classmethod
    def get_source_feature(cls, feature_name: str) -> str:
        """Extract the source feature name from the feature name."""
        return cls.parse_feature_name(feature_name)[3]

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
            # Try to parse the feature name - if it succeeds, it matches our pattern
            cls.parse_feature_name(feature_name)
            return True
        except ValueError:
            return False

    @classmethod
    def _perform_window_operation(
        cls,
        data: Any,
        window_function: str,
        window_size: int,
        time_unit: str,
        source_feature: str,
        time_filter_feature: Optional[str] = None,
    ) -> Any:
        """
        Method to perform the time window operation. Should be implemented by subclasses.

        Args:
            data: The input data
            window_function: The type of window function to perform
            window_size: The size of the window
            time_unit: The time unit for the window
            source_feature: The name of the source feature
            time_filter_feature: The name of the time filter feature to use for time-based operations.
                                If None, uses the value from get_time_filter_feature().

        Returns:
            The result of the window operation
        """
        raise NotImplementedError(f"_perform_window_operation not implemented in {cls.__name__}")
