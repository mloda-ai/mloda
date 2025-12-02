"""
Base implementation for time window feature groups.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, List, Optional, Set, Type, Union

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
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
    `{in_features}__{window_function}_{window_size}_{time_unit}_window`

    The source feature (in_features) comes first, followed by the window operation.
    Note the double underscore separating the source feature from the operation.

    Examples:
    - `temperature__avg_7_day_window`: 7-day moving average of temperature
    - `cpu_usage__max_3_hour_window`: 3-hour rolling maximum of CPU usage
    - `transactions__sum_30_minute_window`: 30-minute cumulative sum of transactions

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

    # Define PROPERTY_MAPPING for the new unified parser approach
    PROPERTY_MAPPING = {
        # Window function parameter (context parameter)
        WINDOW_FUNCTION: {
            **WINDOW_FUNCTIONS,  # Reference existing WINDOW_FUNCTIONS dict
            DefaultOptionKeys.mloda_context: True,  # Mark as context parameter
            DefaultOptionKeys.mloda_strict_validation: True,  # Enable strict validation
        },
        # Window size parameter (context parameter)
        WINDOW_SIZE: {
            "explanation": "Size of the time window (must be positive integer)",
            DefaultOptionKeys.mloda_context: True,  # Mark as context parameter
            DefaultOptionKeys.mloda_strict_validation: True,  # Enable strict validation
            DefaultOptionKeys.mloda_validation_function: lambda x: (isinstance(x, int) and x > 0)
            or (isinstance(x, str) and x.isdigit() and int(x) > 0),
        },
        # Time unit parameter (context parameter)
        TIME_UNIT: {
            **TIME_UNITS,  # Reference existing TIME_UNITS dict
            DefaultOptionKeys.mloda_context: True,  # Mark as context parameter
            DefaultOptionKeys.mloda_strict_validation: True,  # Enable strict validation
        },
        # Source feature parameter (context parameter)
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature to apply time window operation to",
            DefaultOptionKeys.mloda_context: True,  # Mark as context parameter
            DefaultOptionKeys.mloda_strict_validation: False,  # Flexible validation
        },
    }

    # Define the pattern separator and regex for this feature group
    PATTERN = "__"
    PREFIX_PATTERN = r".*__([\w]+)_(\d+)_([\w]+)_window$"

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """Extract source feature from either configuration-based options or string parsing."""

        source_feature: str | None = None

        # Try string-based parsing first
        _, source_feature = FeatureChainParser.parse_feature_name(feature_name.name, [self.PREFIX_PATTERN])
        if source_feature is not None:
            time_filter_feature = Feature(self.get_time_filter_feature(options))
            return {Feature(source_feature), time_filter_feature}

        # Fall back to configuration-based approach
        source_features = options.get_in_features()
        if len(source_features) != 1:
            raise ValueError(
                f"Expected exactly one source feature, but found {len(source_features)}: {source_features}"
            )

        time_filter_feature = Feature(self.get_time_filter_feature(options))
        return set(source_features) | {time_filter_feature}

    @classmethod
    def parse_time_window_prefix(cls, feature_name: str) -> tuple[str, int, str]:
        """
        Parse the time window suffix into its components.

        Args:
            feature_name: The feature name to parse

        Returns:
            A tuple containing (window_function, window_size, time_unit)

        Raises:
            ValueError: If the suffix doesn't match the expected pattern
        """
        # Extract the suffix part (everything after the last double underscore before the window pattern)
        # Use rfind to support chained features in L->R format (e.g., price__mean_imputed__sum_7_day_window)
        suffix_start = feature_name.rfind("__")
        if suffix_start == -1:
            raise ValueError(
                f"Invalid time window feature name format: {feature_name}. Missing double underscore separator."
            )

        suffix = feature_name[suffix_start + 2 :]

        # Parse the suffix components
        parts = suffix.split("_")
        if len(parts) != 4 or parts[3] != "window":
            raise ValueError(
                f"Invalid time window feature name format: {feature_name}. "
                f"Expected format: {{in_features}}__{{window_function}}_{{window_size}}_{{time_unit}}_window"
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

        # Use unified parser approach with PROPERTY_MAPPING
        return FeatureChainParser.match_configuration_feature_chain_parser(
            feature_name,
            options,
            property_mapping=cls.PROPERTY_MAPPING,
            prefix_patterns=[cls.PREFIX_PATTERN],
        )

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """
        Perform time window operations.

        Processes all requested features, determining the window function, window size,
        time unit, and source feature from each feature name.

        Supports multi-column features by using resolve_multi_column_feature() to
        automatically discover columns matching the pattern feature_name~N.

        Adds the time window results directly to the input data structure.
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

        # Process each requested feature
        for feature in features.features:
            feature_name = feature.get_name()

            # Try string-based parsing first (for legacy features)
            parsed_params, in_features = FeatureChainParser.parse_feature_name(feature_name, [cls.PREFIX_PATTERN])

            if in_features is not None:
                # String-based approach succeeded
                window_function, window_size, time_unit = cls.parse_time_window_prefix(feature_name)
            else:
                # Fall back to configuration-based approach
                has_config_params = (
                    feature.options.get(cls.WINDOW_FUNCTION) is not None
                    and feature.options.get(cls.WINDOW_SIZE) is not None
                    and feature.options.get(cls.TIME_UNIT) is not None
                )

                if not has_config_params:
                    raise ValueError(
                        f"Feature '{feature_name}' does not match string pattern and lacks configuration parameters"
                    )

                # Configuration-based approach
                source_features = feature.options.get_in_features()
                if len(source_features) != 1:
                    raise ValueError(
                        f"Expected exactly one source feature, but found {len(source_features)}: {source_features}"
                    )
                source_feature = next(iter(source_features))
                in_features = source_feature.get_name()

                # Extract parameters from options
                window_function = feature.options.get(cls.WINDOW_FUNCTION)
                window_size = feature.options.get(cls.WINDOW_SIZE)
                time_unit = feature.options.get(cls.TIME_UNIT)

                # Convert window_size to int if it's a string
                if isinstance(window_size, str):
                    window_size = int(window_size)

            # Resolve multi-column features automatically
            # If in_features is "onehot_encoded__product", this discovers
            # ["onehot_encoded__product~0", "onehot_encoded__product~1", ...]
            available_columns = cls._get_available_columns(data)
            resolved_columns = cls.resolve_multi_column_feature(in_features, available_columns)

            # Check that resolved columns exist
            cls._check_source_features_exist(data, resolved_columns)

            result = cls._perform_window_operation(
                data, window_function, window_size, time_unit, resolved_columns, time_filter_feature
            )

            data = cls._add_result_to_data(data, feature.get_name(), result)

        return data

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
    def _perform_window_operation(
        cls,
        data: Any,
        window_function: str,
        window_size: int,
        time_unit: str,
        in_features: List[str],
        time_filter_feature: Optional[str] = None,
    ) -> Any:
        """
        Method to perform the time window operation. Should be implemented by subclasses.

        Supports both single-column and multi-column window operations:
        - Single column: [feature_name] - performs window operation on the column
        - Multi-column: [feature~0, feature~1, ...] - performs window operation across columns

        Args:
            data: The input data
            window_function: The type of window function to perform
            window_size: The size of the window
            time_unit: The time unit for the window
            in_features: List of resolved source feature names to perform window operation on
            time_filter_feature: The name of the time filter feature to use for time-based operations.
                                If None, uses the value from get_time_filter_feature().

        Returns:
            The result of the window operation
        """
        ...
