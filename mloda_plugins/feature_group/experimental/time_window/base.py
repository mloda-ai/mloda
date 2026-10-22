"""
Base implementation for time window feature groups.
"""

from __future__ import annotations

import datetime
from typing import Any, Callable, cast

from mloda.provider import OperationFeatureGroup, SubtypeDeclaration
from mloda.user import Feature
from mloda.provider import (
    FeatureChainParser,
    FeatureChainParserMixin,
)
from mloda.user import FeatureName
from mloda.provider import FeatureSet
from mloda.user import Options
from mloda.provider import DefaultOptionKeys
from mloda.provider import PropertySpec, is_positive_int
from mloda_plugins.feature_group.experimental.key_features import with_key_features
from mloda_plugins.feature_group.experimental.time_reference_mixin import TimeReferenceMixin


class TimeWindowFeatureGroup(TimeReferenceMixin, FeatureChainParserMixin, OperationFeatureGroup):
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

    Reference contract: per window null is skipped; NaN propagates in sum, avg, mean, std, var;
    count counts non-null values including NaN; min, max and median skip NaN; first and last take the
    window's first and last value as is. A multi-column feature reduces the per-column results across
    columns with the same function, skipping missing values (first and last use mean, count gives the
    number of columns); a row with no value gives NaN.

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
    - By default, the feature group will use DefaultOptionKeys.reference_time (default: "reference_time")
    - You can specify a custom time column by setting the reference_time option in the feature group options

    """

    # Option keys for time window configuration
    WINDOW_FUNCTION = "window_function"
    WINDOW_SIZE = "window_size"
    TIME_UNIT = "time_unit"

    SUBTYPES = SubtypeDeclaration(key=WINDOW_FUNCTION)

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

    # Define PROPERTY_MAPPING for the new unified parser approach
    PROPERTY_MAPPING = {
        WINDOW_FUNCTION: PropertySpec(
            "Window function to apply over the time window",
            allowed_values=WINDOW_FUNCTIONS,
            context=True,
            strict_validation=True,
        ),
        WINDOW_SIZE: PropertySpec(
            "Size of the time window (must be positive integer)",
            context=True,
            strict_validation=True,
            element_validator=is_positive_int,
        ),
        TIME_UNIT: PropertySpec(
            "Time unit of the window size",
            allowed_values=TimeReferenceMixin.TIME_UNITS,
            context=True,
            strict_validation=True,
        ),
        DefaultOptionKeys.in_features: PropertySpec(
            "Source feature to apply time window operation to",
            context=True,
            strict_validation=False,
        ),
        DefaultOptionKeys.reference_time: TimeReferenceMixin.REFERENCE_TIME_SPEC,
    }

    # Named captures bind window_function/window_size/time_unit; core validates them at match time.
    PREFIX_PATTERN = r".*__(?P<window_function>[\w]+)_(?P<window_size>\d+)_(?P<time_unit>[\w]+)_window$"

    # In-feature configuration for FeatureChainParserMixin
    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        """Source features from the shared resolution plus the reference-time feature."""
        source_features = super().input_features(options, feature_name) or set()
        return with_key_features(source_features, [self.get_reference_time_column(options)])

    @classmethod
    def _has_valid_time_window_suffix(cls, feature_name: str) -> bool:
        """Check if feature_name has a suffix matching the time window pattern."""
        parsed = FeatureChainParser.parse_name(feature_name, cls._get_prefix_patterns())
        if not parsed.matched:
            return False
        captures = cast(dict[str, str], parsed.named_captures)
        if captures[cls.WINDOW_FUNCTION] not in cls.WINDOW_FUNCTIONS:
            return False
        if captures[cls.TIME_UNIT] not in cls.TIME_UNITS:
            return False
        return int(captures[cls.WINDOW_SIZE]) > 0

    @classmethod
    def _extract_time_window_params(cls, feature: Feature) -> tuple[str | None, int | None, str | None]:
        """
        Extract time window parameters (window_function, window_size, time_unit) from a feature.

        Resolves each parameter from the feature name first, then from options.

        Args:
            feature: The feature to extract parameters from

        Returns:
            Tuple of (window_function, window_size, time_unit), where any value may be None if not found
        """
        window_function = cls._resolve_operation(feature, cls.WINDOW_FUNCTION)
        window_size: Any = cls._resolve_operation(feature, cls.WINDOW_SIZE)
        time_unit = cls._resolve_operation(feature, cls.TIME_UNIT)

        if window_size is not None:
            window_size = int(window_size)

        return window_function, window_size, time_unit

    @classmethod
    def _extract_time_window_params_and_source_features(cls, feature: Feature) -> tuple[str, int, str, str]:
        """
        Extract time window parameters and source feature from a feature.

        Tries string-based parsing first, falls back to configuration-based approach.

        Args:
            feature: The feature to extract parameters from

        Returns:
            Tuple of (window_function, window_size, time_unit, source_feature_name)

        Raises:
            ValueError: If parameters cannot be extracted
        """
        window_function, window_size, time_unit = cls._extract_time_window_params(feature)

        if window_function is None or window_size is None or time_unit is None:
            raise ValueError(f"Could not extract time window parameters from: {feature.name}")

        return window_function, window_size, time_unit, cls._extract_single_source_feature(feature)

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
        parsed = FeatureChainParser.parse_name(feature_name, cls._get_prefix_patterns())
        if not parsed.matched:
            raise ValueError(
                f"Invalid time window feature name format: {feature_name}. "
                f"Expected format: {{in_features}}__{{window_function}}_{{window_size}}_{{time_unit}}_window"
            )

        captures = cast(dict[str, str], parsed.named_captures)
        window_function = captures[cls.WINDOW_FUNCTION]
        window_size_str = captures[cls.WINDOW_SIZE]
        time_unit = captures[cls.TIME_UNIT]

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
        if int(window_size_str) <= 0:
            raise ValueError(f"Invalid window size: {window_size_str}. Must be a positive integer.")
        window_size = int(window_size_str)

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
    def _get_time_delta(cls, window_size: int, time_unit: str) -> datetime.timedelta:
        """
        Convert window size and time unit into the window span as a timedelta.

        Shared by both backends so the time-based window is IDENTICAL across them.
        week/month/year use fixed day approximations (7 / 30 / 365 days).
        """
        if time_unit == "second":
            return datetime.timedelta(seconds=window_size)
        elif time_unit == "minute":
            return datetime.timedelta(minutes=window_size)
        elif time_unit == "hour":
            return datetime.timedelta(hours=window_size)
        elif time_unit == "day":
            return datetime.timedelta(days=window_size)
        elif time_unit == "week":
            return datetime.timedelta(weeks=window_size)
        elif time_unit == "month":
            return datetime.timedelta(days=30 * window_size)
        elif time_unit == "year":
            return datetime.timedelta(days=365 * window_size)
        else:
            raise ValueError(f"Unsupported time unit: {time_unit}")

    # match_feature_group_criteria() inherited from FeatureChainParserMixin

    @classmethod
    def window(
        cls,
        data: Any,
        features: FeatureSet,
        *,
        perform: Callable[[Any, str, datetime.timedelta, list[str], str], Any],
        add_column: Callable[[Any, str, Any], Any],
        columns: Callable[[Any], set[str]],
        column_semantics: Callable[[Any, str], Any],
    ) -> Any:
        """Shared loop: validate the reference time, resolve each feature's sources, run perform, add the result."""
        _options = None
        for feature in features.get_sorted_features():
            if _options:
                if _options != feature.options:
                    raise ValueError("All features must have the same options.")
            _options = feature.options

        reference_time_column = cls.get_reference_time_column(_options)

        if reference_time_column not in columns(data):
            raise ValueError(
                f"Reference time column '{reference_time_column}' not found in data. "
                f"Please ensure the data contains this column."
            )
        cls._validate_reference_time_column(column_semantics(data, reference_time_column), reference_time_column)

        for feature in features.get_sorted_features():
            window_function, window_size, time_unit, in_features = cls._extract_time_window_params_and_source_features(
                feature
            )
            available = columns(data)
            resolved = cls.resolve_multi_column_feature(in_features, available)
            if not any(name in available for name in resolved):
                listing = sorted(available, key=str)
                raise ValueError(f"None of the source features {resolved} found in data. Available columns: {listing}")
            span = cls._get_time_delta(window_size, time_unit)
            result = perform(data, window_function, span, resolved, reference_time_column)
            data = add_column(data, feature.name, result)

        return data


def raise_null_reference_time(reference_time_column: str) -> None:
    """Raise a uniform ValueError when the reference time column contains null/NaT values."""
    raise ValueError(
        f"Reference time column '{reference_time_column}' contains null values. "
        f"Time window operations require non-null reference times."
    )
