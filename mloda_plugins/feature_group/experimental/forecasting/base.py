"""
Base implementation for forecasting feature groups.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, cast

from mloda.provider import FeatureGroup
from mloda.provider import BaseArtifact
from mloda.user import Feature
from mloda.provider import FeatureChainParser, FeatureChainParserMixin, FeatureSet
from mloda.provider import COLUMN_DISCOVERY_HOOKS
from mloda.user import FeatureName
from mloda.user import Options
from mloda.provider import DefaultOptionKeys
from mloda.provider import PropertySpec, is_positive_int
from mloda_plugins.feature_group.experimental.forecasting.forecasting_artifact import ForecastingArtifact
from mloda_plugins.feature_group.experimental.time_reference_mixin import TimeReferenceMixin


def _is_bool(value: Any) -> bool:
    """Accept only a real bool, so 1 or "true" is rejected rather than silently coerced."""
    return isinstance(value, bool)


# Selector tail of PREFIX_PATTERN: excludes the reserved names lower/upper and `_`.
_SELECTOR_TAIL = r"(?!(?:lower|upper)\Z)[^~_\n]+"


class ForecastingFeatureGroup(TimeReferenceMixin, FeatureChainParserMixin, FeatureGroup):
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

    Multi-column (`~`) sources produce one forecast per column as `name~<suffix>` (bounds as
    `name~<suffix>~lower/~upper`), with one model per column. Request `name~<suffix>` for a
    single column; a forecast chained on `name` skips the bounds.

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
    - By default, the feature group will use DefaultOptionKeys.reference_time (default: "reference_time")
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

    # Define the prefix pattern for this feature group
    PREFIX_PATTERN = rf".*__(?P<algorithm>[\w]+)_forecast_(?P<horizon>\d+)(?P<time_unit>[\w]+)(?:~{_SELECTOR_TAIL})?$"

    # In-feature configuration for FeatureChainParserMixin
    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    # Hooks calculate_feature calls: _get_available_columns, _check_source_features_exist, _add_result_to_data.
    REQUIRED_COLUMNWISE_HOOKS = COLUMN_DISCOVERY_HOOKS

    # Property mapping for configuration-based features with group/context separation
    PROPERTY_MAPPING = {
        ALGORITHM: PropertySpec(
            "Forecasting algorithm to use",
            allowed_values=FORECASTING_ALGORITHMS,
            context=True,
            strict_validation=True,
        ),
        HORIZON: PropertySpec(
            "Forecast horizon (number of time units to predict)",
            context=True,
            strict_validation=True,
            element_validator=is_positive_int,
        ),
        TIME_UNIT: PropertySpec(
            "Time unit of the forecast horizon",
            allowed_values=TimeReferenceMixin.TIME_UNITS,
            context=True,
            strict_validation=True,
        ),
        DefaultOptionKeys.in_features: PropertySpec(
            "Source feature to generate forecasts for",
            context=True,
            strict_validation=False,
        ),
        # Both hooks share _is_bool. element_validator produces the rejection message and checks the
        # declared default at construction; it runs on BOTH match paths, so it alone enforces the
        # value space. match_guard additionally judges the raw, un-unpacked value.
        OUTPUT_CONFIDENCE_INTERVALS: PropertySpec(
            "Whether to output confidence intervals as separate columns using ~lower and ~upper suffix pattern",
            context=True,
            strict_validation=True,
            default=False,
            element_validator=_is_bool,
            match_guard=_is_bool,
        ),
        DefaultOptionKeys.reference_time: TimeReferenceMixin.REFERENCE_TIME_SPEC,
    }

    @staticmethod
    def artifact() -> type[BaseArtifact] | None:
        """
        Returns the artifact class for this feature group.

        The ForecastingFeatureGroup uses the ForecastingArtifact to store
        trained models and other components needed for forecasting.
        """
        return ForecastingArtifact

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        """Source features from the shared resolution plus the reference-time feature."""
        source_features = super().input_features(options, feature_name) or set()
        return source_features | {Feature(self.get_reference_time_column(options))}

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
        parsed = FeatureChainParser.parse_name(feature_name, cls._get_prefix_patterns())
        if not parsed.matched:
            raise ValueError(
                f"Invalid forecast feature name format: {feature_name}. "
                f"Expected format: {{in_features}}__{{algorithm}}_forecast_{{horizon}}{{time_unit}}"
            )

        captures = cast(dict[str, str], parsed.named_captures)
        algorithm = captures[cls.ALGORITHM]
        horizon_str = captures[cls.HORIZON]
        time_unit = captures[cls.TIME_UNIT]

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
        if not horizon_str.isdigit() or int(horizon_str) <= 0:
            raise ValueError(f"Invalid horizon: {horizon_str}. Must be a positive integer.")
        horizon = int(horizon_str)

        return algorithm, horizon, time_unit

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
        for feature in features.get_sorted_features():
            if _options:
                if _options != feature.options:
                    raise ValueError("All features must have the same options.")
            _options = feature.options

        reference_time_column = cls.get_reference_time_column(_options)

        cls._check_reference_time_column_exists(data, reference_time_column)
        cls._check_reference_time_column_is_datetime(data, reference_time_column)

        # Store the original clean data
        original_data = data

        # Collect all results before modifying the data
        results: list[tuple[str, Any]] = []

        # Process each requested feature with the original clean data
        for feature in features.get_sorted_features():
            algorithm, horizon, time_unit, in_features = cls._extract_forecasting_parameters(feature)

            # A "~" source such as "product__onehot_encoded" expands to ["...~0", "...~1", ...]
            available_columns = cls._get_available_columns(original_data)
            resolved_columns = cls.resolve_multi_column_feature(in_features, available_columns)
            # Chaining skips bound columns such as "~0~lower" or "~0~upper"
            resolved_columns = [
                c
                for c in resolved_columns
                if c == in_features or not c.removeprefix(f"{in_features}~").endswith(("~lower", "~upper"))
            ]

            # Check that resolved columns exist
            cls._check_source_features_exist(original_data, resolved_columns)

            # Check if we have a trained model in the artifact
            model_artifact = None
            if features.artifact_to_load is not None:
                model_artifact = cls.load_artifact(features)
                if model_artifact is None:
                    raise ValueError("No artifact to load although it was requested.")

            output_confidence_intervals = feature.options.get(cls.OUTPUT_CONFIDENCE_INTERVALS)

            # Suffix per resolved column: None for the literal source, else the part after "<in_features>~"
            expanded_prefix = f"{in_features}~"
            suffixes: list[str | None] = [
                None if column == in_features else column.removeprefix(expanded_prefix) for column in resolved_columns
            ]
            selector = cls._extract_selector(feature.name)
            if selector is not None:
                if all(suffix is None for suffix in suffixes):
                    raise ValueError(f"Source '{in_features}' is a single column and takes no selector '~{selector}'.")
                if selector not in suffixes:
                    available = sorted(str(s) for s in suffixes if s is not None)
                    raise ValueError(
                        f"Unknown selector '~{selector}' for source '{in_features}'. Available suffixes: {available}"
                    )
                resolved_columns = [f"{in_features}~{selector}"]
                suffixes = [None]
            expanded = any(suffix is not None for suffix in suffixes)
            column_artifacts = cls._column_artifacts(model_artifact, suffixes, expanded, selector)

            new_artifacts: dict[str, Any] = {}
            for column, suffix in zip(resolved_columns, suffixes):
                output_name = feature.name if suffix is None else f"{feature.name}~{suffix}"
                column_artifact = column_artifacts.get(suffix) if column_artifacts is not None else None

                if output_confidence_intervals:
                    result, lower_bound, upper_bound, updated_artifact = cls._perform_forecasting_with_confidence(
                        original_data,
                        algorithm,
                        horizon,
                        time_unit,
                        column,
                        reference_time_column,
                        column_artifact,
                    )
                    results.append((output_name, result))
                    results.append((f"{output_name}~lower", lower_bound))
                    results.append((f"{output_name}~upper", upper_bound))
                else:
                    result, updated_artifact = cls._perform_forecasting(
                        original_data,
                        algorithm,
                        horizon,
                        time_unit,
                        column,
                        reference_time_column,
                        column_artifact,
                    )
                    results.append((output_name, result))

                if updated_artifact:
                    new_artifacts["" if suffix is None else suffix] = updated_artifact

            owns_artifact = features.artifact_to_save == feature.name
            if owns_artifact and new_artifacts and features.artifact_to_load is None:
                features.save_artifact = {"columns": new_artifacts} if expanded else new_artifacts[""]

        # Add all results to the data at once
        for feature_name, result in results:
            data = cls._add_result_to_data(data, feature_name, result)

        return data

    @classmethod
    def _extract_selector(cls, feature_name: str) -> str | None:
        """Return the trailing `~<selector>` of a string-based forecast name, else None."""
        if not cls._has_valid_forecast_suffix(feature_name):
            return None
        parsed = FeatureChainParser.parse_name(feature_name, cls._get_prefix_patterns())
        return cast(str, parsed.operation_part).partition("~")[2] or None

    @classmethod
    def _column_artifacts(
        cls, model_artifact: Any | None, suffixes: list[str | None], expanded: bool, selector: str | None = None
    ) -> dict[str | None, Any] | None:
        """Split a loaded artifact per column suffix, raising ValueError when its shape does not match the source."""
        if model_artifact is None:
            return None
        nested = "columns" in model_artifact
        if selector is not None and nested:
            if selector not in model_artifact["columns"]:
                raise ValueError(
                    f"The loaded artifact has no model for selector '~{selector}'. "
                    f"Artifact columns: {sorted(model_artifact['columns'])}"
                )
            return {None: model_artifact["columns"][selector]}
        if not expanded:
            if nested:
                raise ValueError("The loaded artifact holds per-column models but the source is a single column.")
            return {None: model_artifact}
        if not nested:
            raise ValueError("The loaded artifact is for a single column but the source expands to multiple columns.")
        columns: dict[str, Any] = model_artifact["columns"]
        if set(columns) != {suffix for suffix in suffixes if suffix is not None}:
            raise ValueError(
                f"The loaded artifact columns {sorted(columns)} do not match the source columns "
                f"{sorted(str(suffix) for suffix in suffixes)}."
            )
        return {suffix: artifact for suffix, artifact in columns.items()}

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
        algorithm, horizon, time_unit = cls._extract_forecast_params(feature)
        if algorithm is None or horizon is None or time_unit is None:
            raise ValueError(f"Could not extract forecasting parameters from: {feature.name}")
        return algorithm, horizon, time_unit, cls._extract_single_source_feature(feature)

    @classmethod
    def _has_valid_forecast_suffix(cls, feature_name: str) -> bool:
        """Check if feature_name has a suffix matching the forecast pattern."""
        parsed = FeatureChainParser.parse_name(feature_name, cls._get_prefix_patterns())
        if not parsed.matched:
            return False
        captures = cast(dict[str, str], parsed.named_captures)
        if captures[cls.ALGORITHM] not in cls.FORECASTING_ALGORITHMS:
            return False
        if captures[cls.TIME_UNIT] not in cls.TIME_UNITS:
            return False
        return int(captures[cls.HORIZON]) > 0

    @classmethod
    def _extract_forecast_params(cls, feature: Feature) -> tuple[str | None, int | None, str | None]:
        """
        Extract forecast-specific parameters (algorithm, horizon, time_unit) from a feature.

        Each value comes from the feature name when it owns the match, otherwise from options.

        Args:
            feature: The feature to extract parameters from

        Returns:
            Tuple of (algorithm, horizon, time_unit), where any value may be None if not found
        """
        algorithm = cls._resolve_operation(feature, cls.ALGORITHM)
        horizon: Any = cls._resolve_operation(feature, cls.HORIZON)
        time_unit = cls._resolve_operation(feature, cls.TIME_UNIT)
        if horizon is not None:
            horizon = int(horizon)
        return algorithm, horizon, time_unit

    @classmethod
    @abstractmethod
    def _check_reference_time_column_exists(cls, data: Any, reference_time_column: str) -> None:
        """
        Check if the reference time column exists in the data.

        Args:
            data: The input data
            reference_time_column: The name of the reference time column

        Raises:
            ValueError: If the reference time column does not exist in the data
        """
        ...

    @classmethod
    @abstractmethod
    def _check_reference_time_column_is_datetime(cls, data: Any, reference_time_column: str) -> None:
        """
        Check if the reference time column is a datetime column.

        Args:
            data: The input data
            reference_time_column: The name of the reference time column

        Raises:
            ValueError: If the reference time column is not a datetime column
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
        source_column: str,
        time_filter_feature: str,
        model_artifact: Any | None = None,
    ) -> tuple[Any, Any | None]:
        """
        Method to perform the forecasting. Should be implemented by subclasses.

        Forecasts one source column per call; the caller loops over multi-column sources.

        Args:
            data: The input data
            algorithm: The forecasting algorithm to use
            horizon: The forecast horizon
            time_unit: The time unit for the horizon
            source_column: The single source column to forecast
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
        source_column: str,
        time_filter_feature: str,
        model_artifact: Any | None = None,
    ) -> tuple[Any, Any, Any, Any | None]:
        """
        Method to perform forecasting and return point forecast plus confidence intervals.

        Forecasts one source column per call.

        Args:
            data: The input data
            algorithm: The forecasting algorithm to use
            horizon: The forecast horizon
            time_unit: The time unit for the horizon
            source_column: The single source column to forecast
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
