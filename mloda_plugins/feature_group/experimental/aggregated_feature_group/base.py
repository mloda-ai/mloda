"""
Base implementation for aggregated feature groups.
"""

from __future__ import annotations

from typing import Any, Optional, Set, Union

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


class AggregatedFeatureGroup(AbstractFeatureGroup):
    """
    Base class for all aggregated feature groups.

    The AggregatedFeatureGroup performs aggregation operations on source features,
    such as sum, average, minimum, maximum, etc. It supports both string-based
    feature creation and configuration-based creation with proper group/context
    parameter separation.

    ## Supported Aggregation Types

    - `sum`: Sum of values
    - `min`: Minimum value
    - `max`: Maximum value
    - `avg`: Average (mean) of values
    - `mean`: Average (mean) of values
    - `count`: Count of non-null values
    - `std`: Standard deviation of values
    - `var`: Variance of values
    - `median`: Median value

    ## Feature Creation Methods

    ### 1. String-Based Creation

    Features follow the naming pattern: `{aggregation_type}_aggr__{mloda_source_feature}`

    Examples:
    ```python
    features = [
        "sum_aggr__sales",           # Sum of sales values
        "avg_aggr__temperature",     # Average temperature
        "max_aggr__price",           # Maximum price
        "count_aggr__transactions"   # Count of transactions
    ]
    ```

    ### 2. Configuration-Based Creation

    Uses Options with proper group/context parameter separation:

    ```python
    feature = Feature(
        name="placeholder",  # Placeholder name, will be replaced
        options=Options(
            context={
                AggregatedFeatureGroup.AGGREGATION_TYPE: "sum",
                DefaultOptionKeys.mloda_source_feature: "sales",
            }
        )
    )
    ```

    ## Parameter Classification

    ### Context Parameters (Default)
    These parameters don't affect Feature Group resolution/splitting:
    - `aggregation_type`: The type of aggregation to perform
    - `mloda_source_feature`: The source feature to aggregate

    ### Group Parameters
    Currently none for AggregatedFeatureGroup. Parameters that affect Feature Group
    resolution/splitting would be placed here.
    """

    # Option key for aggregation type
    AGGREGATION_TYPE = "aggregation_type"

    # Define supported aggregation types
    AGGREGATION_TYPES = {
        "sum": "Sum of values",
        "min": "Minimum value",
        "max": "Maximum value",
        "avg": "Average (mean) of values",
        "mean": "Average (mean) of values",
        "count": "Count of non-null values",
        "std": "Standard deviation of values",
        "var": "Variance of values",
        "median": "Median value",
    }

    PATTERN = "_aggr__"
    PREFIX_PATTERN = r"^([\w]+)_aggr__"

    # Property mapping for configuration-based feature creation
    PROPERTY_MAPPING = {
        AGGREGATION_TYPE: {
            **AGGREGATION_TYPES,  # All supported aggregation types as valid values
            DefaultOptionKeys.mloda_context: True,  # Mark as context parameter
            DefaultOptionKeys.mloda_strict_validation: True,  # Enable strict validation
        },
        DefaultOptionKeys.mloda_source_feature: {
            "explanation": "Source feature to aggregate",
            DefaultOptionKeys.mloda_context: True,  # Mark as context parameter
            DefaultOptionKeys.mloda_strict_validation: False,  # Flexible validation
        },
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """Extract source feature from either configuration-based options or string parsing."""

        source_feature: str | None = None

        # string based
        _, source_feature = FeatureChainParser.parse_feature_name(feature_name, self.PATTERN, [self.PREFIX_PATTERN])
        if source_feature is not None:
            return {Feature(source_feature)}

        # configuration based
        source_features = options.get_source_features()
        if len(source_features) != 1:
            raise ValueError(
                f"Expected exactly one source feature, but found {len(source_features)}: {source_features}"
            )
        return set(source_features)

    @classmethod
    def get_aggregation_type(cls, feature_name: str) -> str:
        """Extract the aggregation type from the feature name."""
        prefix_part, _ = FeatureChainParser.parse_feature_name(feature_name, cls.PATTERN, [cls.PREFIX_PATTERN])
        if prefix_part is None:
            raise ValueError(f"Could not extract aggregation type from feature name: {feature_name}")
        return prefix_part

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Optional[Any] = None,
    ) -> bool:
        """Check if feature name matches the expected pattern and aggregation type."""

        # Use the unified parser with property mapping for full configuration support
        return FeatureChainParser.match_configuration_feature_chain_parser(
            feature_name,
            options,
            property_mapping=cls.PROPERTY_MAPPING,
            pattern=cls.PATTERN,
            prefix_patterns=[cls.PREFIX_PATTERN],
        )

    @classmethod
    def _extract_aggr_and_source_feature(cls, feature: Feature) -> tuple[str, str]:
        """
        Extract aggregation type and source feature name from a feature.

        Tries configuration-based approach first, falls back to string parsing.

        Args:
            feature: The feature to extract parameters from

        Returns:
            Tuple of (aggregation_type, source_feature_name)

        Raises:
            ValueError: If parameters cannot be extracted
        """
        aggregation_type = None
        source_feature_name: str | None = None

        # string based
        aggregation_type, source_feature_name = FeatureChainParser.parse_feature_name(
            feature.name, cls.PATTERN, [cls.PREFIX_PATTERN]
        )
        if aggregation_type is not None and source_feature_name is not None:
            return aggregation_type, source_feature_name

        # configuration based
        source_features = feature.options.get_source_features()
        source_feature = next(iter(source_features))
        source_feature_name = source_feature.get_name()

        aggregation_type = feature.options.get(cls.AGGREGATION_TYPE)

        if aggregation_type is None or source_feature_name is None:
            raise ValueError(f"Could not extract aggregation type and source feature from: {feature.name}")

        return aggregation_type, source_feature_name

    @classmethod
    def _supports_aggregation_type(cls, aggregation_type: str) -> bool:
        """Check if this feature group supports the given aggregation type."""
        return aggregation_type in cls.AGGREGATION_TYPES

    @classmethod
    def _raise_unsupported_aggregation_type(cls, aggregation_type: str) -> bool:
        """
        Raise an error for unsupported aggregation type.
        """
        raise ValueError(f"Unsupported aggregation type: {aggregation_type}")

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """
        Perform aggregations.

        Processes all requested features, determining the aggregation type
        and source feature from either string parsing or configuration-based options.

        Adds the aggregated results directly to the input data structure.
        """
        # Process each requested feature
        for feature in features.features:
            aggregation_type, source_feature_name = cls._extract_aggr_and_source_feature(feature)

            cls._check_source_feature_exists(data, source_feature_name)

            if aggregation_type not in cls.AGGREGATION_TYPES:
                raise ValueError(f"Unsupported aggregation type: {aggregation_type}")

            result = cls._perform_aggregation(data, aggregation_type, source_feature_name)

            data = cls._add_result_to_data(data, feature.get_name(), result)

        return data

    @classmethod
    def _check_source_feature_exists(cls, data: Any, feature_name: str) -> None:
        """
        Check if the source feature exists in the data.

        Args:
            data: The input data
            feature_name: The name of the feature to check

        Raises:
            ValueError: If the feature does not exist in the data
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
    def _perform_aggregation(cls, data: Any, aggregation_type: str, mloda_source_feature: str) -> Any:
        """
        Method to perform the aggregation. Should be implemented by subclasses.

        Args:
            data: The input data
            aggregation_type: The type of aggregation to perform
            mloda_source_feature: The name of the source feature to aggregate

        Returns:
            The result of the aggregation
        """
        raise NotImplementedError(f"_perform_aggregation not implemented in {cls.__name__}")
