"""
Base implementation for aggregated feature groups.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, List, Optional, Set, Union

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

    Features follow the naming pattern: `{in_features}__{aggregation_type}_aggr`

    Examples:
    ```python
    features = [
        "sales__sum_aggr",           # Sum of sales values
        "temperature__avg_aggr",     # Average temperature
        "price__max_aggr",           # Maximum price
        "transactions__count_aggr"   # Count of transactions
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
                DefaultOptionKeys.in_features: "sales",
            }
        )
    )
    ```

    ## Parameter Classification

    ### Context Parameters (Default)
    These parameters don't affect Feature Group resolution/splitting:
    - `aggregation_type`: The type of aggregation to perform
    - `in_features`: The source feature to aggregate

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

    PREFIX_PATTERN = r".*__([\w]+)_aggr$"

    # Property mapping for configuration-based feature creation
    PROPERTY_MAPPING = {
        AGGREGATION_TYPE: {
            **AGGREGATION_TYPES,  # All supported aggregation types as valid values
            DefaultOptionKeys.mloda_context: True,  # Mark as context parameter
            DefaultOptionKeys.mloda_strict_validation: True,  # Enable strict validation
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature to aggregate",
            DefaultOptionKeys.mloda_context: True,  # Mark as context parameter
            DefaultOptionKeys.mloda_strict_validation: False,  # Flexible validation
        },
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """Extract source feature from either configuration-based options or string parsing."""

        source_feature: str | None = None

        # string based
        _, source_feature = FeatureChainParser.parse_feature_name(feature_name, [self.PREFIX_PATTERN])
        if source_feature is not None:
            return {Feature(source_feature)}

        # configuration based
        source_features = options.get_in_features()
        if len(source_features) != 1:
            raise ValueError(
                f"Expected exactly one source feature, but found {len(source_features)}: {source_features}"
            )
        return set(source_features)

    @classmethod
    def get_aggregation_type(cls, feature_name: str) -> str:
        """Extract the aggregation type from the feature name."""
        prefix_part, _ = FeatureChainParser.parse_feature_name(feature_name, [cls.PREFIX_PATTERN])
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
            feature.name, [cls.PREFIX_PATTERN]
        )
        if aggregation_type is not None and source_feature_name is not None:
            return aggregation_type, source_feature_name

        # configuration based
        source_features = feature.options.get_in_features()
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

        Supports multi-column features by using resolve_multi_column_feature() to
        automatically discover columns matching the pattern feature_name~N.

        Adds the aggregated results directly to the input data structure.
        """
        # Process each requested feature
        for feature in features.features:
            aggregation_type, source_feature_name = cls._extract_aggr_and_source_feature(feature)

            # Resolve multi-column features automatically
            # If source_feature_name is "onehot_encoded__product", this discovers
            # ["onehot_encoded__product~0", "onehot_encoded__product~1", ...]
            available_columns = cls._get_available_columns(data)
            resolved_columns = cls.resolve_multi_column_feature(source_feature_name, available_columns)

            # Check that resolved columns exist
            cls._check_source_features_exist(data, resolved_columns)

            if aggregation_type not in cls.AGGREGATION_TYPES:
                raise ValueError(f"Unsupported aggregation type: {aggregation_type}")

            result = cls._perform_aggregation(data, aggregation_type, resolved_columns)

            data = cls._add_result_to_data(data, feature.get_name(), result)

        return data

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
    def _perform_aggregation(cls, data: Any, aggregation_type: str, in_features: List[str]) -> Any:
        """
        Method to perform the aggregation. Should be implemented by subclasses.

        Supports both single-column and multi-column aggregation:
        - Single column: [feature_name] - aggregates values within the column
        - Multi-column: [feature~0, feature~1, ...] - aggregates across columns

        Args:
            data: The input data
            aggregation_type: The type of aggregation to perform
            in_features: List of resolved source feature names to aggregate

        Returns:
            The result of the aggregation
        """
        ...
