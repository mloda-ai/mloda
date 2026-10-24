"""
Base implementation for aggregated feature groups.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mloda.user import Feature
from mloda.provider import FeatureSet
from mloda.provider import (
    FeatureChainParserMixin,
)
from mloda.provider import OperationFeatureGroup
from mloda.provider import DefaultOptionKeys
from mloda.provider import PropertySpec
from mloda.provider import SubtypeDeclaration


class AggregatedFeatureGroup(FeatureChainParserMixin, OperationFeatureGroup):
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
    - `count`: Count of non-missing values
    - `std`: Standard deviation of values
    - `var`: Variance of values
    - `median`: Median value

    Reference contract: null and NaN are both missing and skipped; count counts non-missing values.
    Over no values (empty or all missing), sum and count are 0 and the other functions give a missing value.

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

    # Declarative subtype dimension of this family.
    SUBTYPES = SubtypeDeclaration(key=AGGREGATION_TYPE)

    # Define supported aggregation types
    AGGREGATION_TYPES = {
        "sum": "Sum of values",
        "min": "Minimum value",
        "max": "Maximum value",
        "avg": "Average (mean) of values",
        "mean": "Average (mean) of values",
        "count": "Count of non-missing values",
        "std": "Standard deviation of values",
        "var": "Variance of values",
        "median": "Median value",
    }

    PREFIX_PATTERN = r".*__(?P<aggregation_type>[\w]+)_aggr$"

    # In-feature configuration for FeatureChainParserMixin
    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    # Property mapping for configuration-based feature creation
    PROPERTY_MAPPING = {
        AGGREGATION_TYPE: PropertySpec(
            "Type of aggregation to perform",
            allowed_values=AGGREGATION_TYPES,
            context=True,
            strict_validation=True,
        ),
        DefaultOptionKeys.in_features: PropertySpec(
            "Source feature to aggregate",
            context=True,
            strict_validation=False,
        ),
    }

    @classmethod
    def get_aggregation_type(cls, feature_name: str) -> str:
        """Extract the aggregation type from the feature name."""
        prefix_part = cls.resolve_feature_name(feature_name).value_for(cls.AGGREGATION_TYPE)
        if prefix_part is None:
            raise ValueError(f"Could not extract aggregation type from feature name: {feature_name}")
        return prefix_part

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
        return cls._extract_operation_and_source_feature(
            feature, lambda f: cls._resolve_operation(f, cls.AGGREGATION_TYPE), "aggregation type"
        )

    @classmethod
    def _supports_aggregation_type(cls, aggregation_type: str) -> bool:
        """Check if this feature group supports the given aggregation type."""
        return aggregation_type in cls.AGGREGATION_TYPES

    @classmethod
    def aggregate(
        cls,
        data: Any,
        features: FeatureSet,
        *,
        perform: Callable[[Any, str, list[str]], Any],
        add_column: Callable[[Any, str, Any], Any],
        columns: Callable[[Any], set[str]],
    ) -> Any:
        """Shared loop: resolve each feature's source columns, run perform, add the result with add_column."""
        for feature in features.get_sorted_features():
            aggregation_type, source_feature_name = cls._extract_aggr_and_source_feature(feature)
            available = columns(data)
            resolved = cls.resolve_multi_column_feature(source_feature_name, available)
            if not any(name in available for name in resolved):
                listing = sorted(available, key=str)
                raise ValueError(f"None of the source features {resolved} found in data. Available columns: {listing}")
            if aggregation_type not in cls.AGGREGATION_TYPES:
                raise ValueError(f"Unsupported aggregation type: {aggregation_type}")
            data = add_column(data, feature.name, perform(data, aggregation_type, resolved))
        return data
