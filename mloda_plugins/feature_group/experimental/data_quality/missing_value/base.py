"""
Base implementation for missing value imputation feature groups.
"""

from __future__ import annotations

import copy
from abc import abstractmethod
from typing import Any, List, Optional, Set, Union

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


class MissingValueFeatureGroup(AbstractFeatureGroup):
    """
    Base class for all missing value imputation feature groups.

    Missing value feature groups impute missing values in the source feature using
    the specified imputation method. They support both string-based feature creation
    and configuration-based creation with proper group/context parameter separation.

    ## Supported Imputation Methods

    - `mean`: Impute with the mean of non-missing values
    - `median`: Impute with the median of non-missing values
    - `mode`: Impute with the most frequent value
    - `constant`: Impute with a specified constant value
    - `ffill`: Forward fill (use the last valid value)
    - `bfill`: Backward fill (use the next valid value)

    ## Feature Creation Methods

    ### 1. String-Based Creation

    Features follow the naming pattern: `{in_features}__{imputation_method}_imputed`

    Examples:
    ```python
    features = [
        "income__mean_imputed",      # Impute missing values in income with the mean
        "age__median_imputed",       # Impute missing values in age with the median
        "category__constant_imputed" # Impute missing values in category with a constant value
    ]
    ```

    ### 2. Configuration-Based Creation

    Uses Options with proper group/context parameter separation:

    ```python
    feature = Feature(
        name="placeholder",  # Placeholder name, will be replaced
        options=Options(
            context={
                MissingValueFeatureGroup.IMPUTATION_METHOD: "mean",
                DefaultOptionKeys.in_features: "income",
            }
        )
    )
    ```

    ## Parameter Classification

    ### Context Parameters (Default)
    These parameters don't affect Feature Group resolution/splitting:
    - `imputation_method`: The type of imputation to perform
    - `in_features`: The source feature to impute missing values
    - `constant_value`: Constant value for constant imputation (optional)
    - `group_by_features`: Features to group by before imputation (optional)

    ### Group Parameters
    Currently none for MissingValueFeatureGroup. Parameters that affect Feature Group
    resolution/splitting would be placed here.

    ## Usage Examples

    ### String-Based Creation

    ```python
    from mloda_core.abstract_plugins.components.feature import Feature

    # Impute missing income values with mean
    feature = Feature(name="income__mean_imputed")

    # Impute missing age values with median
    feature = Feature(name="age__median_imputed")

    # Impute missing category values with mode
    feature = Feature(name="category__mode_imputed")

    # Forward fill missing temperature values
    feature = Feature(name="temperature__ffill_imputed")
    ```

    ### Configuration-Based Creation

    ```python
    from mloda_core.abstract_plugins.components.feature import Feature
    from mloda_core.abstract_plugins.components.options import Options
    from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

    # Mean imputation using configuration
    feature = Feature(
        name="placeholder",
        options=Options(
            context={
                MissingValueFeatureGroup.IMPUTATION_METHOD: "mean",
                DefaultOptionKeys.in_features: "income",
            }
        )
    )

    # Constant imputation with a specific value
    feature = Feature(
        name="placeholder",
        options=Options(
            context={
                MissingValueFeatureGroup.IMPUTATION_METHOD: "constant",
                DefaultOptionKeys.in_features: "status",
                "constant_value": "unknown",
            }
        )
    )

    # Group-based imputation (e.g., mean by category)
    feature = Feature(
        name="placeholder",
        options=Options(
            context={
                MissingValueFeatureGroup.IMPUTATION_METHOD: "mean",
                DefaultOptionKeys.in_features: "price",
                "group_by_features": ["product_category", "region"],
            }
        )
    )
    ```

    ## Requirements
    - Input data must contain the source feature to be imputed
    - For group-based imputation, grouping features must also be present
    - For constant imputation, a constant_value must be provided
    """

    IMPUTATION_METHOD = "imputation_method"
    # Define supported imputation methods
    IMPUTATION_METHODS = {
        "mean": "Impute with the mean of non-missing values",
        "median": "Impute with the median of non-missing values",
        "mode": "Impute with the most frequent value",
        "constant": "Impute with a specified constant value",
        "ffill": "Forward fill (use the last valid value)",
        "bfill": "Backward fill (use the next valid value)",
    }

    PREFIX_PATTERN = r".*__([\w]+)_imputed$"

    PROPERTY_MAPPING = {
        IMPUTATION_METHOD: {
            **IMPUTATION_METHODS,
            DefaultOptionKeys.mloda_context: True,
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature to impute missing values",
            DefaultOptionKeys.mloda_context: True,
        },
        "constant_value": {
            "explanation": "Constant value to use for constant imputation method",
            DefaultOptionKeys.mloda_context: True,
            DefaultOptionKeys.mloda_default: None,  # Default is None, required only for constant method
        },
        "group_by_features": {
            "explanation": "Optional list of features to group by before imputation",
            DefaultOptionKeys.mloda_context: True,
            DefaultOptionKeys.mloda_default: None,  # Default is None (no grouping)
        },
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """Extract source feature from either configuration-based options or string parsing."""

        source_feature: str | None = None

        # Try string-based parsing first
        # parse_feature_name returns (operation_config, source_feature)
        operation_config, source_feature = FeatureChainParser.parse_feature_name(feature_name, [self.PREFIX_PATTERN])
        if source_feature is not None:
            return {Feature(source_feature)}

        # Fall back to configuration-based approach
        source_features = options.get_in_features()
        if len(source_features) != 1:
            raise ValueError(
                f"Expected exactly one source feature, but found {len(source_features)}: {source_features}"
            )
        return set(source_features)

    @classmethod
    def get_imputation_method(cls, feature_name: str) -> str:
        """Extract the imputation method from the feature name."""
        # parse_feature_name returns (operation_config, source_feature)
        # The operation_config contains the imputation method extracted from the suffix pattern
        operation_config, _ = FeatureChainParser.parse_feature_name(feature_name, [cls.PREFIX_PATTERN])
        if operation_config is None:
            raise ValueError(f"Invalid missing value feature name format: {feature_name}")

        # The PREFIX_PATTERN captures the method name (e.g., "mean" from "mean_imputed")
        # So operation_config already contains just the method name
        imputation_method = operation_config

        # Validate imputation method
        if imputation_method not in cls.IMPUTATION_METHODS:
            raise ValueError(
                f"Unsupported imputation method: {imputation_method}. "
                f"Supported methods: {', '.join(cls.IMPUTATION_METHODS.keys())}"
            )

        return imputation_method

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Optional[Any] = None,
    ) -> bool:
        """Check if feature name matches the expected pattern for missing value features."""

        # Use the unified parser with property mapping for full configuration support
        return FeatureChainParser.match_configuration_feature_chain_parser(
            feature_name,
            options,
            property_mapping=cls.PROPERTY_MAPPING,
            prefix_patterns=[cls.PREFIX_PATTERN],
        )

    @classmethod
    def _extract_imputation_method_and_source_feature(cls, feature: Feature) -> tuple[str, str]:
        """
        Extract imputation method and source feature name from a feature.

        Tries string-based parsing first, falls back to configuration-based approach.

        Args:
            feature: The feature to extract parameters from

        Returns:
            Tuple of (imputation_method, source_feature_name)

        Raises:
            ValueError: If parameters cannot be extracted
        """
        imputation_method = None
        source_feature_name: str | None = None

        # Try string-based parsing first
        feature_name_str = feature.name.name if hasattr(feature.name, "name") else str(feature.name)

        if FeatureChainParser.is_chained_feature(feature_name_str):
            # Use get_imputation_method which already handles parse_feature_name correctly
            imputation_method = cls.get_imputation_method(feature_name_str)
            # Use extract_source_feature which returns everything before the last __
            source_feature_name = FeatureChainParser.extract_source_feature(feature_name_str, cls.PREFIX_PATTERN)
            return imputation_method, source_feature_name

        # Fall back to configuration-based approach
        source_features = feature.options.get_in_features()
        source_feature = next(iter(source_features))
        source_feature_name = source_feature.get_name()

        imputation_method = feature.options.get(cls.IMPUTATION_METHOD)

        if imputation_method is None or source_feature_name is None:
            raise ValueError(f"Could not extract imputation method and source feature from: {feature.name}")

        # Validate imputation method (no need to strip "imputed" from config-based method)
        if imputation_method not in cls.IMPUTATION_METHODS:
            raise ValueError(
                f"Unsupported imputation method: {imputation_method}. "
                f"Supported methods: {', '.join(cls.IMPUTATION_METHODS.keys())}"
            )

        return imputation_method, source_feature_name

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """
        Perform missing value imputation.

        Processes all requested features, determining the imputation method
        and source feature from either string parsing or configuration-based options.

        Adds the imputed results directly to the input data structure.
        """

        # Process each requested feature
        for feature in features.features:
            imputation_method, source_feature = cls._extract_imputation_method_and_source_feature(feature)

            # Resolve multi-column features automatically
            # If source_feature is "onehot_encoded__product", this discovers
            # ["onehot_encoded__product~0", "onehot_encoded__product~1", ...]
            available_columns = cls._get_available_columns(data)
            resolved_columns = cls.resolve_multi_column_feature(source_feature, available_columns)

            constant_value = feature.options.get("constant_value")
            group_by_features = feature.options.get("group_by_features")

            cls._check_source_features_exist(data, resolved_columns)

            # Validate group by features if provided
            if group_by_features:
                for group_feature in group_by_features:
                    cls._check_source_features_exist(data, [group_feature])

            # Validate constant value is provided for constant imputation
            if imputation_method == "constant" and constant_value is None:
                raise ValueError("Constant value must be provided for constant imputation method")

            # Apply the appropriate imputation function
            result = cls._perform_imputation(
                data, imputation_method, resolved_columns, constant_value, group_by_features
            )

            # Add the result to the data
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
    def _perform_imputation(
        cls,
        data: Any,
        imputation_method: str,
        in_features: List[str],
        constant_value: Optional[Any] = None,
        group_by_features: Optional[List[str]] = None,
    ) -> Any:
        """
        Method to perform the imputation. Should be implemented by subclasses.

        Supports both single-column and multi-column imputation:
        - Single column: [feature_name] - imputes values within the column
        - Multi-column: [feature~0, feature~1, ...] - imputes across columns

        Args:
            data: The input data
            imputation_method: The type of imputation to perform
            in_features: List of resolved source feature names to impute
            constant_value: The constant value to use for imputation (if method is 'constant')
            group_by_features: Optional list of features to group by before imputation

        Returns:
            The result of the imputation
        """
        ...
