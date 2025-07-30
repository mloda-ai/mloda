"""
Base implementation for missing value imputation feature groups.
"""

from __future__ import annotations

import copy
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

    Features follow the naming pattern: `{imputation_method}_imputed__{mloda_source_feature}`

    Examples:
    ```python
    features = [
        "mean_imputed__income",      # Impute missing values in income with the mean
        "median_imputed__age",       # Impute missing values in age with the median
        "constant_imputed__category" # Impute missing values in category with a constant value
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
                DefaultOptionKeys.mloda_source_feature: "income",
            }
        )
    )
    ```

    ## Parameter Classification

    ### Context Parameters (Default)
    These parameters don't affect Feature Group resolution/splitting:
    - `imputation_method`: The type of imputation to perform
    - `mloda_source_feature`: The source feature to impute missing values

    ### Group Parameters
    Currently none for MissingValueFeatureGroup. Parameters that affect Feature Group
    resolution/splitting would be placed here.

    ## Requirements
    - The input data must contain the source feature to be imputed
    - For group-based imputation, the grouping features must also be present
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

    PATTERN = "__"
    PREFIX_PATTERN = r"^([\w]+)_imputed__"

    PROPERTY_MAPPING = {
        IMPUTATION_METHOD: {
            **IMPUTATION_METHODS,
            DefaultOptionKeys.mloda_context: True,
        },
        DefaultOptionKeys.mloda_source_feature: {
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
        _, source_feature = FeatureChainParser.parse_feature_name(feature_name, self.PATTERN, [self.PREFIX_PATTERN])
        if source_feature is not None:
            return {Feature(source_feature)}

        # Fall back to configuration-based approach
        source_features = options.get_source_features()
        if len(source_features) != 1:
            raise ValueError(
                f"Expected exactly one source feature, but found {len(source_features)}: {source_features}"
            )
        return set(source_features)

    @classmethod
    def get_imputation_method(cls, feature_name: str) -> str:
        """Extract the imputation method from the feature name."""
        imputation_method, _ = FeatureChainParser.parse_feature_name(feature_name, cls.PATTERN, [cls.PREFIX_PATTERN])
        if imputation_method is None:
            raise ValueError(f"Invalid missing value feature name format: {feature_name}")

        imputation_method = imputation_method.replace("imputed", "").strip("_")
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
            pattern=cls.PATTERN,
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

        if cls.PATTERN in feature_name_str:
            imputation_method = cls.get_imputation_method(feature_name_str)
            source_feature_name = FeatureChainParser.extract_source_feature(feature_name_str, cls.PREFIX_PATTERN)
            return imputation_method, source_feature_name

        # Fall back to configuration-based approach
        source_features = feature.options.get_source_features()
        source_feature = next(iter(source_features))
        source_feature_name = source_feature.get_name()

        imputation_method = feature.options.get(cls.IMPUTATION_METHOD)

        if imputation_method is None or source_feature_name is None:
            raise ValueError(f"Could not extract imputation method and source feature from: {feature.name}")

        imputation_method = imputation_method.replace("imputed", "").strip("_")
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

            constant_value = feature.options.get("constant_value")
            group_by_features = feature.options.get("group_by_features")

            cls._check_source_feature_exists(data, source_feature)

            # Validate group by features if provided
            if group_by_features:
                for group_feature in group_by_features:
                    cls._check_source_feature_exists(data, group_feature)

            # Validate constant value is provided for constant imputation
            if imputation_method == "constant" and constant_value is None:
                raise ValueError("Constant value must be provided for constant imputation method")

            # Apply the appropriate imputation function
            result = cls._perform_imputation(data, imputation_method, source_feature, constant_value, group_by_features)

            # Add the result to the data
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
    def _perform_imputation(
        cls,
        data: Any,
        imputation_method: str,
        mloda_source_feature: str,
        constant_value: Optional[Any] = None,
        group_by_features: Optional[List[str]] = None,
    ) -> Any:
        """
        Method to perform the imputation. Should be implemented by subclasses.

        Args:
            data: The input data
            imputation_method: The type of imputation to perform
            mloda_source_feature: The name of the source feature to impute
            constant_value: The constant value to use for imputation (if method is 'constant')
            group_by_features: Optional list of features to group by before imputation

        Returns:
            The result of the imputation
        """
        raise NotImplementedError(f"_perform_imputation not implemented in {cls.__name__}")
