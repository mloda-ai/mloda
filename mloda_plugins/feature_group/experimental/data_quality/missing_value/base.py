"""
Base implementation for missing value imputation feature groups.
"""

from __future__ import annotations

from typing import Any, List, Optional, Set, Type, Union

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


class MissingValueFeatureGroup(AbstractFeatureGroup):
    # Option key for imputation method
    IMPUTATION_METHOD = "imputation_method"
    """
    Base class for all missing value imputation feature groups.

    Missing value feature groups impute missing values in the source feature using
    the specified imputation method.

    ## Feature Naming Convention

    Missing value features follow this naming pattern:
    `{imputation_method}_imputed__{mloda_source_feature}`

    The source feature (mloda_source_feature) is extracted from the feature name and used
    as input for the imputation operation. Note the double underscore before the source feature.

    Examples:
    - `mean_imputed__income`: Impute missing values in income with the mean
    - `median_imputed__age`: Impute missing values in age with the median
    - `constant_imputed__category`: Impute missing values in category with a constant value

    ## Supported Imputation Methods

    - `mean`: Impute with the mean of non-missing values
    - `median`: Impute with the median of non-missing values
    - `mode`: Impute with the most frequent value
    - `constant`: Impute with a specified constant value
    - `ffill`: Forward fill (use the last valid value)
    - `bfill`: Backward fill (use the next valid value)

    ## Requirements
    - The input data must contain the source feature to be imputed
    - For group-based imputation, the grouping features must also be present
    """

    # Define supported imputation methods
    IMPUTATION_METHODS = {
        "mean": "Impute with the mean of non-missing values",
        "median": "Impute with the median of non-missing values",
        "mode": "Impute with the most frequent value",
        "constant": "Impute with a specified constant value",
        "ffill": "Forward fill (use the last valid value)",
        "bfill": "Backward fill (use the next valid value)",
    }

    # Define the prefix pattern for this feature group
    PREFIX_PATTERN = r"^([\w]+)_imputed__"

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """Extract source feature from the imputed feature name."""
        mloda_source_feature = FeatureChainParser.extract_source_feature(feature_name.name, self.PREFIX_PATTERN)
        return {Feature(mloda_source_feature)}

    @classmethod
    def get_imputation_method(cls, feature_name: str) -> str:
        """Extract the imputation method from the feature name."""
        imputation_method = FeatureChainParser.get_prefix_part(feature_name, cls.PREFIX_PATTERN)
        if imputation_method is None:
            raise ValueError(f"Invalid missing value feature name format: {feature_name}")

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
        if isinstance(feature_name, FeatureName):
            feature_name = feature_name.name

        try:
            # First validate that this is a valid feature name for this feature group
            if not FeatureChainParser.validate_feature_name(feature_name, cls.PREFIX_PATTERN):
                return False

            # Then check if the imputation method is supported
            cls.get_imputation_method(feature_name)
            return True
        except ValueError:
            return False

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """
        Perform missing value imputation.

        Processes all requested features, determining the imputation method
        and source feature from each feature name.

        Adds the imputed results directly to the input data structure.
        """
        # Get constant value and group by features from options if available
        constant_value = None
        group_by_features = None

        if features.options:
            constant_value = features.options.get("constant_value")
            group_by_features = features.options.get("group_by_features")

        # Process each requested feature
        for feature_name in features.get_all_names():
            imputation_method = cls.get_imputation_method(feature_name)
            source_feature = FeatureChainParser.extract_source_feature(feature_name, cls.PREFIX_PATTERN)

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
            data = cls._add_result_to_data(data, feature_name, result)

        # Return the modified data
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

    @classmethod
    def _raise_unsupported_imputation_method(cls, imputation_method: str) -> bool:
        """
        Raise an error for unsupported imputation method.
        """
        raise ValueError(
            f"Unsupported imputation method: {imputation_method}. "
            f"Supported methods: {list(cls.IMPUTATION_METHODS.keys())}"
        )

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
                cls.IMPUTATION_METHOD,
                DefaultOptionKeys.mloda_source_feature,
            ],
            feature_name_template="{imputation_method}_imputed__{mloda_source_feature}",
            validation_rules={
                cls.IMPUTATION_METHOD: lambda x: x in cls.IMPUTATION_METHODS
                or cls._raise_unsupported_imputation_method(x),
            },
        )
