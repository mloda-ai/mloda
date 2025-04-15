"""
Base implementation for missing value imputation feature groups.
"""

from __future__ import annotations

import re
from typing import Any, Optional, Set, Union

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.options import Options


class BaseMissingValueFeatureGroup(AbstractFeatureGroup):
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

    FEATURE_NAME_PATTERN = r"^([\w]+)_imputed__([\w]+)$"

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """Extract source feature from the imputed feature name."""
        mloda_source_feature = self.mloda_source_feature(feature_name.name)
        return {Feature(mloda_source_feature)}

    @classmethod
    def parse_feature_name(cls, feature_name: str) -> tuple[str, str]:
        """
        Parse the feature name into its components.

        Args:
            feature_name: The feature name to parse

        Returns:
            A tuple containing (imputation_method, mloda_source_feature)

        Raises:
            ValueError: If the feature name does not match the expected pattern
        """
        match = re.match(cls.FEATURE_NAME_PATTERN, feature_name)
        if not match:
            raise ValueError(
                f"Invalid missing value feature name format: {feature_name}. "
                f"Expected format: {{imputation_method}}_imputed__{{mloda_source_feature}}"
            )

        imputation_method, mloda_source_feature = match.groups()

        # Validate imputation method
        if imputation_method not in cls.IMPUTATION_METHODS:
            raise ValueError(
                f"Unsupported imputation method: {imputation_method}. "
                f"Supported methods: {', '.join(cls.IMPUTATION_METHODS.keys())}"
            )

        return imputation_method, mloda_source_feature

    @classmethod
    def get_imputation_method(cls, feature_name: str) -> str:
        """Extract the imputation method from the feature name."""
        return cls.parse_feature_name(feature_name)[0]

    @classmethod
    def mloda_source_feature(cls, feature_name: str) -> str:
        """Extract the source feature name from the feature name."""
        return cls.parse_feature_name(feature_name)[1]

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
            # Try to parse the feature name - if it succeeds, it matches our pattern
            cls.parse_feature_name(feature_name)
            return True
        except ValueError:
            return False

    @classmethod
    def _perform_imputation(
        cls,
        data: Any,
        imputation_method: str,
        mloda_source_feature: str,
        constant_value: Optional[Any] = None,
        group_by_features: Optional[list[str]] = None,
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
