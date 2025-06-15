"""
Base implementation for scikit-learn scaling feature groups.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional, Set, Type, Union

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_chainer.feature_chainer_parser_configuration import (
    FeatureChainParserConfiguration,
    create_configurable_parser,
)
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda_core.abstract_plugins.components.base_artifact import BaseArtifact
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys
from mloda_plugins.feature_group.experimental.sklearn.sklearn_artifact import SklearnArtifact


class ScalingFeatureGroup(AbstractFeatureGroup):
    """
    Base class for scikit-learn scaling feature groups.

    The ScalingFeatureGroup provides individual scaling transformations for granular control
    over data preprocessing, demonstrating mloda's fine-grained transformation capabilities.

    ## Feature Naming Convention

    Scaling features follow this naming pattern:
    `{scaler_type}_scaled__{mloda_source_feature}`

    The scaler type determines which sklearn scaler to use, and the source feature
    is extracted from the feature name and used as input for the scaler.

    Examples:
    - `standard_scaled__income`: Apply StandardScaler to income feature
    - `minmax_scaled__age`: Apply MinMaxScaler to age feature
    - `robust_scaled__outlier_prone_feature`: Apply RobustScaler to outlier_prone_feature
    - `normalizer_scaled__feature_vector`: Apply Normalizer to feature_vector

    ## Supported Scalers

    - **standard**: StandardScaler (mean=0, std=1)
    - **minmax**: MinMaxScaler (scale to [0,1] range)
    - **robust**: RobustScaler (uses median and IQR, robust to outliers)
    - **normalizer**: Normalizer (scale individual samples to unit norm)

    ## Configuration-Based Creation

    ScalingFeatureGroup supports configuration-based creation through the
    FeatureChainParserConfiguration mechanism. This allows features to be created
    from options rather than explicit feature names.

    To create a scaling feature using configuration:

    ```python
    feature = Feature(
        "PlaceHolder",  # Placeholder name, will be replaced
        Options({
            ScalingFeatureGroup.SCALER_TYPE: "standard",
            DefaultOptionKeys.mloda_source_feature: "income"
        })
    )

    # The Engine will automatically parse this into a feature with name
    # "standard_scaled__income"
    ```
    """

    # Option keys for scaling configuration
    SCALER_TYPE = "scaler_type"

    # Supported scaler types
    SUPPORTED_SCALERS = {
        "standard": "StandardScaler",
        "minmax": "MinMaxScaler",
        "robust": "RobustScaler",
        "normalizer": "Normalizer",
    }

    @staticmethod
    def artifact() -> Type[BaseArtifact] | None:
        """Return the artifact class for sklearn scaler persistence."""
        return SklearnArtifact

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """Extract source features from the scaling feature name."""
        mloda_source_feature = FeatureChainParser.extract_source_feature(feature_name.name, self.PREFIX_PATTERN)
        return {Feature(mloda_source_feature)}

    # Define the prefix pattern for this feature group
    PREFIX_PATTERN = r"^(standard|minmax|robust|normalizer)_scaled__"

    @classmethod
    def get_scaler_type(cls, feature_name: str) -> str:
        """Extract the scaler type from the feature name."""
        prefix_part = FeatureChainParser.get_prefix_part(feature_name, cls.PREFIX_PATTERN)
        if prefix_part is None:
            raise ValueError(f"Invalid scaling feature name format: {feature_name}")
        return prefix_part

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Optional[Any] = None,
    ) -> bool:
        """Check if feature name matches the expected pattern."""
        if isinstance(feature_name, FeatureName):
            feature_name = feature_name.name

        try:
            # Validate that this is a valid feature name for this feature group
            if not FeatureChainParser.validate_feature_name(feature_name, cls.PREFIX_PATTERN):
                return False

            # Extract scaler type to ensure it's supported
            scaler_type = cls.get_scaler_type(feature_name)
            return scaler_type in cls.SUPPORTED_SCALERS
        except ValueError:
            return False

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """
        Apply scikit-learn scalers to features.

        Processes all requested features, determining the scaler type
        and source feature from each feature name.

        Adds the scaling results directly to the input data structure.
        """
        # Process each requested feature
        for feature_name in features.get_all_names():
            scaler_type = cls.get_scaler_type(feature_name)
            source_feature = FeatureChainParser.extract_source_feature(feature_name, cls.PREFIX_PATTERN)

            # Check that source feature exists
            cls._check_source_feature_exists(data, source_feature)

            # Create unique artifact key for this scaler
            artifact_key = f"{scaler_type}_scaled__{source_feature}"

            # Try to load existing fitted scaler from artifact using helper method
            fitted_scaler = None
            artifact = SklearnArtifact.load_sklearn_artifact(features, artifact_key)
            if artifact:
                fitted_scaler = artifact["fitted_transformer"]
                cls._scaler_matches_type(fitted_scaler, scaler_type)

            # If no fitted scaler available, create and fit new one
            if fitted_scaler is None:
                fitted_scaler = cls._create_and_fit_scaler(data, source_feature, scaler_type)

                # Save the fitted scaler as artifact using helper method
                artifact_data = {
                    "fitted_transformer": fitted_scaler,
                    "feature_name": source_feature,
                    "scaler_type": scaler_type,
                    "training_timestamp": datetime.datetime.now().isoformat(),
                }
                SklearnArtifact.save_sklearn_artifact(features, artifact_key, artifact_data)

            # Apply the fitted scaler to get results
            result = cls._apply_scaler(data, source_feature, fitted_scaler)

            # Add result to data
            data = cls._add_result_to_data(data, feature_name, result)

        return data

    @classmethod
    def _import_sklearn_components(cls) -> Dict[str, Any]:
        """
        Import sklearn components with fallback logic for different versions.

        Returns:
            Dictionary containing imported sklearn components

        Raises:
            ImportError: If sklearn is not available
        """
        components = {}

        try:
            from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer

            components.update(
                {
                    "StandardScaler": StandardScaler,
                    "MinMaxScaler": MinMaxScaler,
                    "RobustScaler": RobustScaler,
                    "Normalizer": Normalizer,
                }
            )

        except ImportError:
            raise ImportError(
                "scikit-learn is required for ScalingFeatureGroup. Install with: pip install scikit-learn"
            )

        return components

    @classmethod
    def _create_scaler_instance(cls, scaler_type: str) -> Any:
        """
        Create a scaler instance based on the scaler type.

        Args:
            scaler_type: The type of scaler to create

        Returns:
            Scaler instance

        Raises:
            ValueError: If scaler type is not supported
            ImportError: If sklearn is not available
        """
        if scaler_type not in cls.SUPPORTED_SCALERS:
            raise ValueError(
                f"Unsupported scaler type: {scaler_type}. Supported types: {list(cls.SUPPORTED_SCALERS.keys())}"
            )

        sklearn_components = cls._import_sklearn_components()
        scaler_class_name = cls.SUPPORTED_SCALERS[scaler_type]
        scaler_class = sklearn_components[scaler_class_name]

        return scaler_class()

    @classmethod
    def _scaler_matches_type(cls, fitted_scaler: Any, scaler_type: str) -> bool:
        """
        Check if a fitted scaler matches the expected type.

        Args:
            fitted_scaler: The fitted scaler
            scaler_type: The expected scaler type

        Returns:
            True if the scaler matches the type

        Raises:
            ValueError: If scaler type mismatch is detected
        """
        try:
            expected_class_name = cls.SUPPORTED_SCALERS.get(scaler_type)
            if expected_class_name is None:
                raise ValueError(f"Unsupported scaler type: {scaler_type}")

            actual_class_name: str = fitted_scaler.__class__.__name__
            if actual_class_name != expected_class_name:
                raise ValueError(
                    f"Artifact scaler type mismatch: expected {scaler_type} "
                    f"({expected_class_name}), but loaded artifact contains {actual_class_name}"
                )
            return True
        except Exception as e:
            if isinstance(e, ValueError):
                raise  # Re-raise ValueError as-is
            # For other exceptions, wrap in ValueError
            raise ValueError(f"Error validating scaler type: {str(e)}")

    @classmethod
    def _create_and_fit_scaler(cls, data: Any, source_feature: str, scaler_type: str) -> Any:
        """
        Create and fit a new scaler.

        Args:
            data: The input data
            source_feature: Name of the source feature
            scaler_type: Type of scaler to create

        Returns:
            Fitted scaler
        """
        # Create scaler instance
        scaler = cls._create_scaler_instance(scaler_type)

        # Extract training data
        X_train = cls._extract_training_data(data, source_feature)

        # Fit the scaler
        scaler.fit(X_train)

        return scaler

    @classmethod
    def _extract_training_data(cls, data: Any, source_feature: str) -> Any:
        """
        Extract training data for the specified feature.

        Args:
            data: The input data
            source_feature: Name of the source feature

        Returns:
            Training data for the feature
        """
        raise NotImplementedError(f"_extract_training_data not implemented in {cls.__name__}")

    @classmethod
    def _apply_scaler(cls, data: Any, source_feature: str, fitted_scaler: Any) -> Any:
        """
        Apply the fitted scaler to the data.

        Args:
            data: The input data
            source_feature: Name of the source feature
            fitted_scaler: The fitted scaler

        Returns:
            Scaled data
        """
        raise NotImplementedError(f"_apply_scaler not implemented in {cls.__name__}")

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
                cls.SCALER_TYPE,
                DefaultOptionKeys.mloda_source_feature,
            ],
            feature_name_template="{scaler_type}_scaled__{mloda_source_feature}",
            validation_rules={
                cls.SCALER_TYPE: lambda x: isinstance(x, str) and x in cls.SUPPORTED_SCALERS,
            },
        )
