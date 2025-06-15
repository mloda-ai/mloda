"""
Base implementation for scikit-learn encoding feature groups.
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


class EncodingFeatureGroup(AbstractFeatureGroup):
    """
    Base class for scikit-learn encoding feature groups.

    The EncodingFeatureGroup provides categorical encoding transformations for granular control
    over categorical data preprocessing, demonstrating mloda's fine-grained transformation capabilities.

    ## Feature Naming Convention

    Encoding features follow this naming pattern:
    `{encoder_type}_encoded__{mloda_source_feature}`

    The encoder type determines which sklearn encoder to use, and the source feature
    is extracted from the feature name and used as input for the encoder.

    Examples:
    - `onehot_encoded__category`: Apply OneHotEncoder to category feature
    - `label_encoded__status`: Apply LabelEncoder to status feature
    - `ordinal_encoded__priority`: Apply OrdinalEncoder to priority feature

    ## Supported Encoders

    - **onehot**: OneHotEncoder (creates binary columns for each category)
    - **label**: LabelEncoder (converts categories to integer labels)
    - **ordinal**: OrdinalEncoder (converts categories to ordinal integers)

    ## Multiple Result Columns

    For encoders that produce multiple output columns (like OneHotEncoder), the feature group
    uses mloda's multiple result columns pattern with the `~` separator:

    - `onehot_encoded__category~feature1`
    - `onehot_encoded__category~feature2`
    - `onehot_encoded__category~feature3`

    ## Configuration-Based Creation

    EncodingFeatureGroup supports configuration-based creation through the
    FeatureChainParserConfiguration mechanism. This allows features to be created
    from options rather than explicit feature names.

    To create an encoding feature using configuration:

    ```python
    feature = Feature(
        "PlaceHolder",  # Placeholder name, will be replaced
        Options({
            EncodingFeatureGroup.ENCODER_TYPE: "onehot",
            DefaultOptionKeys.mloda_source_feature: "category"
        })
    )

    # The Engine will automatically parse this into a feature with name
    # "onehot_encoded__category"
    ```
    """

    # Option keys for encoding configuration
    ENCODER_TYPE = "encoder_type"

    # Supported encoder types
    SUPPORTED_ENCODERS = {
        "onehot": "OneHotEncoder",
        "label": "LabelEncoder",
        "ordinal": "OrdinalEncoder",
    }

    @staticmethod
    def artifact() -> Type[BaseArtifact] | None:
        """Return the artifact class for sklearn encoder persistence."""
        return SklearnArtifact

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """Extract source features from the encoding feature name."""
        mloda_source_feature = FeatureChainParser.extract_source_feature(feature_name.name, self.PREFIX_PATTERN)
        # Remove ~suffix if present (for OneHot column patterns like category~1)
        base_feature = mloda_source_feature.split("~")[0]
        return {Feature(base_feature)}

    # Define the prefix pattern for this feature group
    PREFIX_PATTERN = r"^(onehot|label|ordinal)_encoded__"

    @classmethod
    def get_encoder_type(cls, feature_name: str) -> str:
        """Extract the encoder type from the feature name."""
        prefix_part = FeatureChainParser.get_prefix_part(feature_name, cls.PREFIX_PATTERN)
        if prefix_part is None:
            raise ValueError(f"Invalid encoding feature name format: {feature_name}")
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

            # Extract encoder type to ensure it's supported
            encoder_type = cls.get_encoder_type(feature_name)
            return encoder_type in cls.SUPPORTED_ENCODERS
        except ValueError:
            return False

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """
        Apply scikit-learn encoders to features.

        Processes all requested features, determining the encoder type
        and source feature from each feature name.

        Adds the encoding results directly to the input data structure.
        """
        # Process each requested feature
        for feature_name in features.get_all_names():
            encoder_type = cls.get_encoder_type(feature_name)
            source_feature = FeatureChainParser.extract_source_feature(feature_name, cls.PREFIX_PATTERN)

            # Remove ~suffix if present (for OneHot column patterns like category~1)
            base_source_feature = source_feature.split("~")[0]

            # Check that source feature exists
            cls._check_source_feature_exists(data, base_source_feature)

            # Create unique artifact key that includes encoder type and source feature
            # This ensures different encoders on different features get separate artifacts
            artifact_key = f"{encoder_type}_encoded__{base_source_feature}"

            # Try to load existing fitted encoder from artifact using helper method
            fitted_encoder = None
            artifact = SklearnArtifact.load_sklearn_artifact(features, artifact_key)
            if artifact:
                fitted_encoder = artifact["fitted_transformer"]
                cls._encoder_matches_type(fitted_encoder, encoder_type)

            # If no fitted encoder available, create and fit new one
            if fitted_encoder is None:
                fitted_encoder = cls._create_and_fit_encoder(data, base_source_feature, encoder_type)

                # Save the fitted encoder as artifact using helper method
                artifact_data = {
                    "fitted_transformer": fitted_encoder,
                    "feature_name": base_source_feature,
                    "encoder_type": encoder_type,
                    "training_timestamp": datetime.datetime.now().isoformat(),
                }
                SklearnArtifact.save_sklearn_artifact(features, artifact_key, artifact_data)

            # Appl  y the fitted encoder to get results
            result = cls._apply_encoder(data, base_source_feature, fitted_encoder)

            # Add result to data (handling multiple columns for OneHotEncoder)
            data = cls._add_result_to_data(data, feature_name, result, encoder_type)

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
            from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder

            components.update(
                {
                    "OneHotEncoder": OneHotEncoder,
                    "LabelEncoder": LabelEncoder,
                    "OrdinalEncoder": OrdinalEncoder,
                }
            )

        except ImportError:
            raise ImportError(
                "scikit-learn is required for EncodingFeatureGroup. Install with: pip install scikit-learn"
            )

        return components

    @classmethod
    def _create_encoder_instance(cls, encoder_type: str) -> Any:
        """
        Create an encoder instance based on the encoder type.

        Args:
            encoder_type: The type of encoder to create

        Returns:
            Encoder instance

        Raises:
            ValueError: If encoder type is not supported
            ImportError: If sklearn is not available
        """
        if encoder_type not in cls.SUPPORTED_ENCODERS:
            raise ValueError(
                f"Unsupported encoder type: {encoder_type}. Supported types: {list(cls.SUPPORTED_ENCODERS.keys())}"
            )

        sklearn_components = cls._import_sklearn_components()
        encoder_class_name = cls.SUPPORTED_ENCODERS[encoder_type]
        encoder_class = sklearn_components[encoder_class_name]

        # Configure encoder with appropriate parameters
        if encoder_type == "onehot":
            # OneHotEncoder: handle unknown categories gracefully, don't drop first column
            return encoder_class(handle_unknown="ignore", drop=None)
        elif encoder_type == "ordinal":
            # OrdinalEncoder: handle unknown categories gracefully
            return encoder_class(handle_unknown="use_encoded_value", unknown_value=-1)
        else:
            # LabelEncoder: default configuration
            return encoder_class()

    @classmethod
    def _encoder_matches_type(cls, fitted_encoder: Any, encoder_type: str) -> bool:
        """
        Check if a fitted encoder matches the expected type.

        Args:
            fitted_encoder: The fitted encoder
            encoder_type: The expected encoder type

        Returns:
            True if the encoder matches the type

        Raises:
            ValueError: If encoder type mismatch is detected
        """
        try:
            expected_class_name = cls.SUPPORTED_ENCODERS.get(encoder_type)
            if expected_class_name is None:
                raise ValueError(f"Unsupported encoder type: {encoder_type}")

            actual_class_name: str = fitted_encoder.__class__.__name__
            if actual_class_name != expected_class_name:
                raise ValueError(
                    f"Artifact encoder type mismatch: expected {encoder_type} "
                    f"({expected_class_name}), but loaded artifact contains {actual_class_name}"
                )
            return True
        except Exception as e:
            if isinstance(e, ValueError):
                raise  # Re-raise ValueError as-is
            # For other exceptions, wrap in ValueError
            raise ValueError(f"Error validating encoder type: {str(e)}")

    @classmethod
    def _create_and_fit_encoder(cls, data: Any, source_feature: str, encoder_type: str) -> Any:
        """
        Create and fit a new encoder.

        Args:
            data: The input data
            source_feature: Name of the source feature
            encoder_type: Type of encoder to create

        Returns:
            Fitted encoder
        """
        # Create encoder instance
        encoder = cls._create_encoder_instance(encoder_type)

        # Extract training data
        X_train = cls._extract_training_data(data, source_feature)

        # Reshape data based on encoder type
        if encoder_type == "label":
            # LabelEncoder expects 1D array
            if hasattr(X_train, "shape") and len(X_train.shape) > 1:
                X_train = X_train.flatten()
        else:
            # OneHotEncoder and OrdinalEncoder expect 2D array
            if hasattr(X_train, "shape") and len(X_train.shape) == 1:
                X_train = X_train.reshape(-1, 1)

        # Fit the encoder
        encoder.fit(X_train)

        return encoder

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
    def _apply_encoder(cls, data: Any, source_feature: str, fitted_encoder: Any) -> Any:
        """
        Apply the fitted encoder to the data.

        Args:
            data: The input data
            source_feature: Name of the source feature
            fitted_encoder: The fitted encoder

        Returns:
            Encoded data
        """
        raise NotImplementedError(f"_apply_encoder not implemented in {cls.__name__}")

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
    def _add_result_to_data(cls, data: Any, feature_name: str, result: Any, encoder_type: str) -> Any:
        """
        Add the result to the data.

        Args:
            data: The input data
            feature_name: The name of the feature to add
            result: The result to add
            encoder_type: The type of encoder used

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
                cls.ENCODER_TYPE,
                DefaultOptionKeys.mloda_source_feature,
            ],
            feature_name_template="{encoder_type}_encoded__{mloda_source_feature}",
            validation_rules={
                cls.ENCODER_TYPE: lambda x: isinstance(x, str) and x in cls.SUPPORTED_ENCODERS,
            },
        )
