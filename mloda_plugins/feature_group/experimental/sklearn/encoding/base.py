"""
Base implementation for scikit-learn encoding feature groups.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, Optional, Set, Type, Union

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
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
    over categorical data preprocessing, using scikit-learn's encoding implementations. Supports
    various encoding strategies for converting categorical variables to numerical representations.

    ## Supported Operations

    - `onehot`: OneHotEncoder - creates binary columns for each category
    - `label`: LabelEncoder - converts categories to integer labels
    - `ordinal`: OrdinalEncoder - converts categories to ordinal integers

    Encoding features follow this naming pattern:
    `{in_features}__{encoder_type}_encoded`

    ## Feature Creation Methods

    ### 1. String-Based Creation

    Encoding features follow the naming pattern: `{source_feature}__{encoder_type}_encoded`

    Examples:
    ```python
    features = [
        "category__onehot_encoded",     # OneHot encode the category feature
        "status__label_encoded",        # Label encode the status feature
        "priority__ordinal_encoded"     # Ordinal encode the priority feature
    ]
    ```

    ### 2. Configuration-Based Creation

    Uses Options with proper group/context parameter separation:

    ```python
    from mloda_core.abstract_plugins.components.feature import Feature
    from mloda_core.abstract_plugins.components.options import Options
    from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

    feature = Feature(
        name="placeholder",
        options=Options(
            context={
                EncodingFeatureGroup.ENCODER_TYPE: "onehot",
                DefaultOptionKeys.in_features: "category",
            }
        )
    )
    ```

    ## Usage Examples

    ### String-Based Creation

    ```python
    from mloda_core.abstract_plugins.components.feature import Feature

    # OneHot encoding - creates multiple binary columns
    feature = Feature(name="product_category__onehot_encoded")
    # Results in: product_category__onehot_encoded~cat1,
    #             product_category__onehot_encoded~cat2, ...

    # Label encoding - converts to single integer column
    feature = Feature(name="customer_segment__label_encoded")

    # Ordinal encoding - assigns ordered integer values
    feature = Feature(name="education_level__ordinal_encoded")
    ```

    ### Configuration-Based Creation

    ```python
    from mloda_core.abstract_plugins.components.feature import Feature
    from mloda_core.abstract_plugins.components.options import Options

    # OneHot encoding using configuration
    feature = Feature(
        name="placeholder",
        options=Options(
            context={
                EncodingFeatureGroup.ENCODER_TYPE: "onehot",
                DefaultOptionKeys.in_features: "department",
            }
        )
    )

    # Label encoding with configuration
    feature = Feature(
        name="placeholder",
        options=Options(
            context={
                EncodingFeatureGroup.ENCODER_TYPE: "label",
                DefaultOptionKeys.in_features: "risk_level",
            }
        )
    )
    ```

    ### Multiple Result Columns

    For encoders that produce multiple output columns (like OneHotEncoder), the feature group
    uses mloda's multiple result columns pattern with the `~` separator:

    ```python
    # OneHot encoding of 'category' with 3 unique values produces:
    # - category__onehot_encoded~value1
    # - category__onehot_encoded~value2
    # - category__onehot_encoded~value3
    ```

    ## Parameter Classification

    ### Context Parameters (Default)
    These parameters don't affect Feature Group resolution/splitting:
    - `encoder_type`: Type of encoder to use (onehot, label, or ordinal)
    - `in_features`: Source feature to encode

    ### Group Parameters
    Currently none for EncodingFeatureGroup. Parameters that affect Feature Group
    resolution/splitting would be placed here.

    ## Requirements

    - Input data must contain the source feature to be encoded
    - Source feature should contain categorical data
    - Scikit-learn library must be installed
    - For OneHotEncoder, all categories must be known during fitting

    ## Additional Notes

    - Encoding models are persisted as artifacts for consistent train/test encoding
    - OneHotEncoder handles unknown categories during prediction
    - LabelEncoder assigns integers in alphabetical order by default
    - OrdinalEncoder preserves ordinal relationships if categories are properly ordered
    """

    # Option keys for encoding configuration
    ENCODER_TYPE = "encoder_type"

    # Supported encoder types
    SUPPORTED_ENCODERS = {
        "onehot": "OneHotEncoder",
        "label": "LabelEncoder",
        "ordinal": "OrdinalEncoder",
    }

    # Define patterns for parsing
    PATTERN = "__"
    SUFFIX_PATTERN = r".*__(onehot|label|ordinal)_encoded(~\d+)?$"

    # Property mapping for new configuration-based approach
    PROPERTY_MAPPING = {
        ENCODER_TYPE: {
            **SUPPORTED_ENCODERS,  # All supported encoder types as valid options
            DefaultOptionKeys.mloda_context: True,  # Context parameter
            DefaultOptionKeys.mloda_strict_validation: True,  # Enable strict validation
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature to encode",
            DefaultOptionKeys.mloda_context: True,  # Context parameter
            DefaultOptionKeys.mloda_strict_validation: False,  # Flexible validation
        },
    }

    @staticmethod
    def artifact() -> Type[BaseArtifact] | None:
        """Return the artifact class for sklearn encoder persistence."""
        return SklearnArtifact

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """Extract source feature from either configuration-based options or string parsing."""

        # Try string-based parsing first
        _, source_feature = FeatureChainParser.parse_feature_name(feature_name, [self.SUFFIX_PATTERN])
        if source_feature is not None:
            # Remove ~suffix if present (for OneHot column patterns like category~1)
            base_feature = self.get_column_base_feature(source_feature)
            return {Feature(base_feature)}

        # Fall back to configuration-based approach
        source_features = options.get_in_features()
        if len(source_features) != 1:
            raise ValueError(
                f"Expected exactly one source feature, but found {len(source_features)}: {source_features}"
            )
        return set(source_features)

    @classmethod
    def get_encoder_type(cls, feature_name: str) -> str:
        """Extract the encoder type from the feature name."""
        encoder_type, _ = FeatureChainParser.parse_feature_name(feature_name, [cls.SUFFIX_PATTERN])
        if encoder_type is None:
            raise ValueError(f"Invalid encoding feature name format: {feature_name}")

        # Remove the "_encoded" suffix to get just the encoder type
        encoder_type = encoder_type.replace("_encoded", "").strip("_")
        if encoder_type not in cls.SUPPORTED_ENCODERS:
            raise ValueError(
                f"Unsupported encoder type: {encoder_type}. Supported types: {', '.join(cls.SUPPORTED_ENCODERS.keys())}"
            )

        return encoder_type

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Optional[Any] = None,
    ) -> bool:
        """Check if feature name matches the expected pattern using unified parser."""
        # Use the unified parser with property mapping for full configuration support
        return FeatureChainParser.match_configuration_feature_chain_parser(
            feature_name,
            options,
            property_mapping=cls.PROPERTY_MAPPING,
            prefix_patterns=[cls.SUFFIX_PATTERN],
        )

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """
        Apply scikit-learn encoders to features.

        Processes all requested features, determining the encoder type
        and source feature from either string parsing or configuration-based options.

        Adds the encoding results directly to the input data structure.
        """
        # Process each requested feature
        for feature in features.features:
            encoder_type, source_feature = cls._extract_encoder_type_and_source_feature(feature)

            # Remove ~suffix if present (for OneHot column patterns like category~1)
            base_source_feature = cls.get_column_base_feature(source_feature)

            # Check that source feature exists
            cls._check_source_feature_exists(data, base_source_feature)

            # Create unique artifact key that includes encoder type and source feature
            # This ensures different encoders on different features get separate artifacts
            artifact_key = f"{base_source_feature}__{encoder_type}_encoded"

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

            # Apply the fitted encoder to get results
            result = cls._apply_encoder(data, base_source_feature, fitted_encoder)

            # Add result to data (handling multiple columns for OneHotEncoder)
            data = cls._add_result_to_data(data, feature.get_name(), result, encoder_type)

        return data

    @classmethod
    def _extract_encoder_type_and_source_feature(cls, feature: Feature) -> tuple[str, str]:
        """
        Extract encoder type and source feature name from a feature.

        Tries string-based parsing first, falls back to configuration-based approach.

        Args:
            feature: The feature to extract parameters from

        Returns:
            Tuple of (encoder_type, source_feature_name)

        Raises:
            ValueError: If parameters cannot be extracted
        """
        encoder_type = None
        source_feature_name: str | None = None

        # Try string-based parsing first
        feature_name_str = feature.name.name if hasattr(feature.name, "name") else str(feature.name)

        if FeatureChainParser.is_chained_feature(feature_name_str):
            encoder_type = cls.get_encoder_type(feature_name_str)
            source_feature_name = FeatureChainParser.extract_source_feature(feature_name_str, cls.SUFFIX_PATTERN)
            return encoder_type, source_feature_name

        # Fall back to configuration-based approach
        source_features = feature.options.get_in_features()
        source_feature = next(iter(source_features))
        source_feature_name = source_feature.get_name()

        encoder_type = feature.options.get(cls.ENCODER_TYPE)

        if encoder_type is None or source_feature_name is None:
            raise ValueError(f"Could not extract encoder type and source feature from: {feature.name}")

        if encoder_type not in cls.SUPPORTED_ENCODERS:
            raise ValueError(
                f"Unsupported encoder type: {encoder_type}. Supported types: {', '.join(cls.SUPPORTED_ENCODERS.keys())}"
            )

        return encoder_type, source_feature_name

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
