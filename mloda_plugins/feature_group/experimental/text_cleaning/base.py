"""
Base implementation for text cleaning feature groups.
"""

from __future__ import annotations

from typing import Any, Optional, Set, Union

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


class TextCleaningFeatureGroup(AbstractFeatureGroup):
    # Option key for the list of operations
    CLEANING_OPERATIONS = "cleaning_operations"

    # Define supported cleaning operations with their descriptions
    SUPPORTED_OPERATIONS = {
        "normalize": "Convert text to lowercase and remove accents",
        "remove_stopwords": "Remove common stopwords",
        "remove_punctuation": "Remove punctuation marks",
        "remove_special_chars": "Remove special characters",
        "normalize_whitespace": "Normalize whitespace",
        "remove_urls": "Remove URLs and email addresses",
    }

    # Define prefix pattern and pattern
    PATTERN = "__"
    PREFIX_PATTERN = r".*__cleaned_text$"

    # Property mapping for configuration-based features
    PROPERTY_MAPPING = {
        CLEANING_OPERATIONS: {
            **SUPPORTED_OPERATIONS,  # All supported operations as valid options
            DefaultOptionKeys.mloda_context: True,  # Mark as context parameter
            DefaultOptionKeys.mloda_strict_validation: True,  # Enable strict validation
            DefaultOptionKeys.mloda_validation_function: lambda operations: (
                # Handle both actual tuples/lists and string representations
                (
                    isinstance(operations, (tuple, list))
                    and all(op in TextCleaningFeatureGroup.SUPPORTED_OPERATIONS for op in operations)
                )
                or (
                    isinstance(operations, str)
                    and operations.startswith("(")
                    and operations.endswith(")")
                    and all(
                        op.strip("'\" ,") in TextCleaningFeatureGroup.SUPPORTED_OPERATIONS
                        for op in operations.strip("()").split(",")
                        if op.strip("'\" ,")
                    )
                )
            ),
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature to apply text cleaning operations to",
            DefaultOptionKeys.mloda_context: True,
        },
    }

    """
    Base class for all text cleaning feature groups.

    Text cleaning feature groups provide operations for preprocessing and cleaning text data.
    They allow you to apply multiple cleaning operations in sequence to prepare text for
    further analysis or machine learning tasks.

    ## Feature Naming Convention

    Text cleaning features follow this naming pattern:
    `{in_features}__cleaned_text`

    The source feature comes first, followed by the cleaning operation.
    Note the double underscore separating the source feature from the operation.

    Examples:
    - `review__cleaned_text`: Apply text cleaning operations to the "review" feature
    - `description__cleaned_text`: Apply text cleaning operations to the "description" feature

    ## Configuration-Based Creation

    TextCleaningFeatureGroup supports configuration-based. This allows features to be created
    from options rather than explicit feature names.

    To create a text cleaning feature using configuration:

    ```python
    feature = Feature(
        "PlaceHolder",  # Placeholder name, will be replaced
        Options({
            TextCleaningFeatureGroup.CLEANING_OPERATIONS: ("normalize", "remove_stopwords", "remove_punctuation"),
            DefaultOptionKeys.in_features: "review"
        })
    )

    # The Engine will automatically parse this into a feature with name "review__cleaned_text"
    ```

    ## Supported Cleaning Operations

    - `normalize`: Convert text to lowercase and remove accents
    - `remove_stopwords`: Remove common stopwords
    - `remove_punctuation`: Remove punctuation marks
    - `remove_special_chars`: Remove special characters
    - `normalize_whitespace`: Normalize whitespace
    - `remove_urls`: Remove URLs and email addresses

    ## Requirements
    - The input data must contain the source feature to be used for text cleaning
    - The source feature must contain text data
    """

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """Extract source feature from either configuration-based options or string parsing."""

        source_feature: str | None = None

        # Try string-based parsing first
        _, source_feature = FeatureChainParser.parse_feature_name(feature_name, [self.PREFIX_PATTERN])
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
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Optional[Any] = None,
    ) -> bool:
        """Check if feature name matches the expected pattern for text cleaning features."""

        # Use the unified parser with property mapping for full configuration support
        return FeatureChainParser.match_configuration_feature_chain_parser(
            feature_name,
            options,
            property_mapping=cls.PROPERTY_MAPPING,
            prefix_patterns=[cls.PREFIX_PATTERN],
        )

    @classmethod
    def _extract_operations_and_source_feature(cls, feature: Feature) -> tuple[tuple[Any, Any], str]:
        """
        Extract cleaning operations and source feature name from a feature.

        Tries string-based parsing first, falls back to configuration-based approach.

        Args:
            feature: The feature to extract parameters from

        Returns:
            Tuple of (operations_tuple, source_feature_name)

        Raises:
            ValueError: If parameters cannot be extracted
        """
        operations = None
        source_feature_name: str | None = None

        # Try string-based parsing first
        feature_name_str = feature.name.name if hasattr(feature.name, "name") else str(feature.name)

        if FeatureChainParser.is_chained_feature(feature_name_str):
            _, source_feature_name = FeatureChainParser.parse_feature_name(feature_name_str, [cls.PREFIX_PATTERN])
            # For string-based features, get operations from options
            operations = feature.options.get(cls.CLEANING_OPERATIONS) or ()
            if source_feature_name is None:
                raise ValueError(f"Could not extract source feature from string-based feature: {feature.name}")
            return operations, source_feature_name  # type: ignore

        # Fall back to configuration-based approach
        source_features = feature.options.get_in_features()
        source_feature = next(iter(source_features))
        source_feature_name = source_feature.get_name()

        operations = feature.options.get(cls.CLEANING_OPERATIONS)

        if operations is None or source_feature_name is None:
            raise ValueError(f"Could not extract cleaning operations and source feature from: {feature.name}")

        return operations, source_feature_name

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """
        Perform text cleaning operations.

        Processes all requested features, applying the specified cleaning operations
        to the source features.

        Args:
            data: The input data
            features: The feature set containing the features to process

        Returns:
            The data with the cleaned text features added
        """

        # Process each requested feature
        for feature in features.features:
            operations, source_feature = cls._extract_operations_and_source_feature(feature)

            # Check if source feature exists
            cls._check_source_feature_exists(data, source_feature)

            # Validate operations
            for operation in operations:
                if operation not in cls.SUPPORTED_OPERATIONS:
                    raise ValueError(
                        f"Unsupported cleaning operation: {operation}. "
                        f"Supported operations: {', '.join(cls.SUPPORTED_OPERATIONS.keys())}"
                    )

            # Apply operations in sequence
            result = cls._get_source_text(data, source_feature)

            for operation in operations:
                result = cls._apply_operation(data, result, operation)

            # Add result to data
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
    def _get_source_text(cls, data: Any, feature_name: str) -> Any:
        """
        Get the source text from the data.

        Args:
            data: The input data
            feature_name: The name of the feature to get

        Returns:
            The source text
        """
        raise NotImplementedError(f"_get_source_text not implemented in {cls.__name__}")

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
    def _apply_operation(cls, data: Any, text: Any, operation: str) -> Any:
        """
        Apply a cleaning operation to the text.

        Args:
            data: The input data (for context)
            text: The text to clean
            operation: The operation to apply

        Returns:
            The cleaned text
        """
        raise NotImplementedError(f"_apply_operation not implemented in {cls.__name__}")
