"""
Base implementation for text cleaning feature groups.
"""

from __future__ import annotations

from typing import Any, Optional, Set, Type, Union

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda_core.abstract_plugins.components.feature_chainer.feature_chainer_parser_configuration import (
    FeatureChainParserConfiguration,
)
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

    # Define prefix pattern
    PREFIX_PATTERN = r"^cleaned_text__"

    """
    Base class for all text cleaning feature groups.

    Text cleaning feature groups provide operations for preprocessing and cleaning text data.
    They allow you to apply multiple cleaning operations in sequence to prepare text for
    further analysis or machine learning tasks.

    ## Feature Naming Convention

    Text cleaning features follow this naming pattern:
    `cleaned_text__{mloda_source_feature}`

    The source feature (mloda_source_feature) is extracted from the feature name and used
    as input for the text cleaning operations. Note the double underscore before the source feature.

    Examples:
    - `cleaned_text__review`: Apply text cleaning operations to the "review" feature
    - `cleaned_text__description`: Apply text cleaning operations to the "description" feature

    ## Configuration-Based Creation

    TextCleaningFeatureGroup supports configuration-based creation through the
    FeatureChainParserConfiguration mechanism. This allows features to be created
    from options rather than explicit feature names.

    To create a text cleaning feature using configuration:

    ```python
    feature = Feature(
        "PlaceHolder",  # Placeholder name, will be replaced
        Options({
            TextCleaningFeatureGroup.CLEANING_OPERATIONS: ("normalize", "remove_stopwords", "remove_punctuation"),
            DefaultOptionKeys.mloda_source_feature: "review"
        })
    )

    # The Engine will automatically parse this into a feature with name "cleaned_text__review"
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
        """Extract source feature from the text cleaning feature name."""
        source_feature = FeatureChainParser.extract_source_feature(feature_name.name, self.PREFIX_PATTERN)
        return {Feature(source_feature)}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Optional[Any] = None,
    ) -> bool:
        """Check if feature name matches the expected pattern for text cleaning features."""
        if isinstance(feature_name, FeatureName):
            feature_name = feature_name.name

        try:
            # Validate that this is a valid feature name for this feature group
            if not FeatureChainParser.validate_feature_name(feature_name, cls.PREFIX_PATTERN):
                return False

            return True
        except ValueError:
            return False

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
        print(f"calculate_feature called in {cls.__name__}")
        print(f"Features: {[f.name.name for f in features.features]}")

        # Process each requested feature
        for feature in features.features:
            feature_name = feature.name.name
            print(f"Processing feature: {feature_name}")
            print(f"Feature options: {feature.options.data}")

            # Extract source feature
            source_feature = FeatureChainParser.extract_source_feature(feature_name, cls.PREFIX_PATTERN)
            print(f"Source feature: {source_feature}")

            # Check if source feature exists
            cls._check_source_feature_exists(data, source_feature)

            # Get operations from options
            operations = feature.options.get(cls.CLEANING_OPERATIONS) or ()
            print(f"Operations: {operations}")

            # Validate operations
            for operation in operations:
                if operation not in cls.SUPPORTED_OPERATIONS:
                    raise ValueError(
                        f"Unsupported cleaning operation: {operation}. "
                        f"Supported operations: {', '.join(cls.SUPPORTED_OPERATIONS.keys())}"
                    )

            # Apply operations in sequence
            result = cls._get_source_text(data, source_feature)
            print(f"Source text: {result.iloc[0] if hasattr(result, 'iloc') else result}")

            for operation in operations:
                print(f"Applying operation: {operation}")
                result = cls._apply_operation(data, result, operation)
                print(f"Result after {operation}: {result.iloc[0] if hasattr(result, 'iloc') else result}")

            # Add result to data
            data = cls._add_result_to_data(data, feature_name, result)

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

    @classmethod
    def configurable_feature_chain_parser(cls) -> Optional[Type[FeatureChainParserConfiguration]]:
        """
        Returns the FeatureChainParserConfiguration class for this feature group.

        This method allows the Engine to automatically create features with the correct
        naming convention based on configuration options, rather than requiring explicit
        feature names.

        Returns:
            The TextCleaningFeatureChainParserConfiguration class
        """
        return TextCleaningFeatureChainParserConfiguration


class TextCleaningFeatureChainParserConfiguration(FeatureChainParserConfiguration):
    """
    Feature chain parser configuration for TextCleaningFeatureGroup.

    This class provides the configuration for parsing TextCleaningFeatureGroup features
    from options. It defines the keys used for parsing and implements the parse_from_options
    method to create feature names in the format "cleaned_text__{source_feature}".

    This configuration is used by the Engine to automatically parse features with the
    appropriate options into TextCleaningFeatureGroup features.
    """

    @classmethod
    def parse_keys(cls) -> Set[str]:
        """
        Returns the keys that are used to parse the feature group.

        Returns:
            A set containing only the source feature key, as the cleaning operations
            should be preserved for use in calculate_feature
        """
        return {
            DefaultOptionKeys.mloda_source_feature,
        }

    @classmethod
    def parse_from_options(cls, options: Options) -> Optional[str]:
        """
        Parse a TextCleaningFeatureGroup feature from options.

        Args:
            options: A dictionary containing:
                - CLEANING_OPERATIONS: List of cleaning operations to apply
                - DefaultOptionKeys.mloda_source_feature: The source feature name

        Returns:
            A feature name string in the format "cleaned_text__{mloda_source_feature}"

        Raises:
            ValueError: If any of the specified operations are not supported
        """
        # Extract required options
        operations = options.get(TextCleaningFeatureGroup.CLEANING_OPERATIONS)
        source_feature = options.get(DefaultOptionKeys.mloda_source_feature)

        # Validate options
        if not operations:
            return None
        if not source_feature:
            return None

        # Validate operations
        for operation in operations:
            if operation not in TextCleaningFeatureGroup.SUPPORTED_OPERATIONS:
                raise ValueError(
                    f"Unsupported cleaning operation: {operation}. "
                    f"Supported operations: {', '.join(TextCleaningFeatureGroup.SUPPORTED_OPERATIONS.keys())}"
                )

        # Build and return the feature name
        return f"cleaned_text__{source_feature}"
