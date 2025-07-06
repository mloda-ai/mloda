"""
Feature chain parser for handling feature name chaining across feature groups.
"""

from __future__ import annotations

import re
from typing import List, Optional, Set, Union, TYPE_CHECKING

from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.options import Options


if TYPE_CHECKING:
    from mloda_core.abstract_plugins.components.feature import Feature
    from mloda_core.abstract_plugins.components.options import Options


class FeatureChainParser:
    """
    Mixin class for parsing feature names with chaining support.

    Feature chaining allows feature groups to be composed, where the output of one
    feature group becomes the input to another. This is reflected in the feature name
    using a double underscore pattern: prefix__mloda_source_feature.

    For example:
    - max_aggr__sum_7_day_window__mean_imputed__price

    Each feature group in the chain extracts its relevant portion and passes the
    rest to the next feature group in the chain.
    """

    @classmethod
    def extract_source_feature(cls, feature_name: str, prefix_pattern: str) -> str:
        """
        Extract the source feature from a feature name based on the prefix pattern.

        Args:
            feature_name: The feature name to parse
            prefix_pattern: Regex pattern for the prefix (e.g., r"^([w]+)_aggr__")

        Returns:
            The source feature part of the name

        Raises:
            ValueError: If the feature name doesn't match the expected pattern
        """
        match = re.match(prefix_pattern, feature_name)
        if not match:
            raise ValueError(f"Invalid feature name format: {feature_name}")

        # Extract the prefix part (everything before the double underscore)
        prefix_end = feature_name.find("__")
        if prefix_end == -1:
            raise ValueError(f"Invalid feature name format: {feature_name}. Missing double underscore separator.")

        # Return everything after the double underscore
        return feature_name[prefix_end + 2 :]

    @classmethod
    def build_feature_name(cls, prefix: str, mloda_source_feature: str) -> str:
        """
        Build a feature name from a prefix and source feature.

        Args:
            prefix: The prefix for the feature (e.g., "max_aggr")
            mloda_source_feature: The source feature name

        Returns:
            The combined feature name with double underscore separator
        """
        return f"{prefix}__{mloda_source_feature}"

    @classmethod
    def validate_feature_name(cls, feature_name: str, prefix_patterns: str | List[str]) -> bool:
        """
        Validate that a feature name matches the expected pattern(s).

        Args:
            feature_name: The feature name to validate
            prefix_patterns: Regex pattern(s) for the prefix - can be a string or list of strings

        Returns:
            True if valid, False otherwise
        """

        try:
            # Handle both single pattern and list of patterns
            if isinstance(prefix_patterns, str):
                prefix_patterns = [prefix_patterns]

            # Check if the feature name matches any of the prefix patterns
            for prefix_pattern in prefix_patterns:
                match = re.match(prefix_pattern, feature_name)
                if match:
                    # For patterns that don't require double underscore (like onehot~pattern),
                    # just return True if pattern matches
                    if "__" not in prefix_pattern:
                        return True

                    # For standard patterns, check double underscore requirement
                    if "__" in feature_name:
                        parts = feature_name.split("__", 1)
                        if len(parts) == 2 and parts[1]:
                            return True

            return False
        except Exception:
            return False

    # @classmethod
    # def _is_chained_feature(cls, feature_name: str, prefix_pattern: str) -> bool:
    #    """
    #    Check if a feature name follows the chaining pattern.
    #
    #    Args:
    #        feature_name: The feature name to check
    #        prefix_pattern: Regex pattern for the prefix
    #
    #    Returns:
    #        True if the feature is chained, False otherwise
    #    """
    #    try:
    #        # First validate that this is a valid feature name
    #        if not cls.validate_feature_name(feature_name, prefix_pattern):
    #            return False
    #
    #        # Extract the source feature
    #        mloda_source_feature = cls.extract_source_feature(feature_name, prefix_pattern)#
    #
    #
    #        # If the source feature contains a double underscore, it's chained
    #        return "__" in mloda_source_feature
    #    except Exception:
    #        return False

    @classmethod
    def get_prefix_part(cls, feature_name: str, prefix_patterns: str | List[str]) -> Optional[str]:
        """
        Extract the prefix part from a feature name.

        Args:
            feature_name: The feature name to parse
            prefix_patterns: Regex pattern(s) for the prefix - can be a string or list of strings

        Returns:
            The prefix part of the name, or None if the pattern doesn't match

        Example:
            For "max_aggr__temperature" with pattern r"^([w]+)_aggr__", returns "max"
        """
        # Handle both single pattern and list of patterns
        if isinstance(prefix_patterns, str):
            prefix_patterns = [prefix_patterns]

        # Try each pattern to find a match
        for prefix_pattern in prefix_patterns:
            match = re.match(prefix_pattern, feature_name)
            if match:
                # For OneHot column patterns, return "onehot"
                if "onehot_encoded__" in prefix_pattern and "~" in prefix_pattern:
                    return "onehot"
                # For standard patterns, return the captured group
                elif match.groups():
                    return match.group(1)

        return None

    @classmethod
    def extract_source_feature_from_options(cls, options: "Options") -> Union[str, "Feature", None]:
        """Supports both string and nested Feature objects."""
        return options.get(DefaultOptionKeys.mloda_source_feature)

    @classmethod
    def build_feature_with_nested_source(
        cls, prefix: str, source_feature: Union[str, "Feature"], additional_options: Optional["Options"] = None
    ) -> "Feature":
        """
        NEW METHOD: Build a Feature object with nested source feature support.

        Args:
            prefix: The prefix for the feature (e.g., "max_aggr")
            source_feature: Either a string or Feature object
            additional_options: Additional options to merge

        Returns:
            Feature object with proper options structure
        """

        # Create base options
        options = additional_options or Options()

        # Add source feature to context (metadata, doesn't affect grouping)
        options.add_to_context(DefaultOptionKeys.mloda_source_feature, source_feature)

        return Feature(prefix, options)

    @classmethod
    def supports_nested_chaining(cls, options: "Options") -> bool:
        """Check if options contain nested Feature objects."""
        source = cls.extract_source_feature_from_options(options)
        return isinstance(source, Feature)

    @classmethod
    def extract_source_feature_unified(
        cls, options: "Options", feature_name: str, prefix_pattern: str
    ) -> Optional[Set["Feature"]]:
        """
        This method first checks for nested Feature objects in options, then falls back to string-based parsing.
        This is the recommended method for feature groups to use in their input_features() method.

        Args:
            options: Options object that may contain nested Feature objects
            feature_name: String feature name for fallback string-based parsing
            prefix_pattern: Regex pattern for string-based parsing

        Returns:
            Set of Feature objects, or None if no source features found
        """
        # First, check for nested Feature objects (new approach)
        nested_source = cls.extract_source_feature_from_options(options)
        if isinstance(nested_source, Feature):
            return {nested_source}  # Return the Feature object directly
        elif isinstance(nested_source, str):
            return {Feature(nested_source)}  # Create Feature from string

        # Fall back to string-based parsing (traditional approach)
        source_feature_name = cls.extract_source_feature(feature_name, prefix_pattern)
        return {Feature(source_feature_name)}

    @classmethod
    def convert_nested_to_string_equivalent(cls, feature: "Feature") -> str:
        """
        NEW METHOD: Convert nested Feature structure to equivalent string representation.
        This is useful for debugging, logging, or backward compatibility.
        Supports unlimited depth.

        Args:
            feature: Feature object with potentially nested source features

        Returns:
            String representation of the nested feature chain
        """

        source_feature = cls.extract_source_feature_from_options(feature.options)

        if source_feature is None:
            return feature.name.name
        elif isinstance(source_feature, str):
            return f"{feature.name.name}__{source_feature}"
        elif isinstance(source_feature, Feature):
            # Recursive case - supports unlimited depth
            source_chain = cls.convert_nested_to_string_equivalent(source_feature)
            return f"{feature.name.name}__{source_chain}"
        else:
            return feature.name.name
