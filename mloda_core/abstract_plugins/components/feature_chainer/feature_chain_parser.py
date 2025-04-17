"""
Feature chain parser for handling feature name chaining across feature groups.
"""

from __future__ import annotations

import re
from typing import Optional


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
    def validate_feature_name(cls, feature_name: str, prefix_pattern: str) -> bool:
        """
        Validate that a feature name matches the expected pattern.

        Args:
            feature_name: The feature name to validate
            prefix_pattern: Regex pattern for the prefix

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check if the feature name matches the prefix pattern
            match = re.match(prefix_pattern, feature_name)
            if not match:
                return False

            # Check if the feature name contains a double underscore
            if "__" not in feature_name:
                return False

            # Check if there's something after the double underscore
            parts = feature_name.split("__", 1)
            if len(parts) != 2 or not parts[1]:
                return False

            return True
        except Exception:
            return False

    @classmethod
    def is_chained_feature(cls, feature_name: str, prefix_pattern: str) -> bool:
        """
        Check if a feature name follows the chaining pattern.

        Args:
            feature_name: The feature name to check
            prefix_pattern: Regex pattern for the prefix

        Returns:
            True if the feature is chained, False otherwise
        """
        try:
            # First validate that this is a valid feature name
            if not cls.validate_feature_name(feature_name, prefix_pattern):
                return False

            # Extract the source feature
            mloda_source_feature = cls.extract_source_feature(feature_name, prefix_pattern)

            # If the source feature contains a double underscore, it's chained
            return "__" in mloda_source_feature
        except Exception:
            return False

    @classmethod
    def get_prefix_part(cls, feature_name: str, prefix_pattern: str) -> Optional[str]:
        """
        Extract the prefix part from a feature name.

        Args:
            feature_name: The feature name to parse
            prefix_pattern: Regex pattern for the prefix

        Returns:
            The prefix part of the name, or None if the pattern doesn't match

        Example:
            For "max_aggr__temperature" with pattern r"^([w]+)_aggr__", returns "max"
        """
        match = re.match(prefix_pattern, feature_name)
        if not match:
            return None

        # Return the captured group from the regex
        return match.group(1)
