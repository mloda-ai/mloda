"""
Mixin class providing default implementations for feature chain parsing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, cast

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import (
    FeatureChainParser,
    CHAIN_SEPARATOR,
    INPUT_SEPARATOR,
)
from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys


class FeatureChainParserMixin:
    """
    Mixin providing default implementations for feature chain parsing.

    Subclasses should define:
    - PREFIX_PATTERN or SUFFIX_PATTERN: Regex patterns for matching
    - PROPERTY_MAPPING: Property validation mapping
    - IN_FEATURE_SEPARATOR: Optional custom separator (default: "&")
    - MIN_IN_FEATURES: Optional minimum in_feature count (default: 1)
    - MAX_IN_FEATURES: Optional maximum in_feature count (default: None)
    - REQUIRED_WHEN: Optional dict mapping option keys to predicate callables.
      Each predicate receives the Options object and returns True if the option
      is required. When the predicate returns True and the option is absent,
      match_feature_group_criteria returns False.
    """

    IN_FEATURE_SEPARATOR: str = INPUT_SEPARATOR
    MIN_IN_FEATURES: int = 1
    MAX_IN_FEATURES: Optional[int] = None

    @classmethod
    def _validate_string_match(cls, _feature_name: str, _operation_config: str, _in_feature: str) -> bool:
        """
        Hook for subclasses to provide custom validation for string-based matches.

        Args:
            _feature_name: The full feature name
            _operation_config: The parsed operation configuration
            _in_feature: The parsed in_feature

        Returns:
            True if the match is valid, False otherwise
        """
        return True

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """
        Parse input features from feature name or options.

        First attempts to parse in_features from the feature name string.
        Falls back to options.get_in_features() if string parsing fails.

        Args:
            options: Options containing configuration
            feature_name: Feature name to parse

        Returns:
            Set of Feature objects representing input features, or None

        Raises:
            ValueError: If in_feature constraints are violated
        """
        _feature_name = feature_name.name if isinstance(feature_name, FeatureName) else feature_name

        prefix_patterns = self._get_prefix_patterns()
        operation_config, in_feature = FeatureChainParser.parse_feature_name(
            _feature_name, prefix_patterns, CHAIN_SEPARATOR
        )

        # String-based parsing succeeded
        if operation_config is not None and in_feature is not None and in_feature:
            in_features = in_feature.split(self.IN_FEATURE_SEPARATOR)
            self._validate_in_feature_count(in_features, _feature_name)
            return {Feature(f) for f in in_features}

        # Configuration-based fallback using get_in_features()
        in_features_set = options.get_in_features()
        self._validate_in_feature_count(list(in_features_set), _feature_name)
        return set(in_features_set)

    def _validate_in_feature_count(self, in_features: List[Any], feature_name: str) -> None:
        """
        Validate that in_feature count meets min/max constraints.

        Args:
            in_features: List of in_features (strings or Feature objects)
            feature_name: Original feature name for error messages

        Raises:
            ValueError: If constraints are violated
        """
        count = len(in_features)

        if count < self.MIN_IN_FEATURES:
            raise ValueError(
                f"Feature '{feature_name}' requires at least {self.MIN_IN_FEATURES} in_feature(s), but found {count}"
            )

        if self.MAX_IN_FEATURES is not None and count > self.MAX_IN_FEATURES:
            raise ValueError(
                f"Feature '{feature_name}' allows at most {self.MAX_IN_FEATURES} in_feature(s), but found {count}"
            )

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: str | FeatureName,
        options: Options,
        _data_access_collection: Any = None,
    ) -> bool:
        """
        Match feature against criteria using pattern-based or config-based parsing.

        Delegates to FeatureChainParser.match_configuration_feature_chain_parser() and
        optionally calls _validate_string_match() hook for custom validation.

        Args:
            feature_name: Feature name to match
            options: Options containing configuration
            data_access_collection: Optional data access collection (unused)

        Returns:
            True if feature matches criteria, False otherwise
        """
        _feature_name = feature_name.name if isinstance(feature_name, FeatureName) else feature_name

        prefix_patterns = cls._get_prefix_patterns()
        property_mapping = cls._get_property_mapping()

        try:
            # Use the unified parser for basic matching
            result = FeatureChainParser.match_configuration_feature_chain_parser(
                _feature_name,
                options,
                property_mapping=property_mapping,
                prefix_patterns=prefix_patterns,
            )
        except ValueError:
            return False

        # If basic match succeeded and it's a string-based feature, call validation hook
        if result:
            operation_config, source_feature = FeatureChainParser.parse_feature_name(
                _feature_name, prefix_patterns, CHAIN_SEPARATOR
            )
            if operation_config is not None and source_feature is not None:
                if not cls._validate_string_match(_feature_name, operation_config, source_feature):
                    return False

        # Enforce REQUIRED_WHEN conditional option constraints
        if result and hasattr(cls, "REQUIRED_WHEN"):
            required_when = cast(Dict[str, Any], cls.REQUIRED_WHEN)
            for key, predicate in required_when.items():
                if predicate(options) and options.get(key) is None:
                    return False

        # Enforce MIN/MAX_IN_FEATURES when in_features is present in options
        if result and hasattr(cls, "MIN_IN_FEATURES") and hasattr(cls, "MAX_IN_FEATURES"):
            in_features_raw = options.get(DefaultOptionKeys.in_features)
            if in_features_raw is not None:
                in_features = options.get_in_features()
                count = len(in_features)
                if count < cls.MIN_IN_FEATURES:
                    return False
                if cls.MAX_IN_FEATURES is not None and count > cls.MAX_IN_FEATURES:
                    return False

        return result

    @classmethod
    def _get_prefix_patterns(cls) -> List[str]:
        """Get prefix/suffix patterns from class attributes."""
        patterns = []
        if hasattr(cls, "PREFIX_PATTERN"):
            patterns.append(cls.PREFIX_PATTERN)
        if hasattr(cls, "SUFFIX_PATTERN"):
            patterns.append(cls.SUFFIX_PATTERN)
        return patterns

    @classmethod
    def _get_property_mapping(cls) -> Optional[Dict[str, Any]]:
        """Get property mapping from class attribute."""
        if hasattr(cls, "PROPERTY_MAPPING"):
            return cast(Dict[str, Any], cls.PROPERTY_MAPPING)
        return None

    @classmethod
    def _extract_source_features(cls, feature: Feature) -> List[str]:
        """
        Extract source features from a feature.

        Tries string-based parsing first, falls back to configuration-based.
        Uses class attributes IN_FEATURE_SEPARATOR and PREFIX_PATTERN.

        Args:
            feature: The feature to extract source features from

        Returns:
            List of source feature names
        """
        feature_name = feature.get_name()
        prefix_patterns = cls._get_prefix_patterns()

        operation_config, source_feature = FeatureChainParser.parse_feature_name(
            feature_name, prefix_patterns, CHAIN_SEPARATOR
        )

        # String-based parsing succeeded
        if operation_config is not None and source_feature is not None and source_feature:
            return source_feature.split(cls.IN_FEATURE_SEPARATOR)

        # Configuration-based fallback using get_in_features()
        in_features_set = feature.options.get_in_features()
        return [f.get_name() for f in in_features_set]

    @classmethod
    def _resolve_operation(
        cls,
        feature_or_name: Any,
        options_or_key: Any,
        config_key: Optional[str] = None,
    ) -> Optional[str]:
        """Resolve the operation type from either a chained feature name or options.

        Many feature groups need to extract an operation type (e.g. aggregation type,
        scaler type, algorithm) from a feature. The value can come from the feature
        name string (parsed via PREFIX_PATTERN) or from a configuration key in options.
        This helper encapsulates that dual-path lookup.

        Supports two calling conventions:

        1. ``cls._resolve_operation(feature, config_key)``
           Extracts the name and options from the Feature object.

        2. ``cls._resolve_operation(feature_name, options, config_key)``
           Uses the provided name (str or FeatureName) and Options separately.

        The string-based path always takes precedence. If the feature name matches
        PREFIX_PATTERN, the captured group is returned. Otherwise, falls back to
        ``options.get(config_key)`` and converts to string.

        Args:
            feature_or_name: A Feature object (convention 1) or a feature name
                as str/FeatureName (convention 2).
            options_or_key: The config_key str (convention 1) or an Options
                object (convention 2).
            config_key: The options key to fall back on (convention 2 only).

        Returns:
            The resolved operation as a string, or None if neither path matches.
        """
        if isinstance(feature_or_name, Feature) and isinstance(options_or_key, str):
            _name = feature_or_name.get_name()
            _options = feature_or_name.options
            _key = options_or_key
        else:
            _name = feature_or_name.name if isinstance(feature_or_name, FeatureName) else str(feature_or_name)
            _options = options_or_key
            _key = config_key if config_key is not None else ""

        prefix_patterns = cls._get_prefix_patterns()
        operation_config, _ = FeatureChainParser.parse_feature_name(_name, prefix_patterns, CHAIN_SEPARATOR)
        if operation_config is not None:
            return operation_config
        value = _options.get(_key)
        if value is not None:
            return str(value)
        return None
