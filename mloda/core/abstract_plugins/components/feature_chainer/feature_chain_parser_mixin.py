"""
Mixin class providing default implementations for feature chain parsing.

Validation Design: ``type_validator`` vs ``validation_function``
================================================================

PROPERTY_MAPPING supports two callable-valued keys that validate option values.
They serve different purposes and run at different points in the pipeline.

``validation_function`` (``DefaultOptionKeys.validation_function``)
  - Requires ``strict_validation: True`` on the same mapping entry.
  - Runs inside ``FeatureChainParser._validate_property_value`` during
    ``_validate_options_against_property_mapping``.
  - Receives **individual parsed elements** after list unpacking
    (``_process_found_property_value`` converts lists to frozensets and
    iterates over each element).
  - On failure: raises ``ValueError`` with an actionable message identifying
    the property name and the rejected element value.
  - Use case: validating that each individual element satisfies a constraint
    (e.g., ``lambda x: isinstance(x, int) and x > 0``).

``type_validator`` (``DefaultOptionKeys.type_validator``)
  - Does **not** require ``strict_validation``.
  - Runs inside ``FeatureChainParserMixin.match_feature_group_criteria``
    **after** basic matching succeeds (pattern + property mapping validation).
  - Receives the **raw option value** exactly as stored in Options, before any
    list unpacking or element iteration.
  - On failure: logs a debug message and returns ``False`` (non-match). If the
    validator raises an exception, the exception is caught, logged, and the
    value is treated as invalid.
  - Use case: validating the shape or composite type of the whole value
    (e.g., ``lambda v: isinstance(v, list) and all(isinstance(i, str) for i in v)``).

When both are present on the same mapping entry, ``validation_function`` runs
first (during property mapping validation) on each parsed element, then
``type_validator`` runs on the raw value. If ``validation_function`` rejects an
element, the match fails with a ``ValueError`` before ``type_validator`` is
reached.

Validators must be pure functions with no side effects. They may be called
multiple times during feature group resolution (once per candidate feature
group). Return values use truthy/falsy semantics: any falsy return (``False``,
``0``, ``""``, ``[]``) is treated as rejection.
"""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


class FeatureChainParserMixin:
    """
    Mixin providing default implementations for feature chain parsing.

    Subclasses should define:
    - PREFIX_PATTERN or SUFFIX_PATTERN: Regex patterns for matching
    - PROPERTY_MAPPING: Property validation mapping (see docs/in_depth/property-mapping.md)
    - IN_FEATURE_SEPARATOR: Optional custom separator (default: "&")
    - MIN_IN_FEATURES: Optional minimum in_feature count (default: 1)
    - MAX_IN_FEATURES: Optional maximum in_feature count (default: None)

    PROPERTY_MAPPING supports conditional requirements via ``DefaultOptionKeys.required_when``.
    Attach a predicate ``(Options) -> bool`` to any mapping entry. When the predicate returns
    True and the option value is absent, ``match_feature_group_criteria`` rejects the match.
    When the predicate returns False, the option is treated as optional.

    This works for both string-based and configuration-based feature creation. For
    string-based features, the operation value parsed from the feature name is merged
    into effective options before predicate evaluation, so predicates see values from
    both the feature name and explicit options.

    Predicate contract:
    - Signature: ``(Options) -> bool``
    - Must be callable (non-callable values are skipped with a warning)
    - Must not raise exceptions
    - Must be a pure function (no side effects)
    - Non-bool truthy return values are treated as True

    See docs/in_depth/property-mapping.md for full details and examples.
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
        data_access_collection: Any = None,
    ) -> bool:
        """
        Match feature against criteria using pattern-based or config-based parsing.

        Delegates to FeatureChainParser.match_configuration_feature_chain_parser() and
        optionally calls _validate_string_match() hook for custom validation.

        After basic matching succeeds, enforces ``type_validator`` constraints from
        PROPERTY_MAPPING entries. For each entry that defines a
        ``DefaultOptionKeys.type_validator`` callable, the validator is called with
        the raw option value. Returning a falsy value causes this method to return
        False. If the validator raises an exception, it is caught and the value is
        treated as invalid. See the module docstring for the full validation design.

        Also enforces MIN_IN_FEATURES / MAX_IN_FEATURES constraints when
        in_features is present in options.

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

        # Enforce required_when constraints from PROPERTY_MAPPING.
        # Build effective options by merging string-parsed operation_config into
        # the Options object so predicates see values from both sources.
        if result and property_mapping is not None and cls._has_required_when_predicates(property_mapping):
            effective_options = cls._build_effective_options(
                _feature_name, prefix_patterns, property_mapping, options
            )
            for key, mapping_entry in property_mapping.items():
                if not isinstance(mapping_entry, dict):
                    continue
                predicate = mapping_entry.get(DefaultOptionKeys.required_when)
                if predicate is None:
                    continue
                if not callable(predicate):
                    logger.warning(
                        "required_when for '%s' in %s is not callable, skipping.",
                        key,
                        cls.__name__,
                    )
                    continue
                if predicate(effective_options) and effective_options.get(key) is None:
                    logger.debug(
                        "Feature group %s requires option '%s' (predicate %s is satisfied) "
                        "but it was not provided.",
                        cls.__name__,
                        key,
                        getattr(predicate, "__name__", repr(predicate)),
                    )
                    return False

        # Enforce type_validator constraints from PROPERTY_MAPPING
        if result and property_mapping is not None:
            for key, mapping_entry in property_mapping.items():
                if not isinstance(mapping_entry, dict):
                    continue
                validator = mapping_entry.get(DefaultOptionKeys.type_validator)
                if validator is None:
                    continue
                value = options.get(key)
                if value is None:
                    continue
                try:
                    if not validator(value):
                        logger.debug("type_validator for '%s' rejected value %r", key, value)
                        return False
                except (TypeError, ValueError, AttributeError) as exc:
                    logger.debug("type_validator for '%s' raised %s for value %r", key, exc, value)
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

    @staticmethod
    def _has_required_when_predicates(property_mapping: Dict[str, Any]) -> bool:
        """Return True if any entry in property_mapping uses required_when."""
        for value in property_mapping.values():
            if isinstance(value, dict) and DefaultOptionKeys.required_when in value:
                return True
        return False

    @classmethod
    def _build_effective_options(
        cls,
        feature_name: str,
        prefix_patterns: List[str],
        property_mapping: Dict[str, Any],
        options: Options,
    ) -> Options:
        """Build effective options by merging string-parsed values with explicit options.

        When a feature is matched by string pattern, the operation_config value extracted
        from the feature name is mapped to the corresponding PROPERTY_MAPPING key. This
        ensures that required_when predicates see values from both sources.

        If the feature is not string-based or no mapping key matches, returns the
        original options unchanged.
        """
        operation_config, _source_feature = FeatureChainParser.parse_feature_name(
            feature_name, prefix_patterns, CHAIN_SEPARATOR
        )
        if operation_config is None:
            return options

        # Find which property mapping key the operation_config value belongs to
        for prop_key, prop_value in property_mapping.items():
            if not isinstance(prop_value, dict):
                continue
            # Already present in options: no merge needed
            if options.get(prop_key) is not None:
                continue
            # Check if operation_config is a valid value for this property
            extracted = FeatureChainParser._extract_property_values(prop_value)
            if operation_config in extracted:
                category = FeatureChainParser._determine_parameter_category(prop_key, prop_value, options)
                merged_group = dict(options.group)
                merged_context = dict(options.context)
                if category == DefaultOptionKeys.context.value:
                    merged_context[prop_key] = operation_config
                else:
                    merged_group[prop_key] = operation_config
                return Options(
                    group=merged_group,
                    context=merged_context,
                    propagate_context_keys=options.propagate_context_keys,
                )

        return options

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
