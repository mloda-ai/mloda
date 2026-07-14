"""
Mixin class providing default implementations for feature chain parsing.

Validation Design: ``element_validator`` vs ``match_guard``
==========================================================

PROPERTY_MAPPING supports two callable-valued keys that validate option values.
They serve different purposes and run at different points in the pipeline.

``element_validator`` (``DefaultOptionKeys.element_validator``)
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

``match_guard`` (``DefaultOptionKeys.match_guard``)
  - Does **not** require ``strict_validation``.
  - Runs inside ``FeatureChainParserMixin.match_feature_group_criteria``
    **after** basic matching succeeds (pattern + property mapping validation).
  - Receives the **raw option value** exactly as stored in Options, before any
    list unpacking or element iteration.
  - On failure: logs a debug message and returns ``False`` (non-match). If the
    guard raises an exception, the exception is caught, logged, and the value is
    treated as invalid.
  - Use case: validating the shape or composite type of the whole value
    (e.g., ``lambda v: isinstance(v, list) and all(isinstance(i, str) for i in v)``).

When both are present on the same mapping entry, ``element_validator`` runs
first (during property mapping validation) on each parsed element, then
``match_guard`` runs on the raw value. If ``element_validator`` rejects an
element, the match fails with a ``ValueError`` before ``match_guard`` is
reached.

Validators must be pure functions with no side effects. They may be called
multiple times during feature group resolution (once per candidate feature
group). Return values use truthy/falsy semantics: any falsy return (``False``,
``0``, ``""``, ``[]``) is treated as rejection.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any, Optional, cast

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import (
    FeatureChainParser,
    CHAIN_SEPARATOR,
    INPUT_SEPARATOR,
)
from mloda.core.abstract_plugins.components.feature_chainer.property_spec import PropertySpec
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
    - Must be callable (enforced at ``PropertySpec`` construction)
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

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        """
        Parse input features from feature name or options.

        First attempts to parse in_features from the feature name string.
        Falls back to options.get_in_features() if string parsing fails.

        Chained children are left at the default forward_group, which forwards all
        consumer group options. Authors opt out via forward_group=False, an allowlist,
        or forward_group_exclude.

        Args:
            options: Options containing configuration
            feature_name: Feature name to parse

        Returns:
            Set of Feature objects representing input features, or None

        Raises:
            ValueError: If in_feature constraints are violated
        """
        prefix_patterns = self._get_prefix_patterns()
        operation_config, in_feature = FeatureChainParser.parse_feature_name(
            feature_name, prefix_patterns, CHAIN_SEPARATOR
        )

        # String-based parsing succeeded
        if operation_config is not None and in_feature is not None and in_feature:
            in_features = in_feature.split(self.IN_FEATURE_SEPARATOR)
            self._validate_in_feature_count(in_features, feature_name)
            return {Feature(f) for f in in_features}

        # Configuration-based fallback using get_in_features()
        in_features_set = options.get_in_features()
        self._validate_in_feature_count(list(in_features_set), feature_name)
        return set(in_features_set)

    def _validate_in_feature_count(self, in_features: list[Any], feature_name: str) -> None:
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

        After basic matching succeeds, enforces ``match_guard`` constraints from
        PROPERTY_MAPPING entries. For each entry that defines a
        ``DefaultOptionKeys.match_guard`` callable, the guard is called with the raw
        option value. Returning a falsy value causes this method to return False. If
        the guard raises an exception, it is caught and the value is treated as
        invalid. See the module docstring for the full validation design.

        Also enforces MIN_IN_FEATURES / MAX_IN_FEATURES constraints when
        in_features is present in options.

        For string-parsed features, additionally checks consumer-forwarded option
        values against the value parsed from the feature name: if a PROPERTY_MAPPING
        key arrived via forwarding (it is in ``options.inherited_group_keys``) and its
        value differs from the name-parsed value, a ``ValueError`` is raised, because
        the name-parsed value would otherwise silently win. Setting the environment
        variable ``MLODA_ALLOW_FORWARDED_NAME_MISMATCH=1`` downgrades this error to a
        warning.

        Args:
            feature_name: Feature name to match
            options: Options containing configuration
            data_access_collection: Optional data access collection (unused)

        Returns:
            True if feature matches criteria, False otherwise
        """
        prefix_patterns = cls._get_prefix_patterns()
        property_mapping = cls._get_property_mapping()

        try:
            # Use the unified parser for basic matching
            result = FeatureChainParser.match_configuration_feature_chain_parser(
                feature_name,
                options,
                property_mapping=property_mapping,
                prefix_patterns=prefix_patterns,
            )
        except ValueError:
            return False

        # If basic match succeeded and it's a string-based feature, call validation hook
        if result:
            operation_config, source_feature = FeatureChainParser.parse_feature_name(
                feature_name, prefix_patterns, CHAIN_SEPARATOR
            )
            if operation_config is not None and source_feature is not None:
                if not cls._validate_string_match(feature_name, operation_config, source_feature):
                    return False
                cls._validate_forwarded_name_mismatch(feature_name, operation_config, property_mapping, options)

        if not cls._validate_required_when(result, feature_name, prefix_patterns, property_mapping, options):
            return False

        if not cls._validate_match_guards(result, options, property_mapping):
            return False

        if not cls._validate_in_features(result, options):
            return False

        return result

    @classmethod
    def _strict_validation_rejection_reason(cls, feature_name: str | FeatureName, options: Options) -> str | None:
        """Return the ValueError message that match_feature_group_criteria discards, if any.

        Only surfaces ValueErrors raised by property-mapping validation (genuine
        strict_validation rejections). A ValueError raised while parsing a
        PREFIX_PATTERN match (malformed feature name, no chain separator) is a
        parse error, not an option-value rejection, and is treated as nothing to
        report. Returns None when nothing was rejected (match succeeded, or the
        candidate is unrelated). Diagnostic-only: does not affect
        match_feature_group_criteria's behavior.
        """
        prefix_patterns = cls._get_prefix_patterns()
        if prefix_patterns:
            try:
                if FeatureChainParser._match_pattern_based_feature(feature_name, prefix_patterns, CHAIN_SEPARATOR):
                    return None
            except ValueError:
                return None

        property_mapping = cls._get_property_mapping()
        if property_mapping is None:
            return None

        try:
            FeatureChainParser._validate_options_against_property_mapping(options, property_mapping)
        except ValueError as exc:
            return str(exc)
        return None

    @classmethod
    def _validate_forwarded_name_mismatch(
        cls,
        feature_name: str | FeatureName,
        operation_config: str,
        property_mapping: dict[str, PropertySpec] | None,
        options: Options,
    ) -> None:
        """Reject consumer-forwarded option values that contradict the name-parsed value.

        The name-parsed value takes precedence, so a differing forwarded value would be
        silently ignored. Raises ValueError unless MLODA_ALLOW_FORWARDED_NAME_MISMATCH
        downgrades the error to a warning.
        """
        if property_mapping is None or not options.inherited_group_keys:
            return
        prop_key = cls._find_property_key_for_value(property_mapping, operation_config)
        if prop_key is None or prop_key not in options.inherited_group_keys:
            return
        inherited_value = options.get(prop_key)
        if inherited_value is None or str(inherited_value) == operation_config:
            return
        message = (
            f"Feature '{feature_name}': option '{prop_key}' was forwarded from a consumer with value "
            f"'{inherited_value}', but the feature name parses to '{operation_config}'. The name-parsed value "
            f"takes precedence, so the forwarded value would be silently ignored. Carve the key out with "
            f"forward_group_exclude={{'{prop_key}'}} on the child in the consumer's input_features, or use an "
            f"allowlist / forward_group=False. Set MLODA_ALLOW_FORWARDED_NAME_MISMATCH=1 to downgrade this "
            f"error to a warning."
        )
        if os.environ.get("MLODA_ALLOW_FORWARDED_NAME_MISMATCH", "").lower() in ("1", "true"):
            logger.warning(message)
            return
        raise ValueError(message)

    @staticmethod
    def _find_property_key_for_value(property_mapping: dict[str, PropertySpec], operation_config: str) -> str | None:
        """Return the PROPERTY_MAPPING key whose values contain operation_config, if any."""
        for prop_key, prop_value in property_mapping.items():
            extracted = FeatureChainParser._extract_property_values(prop_value)
            if operation_config in extracted:
                return prop_key
        return None

    @classmethod
    def _validate_required_when(
        cls,
        result: bool,
        feature_name: str | FeatureName,
        prefix_patterns: list[str],
        property_mapping: dict[str, PropertySpec] | None,
        options: Options,
    ) -> bool:
        # Enforce required_when constraints from PROPERTY_MAPPING.
        # Build effective options by merging string-parsed operation_config into
        # the Options object so predicates see values from both sources.
        if result and property_mapping is not None and cls._has_required_when_predicates(property_mapping):
            effective_options = cls._build_effective_options(feature_name, prefix_patterns, property_mapping, options)
            for key, mapping_entry in property_mapping.items():
                predicate = mapping_entry.required_when
                if predicate is None:
                    continue
                if predicate(effective_options) and effective_options.get(key) is None:
                    logger.debug(
                        "Feature group %s requires option '%s' (predicate %s is satisfied) but it was not provided.",
                        cls.__name__,
                        key,
                        getattr(predicate, "__name__", repr(predicate)),
                    )
                    return False
        return True

    @classmethod
    def _validate_match_guards(
        cls, result: bool, options: Options, property_mapping: dict[str, PropertySpec] | None
    ) -> bool:
        # Enforce match_guard constraints from PROPERTY_MAPPING
        if result and property_mapping is not None:
            for key, mapping_entry in property_mapping.items():
                guard = mapping_entry.match_guard
                if guard is None:
                    continue
                value = options.get(key)
                if value is None:
                    continue
                try:
                    if not guard(value):
                        logger.debug("match_guard for '%s' rejected value %r", key, value)
                        return False
                except (TypeError, ValueError, AttributeError) as exc:
                    logger.debug("match_guard for '%s' raised %s for value %r", key, exc, value)
                    return False
        return True

    @classmethod
    def _validate_in_features(cls, result: bool, options: Options) -> bool:
        # Enforce MIN/MAX_IN_FEATURES when in_features is present in options
        if result and hasattr(cls, "MIN_IN_FEATURES") and hasattr(cls, "MAX_IN_FEATURES"):
            in_features_raw = options.get(DefaultOptionKeys.in_features)
            if in_features_raw is not None:
                if isinstance(in_features_raw, (list, tuple, set, frozenset)) and not in_features_raw:
                    # Present but empty: zero in_features, a non-match rather than an error.
                    count = 0
                else:
                    count = len(options.get_in_features())
                if count < cls.MIN_IN_FEATURES:
                    return False
                if cls.MAX_IN_FEATURES is not None and count > cls.MAX_IN_FEATURES:
                    return False
        return True

    @classmethod
    def _get_prefix_patterns(cls) -> list[str]:
        """Get prefix/suffix patterns from class attributes."""
        patterns = []
        if hasattr(cls, "PREFIX_PATTERN"):
            patterns.append(cls.PREFIX_PATTERN)
        if hasattr(cls, "SUFFIX_PATTERN"):
            patterns.append(cls.SUFFIX_PATTERN)
        return patterns

    @classmethod
    def _get_property_mapping(cls) -> Optional[dict[str, PropertySpec]]:
        """Get property mapping from class attribute."""
        if hasattr(cls, "PROPERTY_MAPPING"):
            return cast(Optional[dict[str, PropertySpec]], cls.PROPERTY_MAPPING)
        return None

    @staticmethod
    def _has_required_when_predicates(property_mapping: dict[str, PropertySpec]) -> bool:
        """Return True if any entry in property_mapping uses required_when."""
        for value in property_mapping.values():
            if value.required_when is not None:
                return True
        return False

    @classmethod
    def _build_effective_options(
        cls,
        feature_name: str,
        prefix_patterns: list[str],
        property_mapping: dict[str, PropertySpec],
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
            # Already present in options: no merge needed
            if options.get(prop_key) is not None:
                continue
            # Check if operation_config is a valid value for this property
            extracted = FeatureChainParser._extract_property_values(prop_value)
            if operation_config in extracted:
                category = FeatureChainParser._determine_parameter_category(prop_key, prop_value, options)
                merged_group = dict(options.group)
                merged_context = dict(options.context)
                if category == DefaultOptionKeys.context:
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
    def _extract_source_features(cls, feature: Feature) -> list[str]:
        """
        Extract source features from a feature.

        Tries string-based parsing first, falls back to configuration-based.
        Uses class attributes IN_FEATURE_SEPARATOR and PREFIX_PATTERN.

        Args:
            feature: The feature to extract source features from

        Returns:
            List of source feature names
        """
        prefix_patterns = cls._get_prefix_patterns()

        operation_config, source_feature = FeatureChainParser.parse_feature_name(
            feature.name, prefix_patterns, CHAIN_SEPARATOR
        )

        # String-based parsing succeeded
        if operation_config is not None and source_feature is not None and source_feature:
            return source_feature.split(cls.IN_FEATURE_SEPARATOR)

        # Configuration-based fallback using get_in_features()
        in_features_set = feature.options.get_in_features()
        return [f.name for f in in_features_set]

    @classmethod
    def _extract_operation_and_source_feature(
        cls, feature: Feature, extract_fn: Callable[[Feature], Any], label: str
    ) -> tuple[Any, str]:
        """
        Extract an operation parameter and the primary source feature name from a feature.

        Args:
            feature: The feature to extract parameters from
            extract_fn: Callable that returns the operation-specific value (or None if not found)
            label: Human-readable noun used in the error message when extraction fails

        Returns:
            Tuple of (operation_value, source_feature_name)

        Raises:
            ValueError: If the operation value cannot be extracted
        """
        source_features = cls._extract_source_features(feature)
        operation = extract_fn(feature)
        if operation is None:
            raise ValueError(f"Could not extract {label} from: {feature.name}")
        return operation, source_features[0]

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
        _name: str
        if isinstance(feature_or_name, Feature) and isinstance(options_or_key, str):
            _name = feature_or_name.name
            _options = feature_or_name.options
            _key = options_or_key
        else:
            _name = str(feature_or_name)
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
