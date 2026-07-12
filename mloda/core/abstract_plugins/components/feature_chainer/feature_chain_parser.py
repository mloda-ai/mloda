"""
Feature chain parser for handling feature name chaining across feature groups.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.default_options_key import (
    REMOVED_PROPERTY_KEYS,
    RESERVED_PROPERTY_KEYS,
    DefaultOptionKeys,
)

# Separator constants for feature name parsing
CHAIN_SEPARATOR = "__"  # Separates chained transformations (source→suffix)
COLUMN_SEPARATOR = "~"  # Separates multi-column output index
INPUT_SEPARATOR = "&"  # Separates multiple input features


class FeatureChainParser:
    """
    Mixin class for parsing feature names with chaining support.

    Feature chaining allows feature groups to be composed, where the output of one
    feature group becomes the input to another. This is reflected in the feature name
    using separators defined as module constants:

    Separators:
        - CHAIN_SEPARATOR ("__"): Separates chained transformations (source→suffix)
          Example: price__mean_imputed__sum_7_day_window__max_aggr
          (L→R: price is source, each suffix transforms the previous result)
        - COLUMN_SEPARATOR ("~"): Separates multi-column output index
          Example: feature__pca~0, feature__pca~1
        - INPUT_SEPARATOR ("&"): Separates multiple input features
          Example: point1&point2__haversine_distance

    Each feature group in the chain extracts its relevant portion and passes the
    rest to the next feature group in the chain.
    """

    @classmethod
    def is_chained_feature(cls, feature_name: str) -> bool:
        """Check if feature name contains the chain separator.

        Args:
            feature_name: The feature name to check

        Returns:
            True if the feature name contains CHAIN_SEPARATOR, False otherwise
        """
        return CHAIN_SEPARATOR in feature_name

    @classmethod
    def parse_feature_name(
        cls,
        feature_name: FeatureName | str,
        prefix_patterns: list[str],
        pattern: str = CHAIN_SEPARATOR,
    ) -> tuple[str | None, str | None]:
        """Internal method for parsing feature names - used by match_configuration_feature_chain_parser."""
        _feature_name: str = feature_name

        parts = _feature_name.rsplit(pattern, 1)
        source_feature = parts[0] if len(parts) > 1 else ""
        operation_part = parts[1] if len(parts) > 1 else parts[0]

        for suffix_pattern in prefix_patterns:
            if re.match(suffix_pattern, _feature_name) is None:
                continue

            if len(parts) == 1 or not source_feature:
                raise ValueError(f"Matches the pattern {pattern}, but has no source feature: {_feature_name}")

            match = re.match(suffix_pattern, _feature_name)
            if match and match.groups():
                operation_config = match.group(1)
            else:
                operation_config = operation_part.split("_")[0]

            return operation_config, source_feature

        return None, None

    @classmethod
    def _match_pattern_based_feature(
        cls,
        feature_name: str | FeatureName,
        prefix_patterns: list[str],
        pattern: str = CHAIN_SEPARATOR,
    ) -> bool:
        """Internal method for matching pattern-based features - used by match_configuration_feature_chain_parser."""
        _feature_name: FeatureName = FeatureName(feature_name) if isinstance(feature_name, str) else feature_name

        has_prefix_configuration, source_feature = cls.parse_feature_name(_feature_name, prefix_patterns, pattern)
        if has_prefix_configuration is None or source_feature is None:
            return False
        return True

    @classmethod
    def _can_skip_required_check(cls, property_value: Any) -> bool:
        """Check if the base parser should treat this property as optional.

        Returns True when the property has a default value or uses conditional
        requirements (required_when).  In both cases the base validation loop
        should not reject the match just because the option is absent; either
        the default will be applied later, or the required_when predicate in
        FeatureChainParserMixin will decide.
        """
        if not isinstance(property_value, dict):
            return False
        return DefaultOptionKeys.default in property_value or DefaultOptionKeys.required_when in property_value

    @classmethod
    def _is_context_parameter(cls, property_value: Any) -> bool:
        """Check if property is marked as context parameter in mapping."""
        return isinstance(property_value, dict) and property_value.get(DefaultOptionKeys.context, False)

    @classmethod
    def _is_strict_validation(cls, property_value: Any) -> bool:
        """Check if property requires strict validation (values must be in mapping)."""
        return isinstance(property_value, dict) and property_value.get(DefaultOptionKeys.strict_validation, False)

    @classmethod
    def _get_element_validator(cls, property_value: Any) -> Any:
        """Get the per-element validator from a property mapping spec if present."""
        if isinstance(property_value, dict):
            return property_value.get(DefaultOptionKeys.element_validator, None)
        return None

    @classmethod
    def _validate_property_value(
        cls, found_property_val: Any, property_value: Any, property_name: str, original_property_config: Any
    ) -> None:
        """
        Unified validation: if strict validation -> apply the element_validator OR check membership.

        Raises ValueError if validation fails, otherwise returns None.
        """
        if not cls._is_strict_validation(original_property_config):
            return  # No validation needed

        element_validator = cls._get_element_validator(original_property_config)

        if element_validator is not None:
            # Use the element validator if available
            if not element_validator(found_property_val):
                raise ValueError(f"Property value '{found_property_val}' failed validation for '{property_name}'")
        else:
            # Fallback to membership check. An unhashable element (e.g. a dict) can never be
            # a member of the accepted set, so it is a clean rejection, not a TypeError.
            try:
                is_member = found_property_val in property_value
            except TypeError:
                is_member = False
            if not is_member:
                raise ValueError(f"Property value '{found_property_val}' not found in mapping for '{property_name}'")

    @classmethod
    def _determine_parameter_category(cls, property_name: str, property_value: Any, options: Options) -> str:
        """
        Determine whether a parameter should be in group or context category.

        Priority:
        1. User explicit override (if property exists in specific category)
        2. Property mapping default (mloda_context flag)
        3. Fallback to group

        Args:
            property_name: Name of the property
            property_value: Property configuration from mapping
            options: Options object containing user's parameter placement

        Returns:
            "group" or "context" indicating target category

        Raises:
            ValueError: If parameter exists in both group and context
        """

        if property_name in options.group and property_name in options.context:
            raise ValueError(
                f"Parameter '{property_name}' exists in both group and context. "
                "This is not allowed. Please choose one category."
            )

        if property_name in options.group:
            return DefaultOptionKeys.group
        elif property_name in options.context:
            return DefaultOptionKeys.context
        elif cls._is_context_parameter(property_value):
            return DefaultOptionKeys.context
        else:
            return DefaultOptionKeys.group

    @classmethod
    def _extract_property_values(cls, property_value: Any) -> Any:
        """Extract a spec's declared value space.

        Prefers an explicit ``DefaultOptionKeys.allowed_values`` field; otherwise
        falls back to the legacy flattened form (all entries minus the reserved
        metadata keys).
        """
        if isinstance(property_value, dict):
            if DefaultOptionKeys.allowed_values in property_value:
                return property_value[DefaultOptionKeys.allowed_values]
            return {k: v for k, v in property_value.items() if k not in RESERVED_PROPERTY_KEYS}
        return property_value

    @classmethod
    def check_declared_default(cls, owner: str, key: str, spec: dict[str, Any]) -> None:
        """Check one spec's declared default against its own strict-validation rules.

        This is the single implementation of the default-check semantics, shared by
        ``validate_property_mapping_defaults`` (class-definition time) and by the
        ``property_spec`` authoring helper, so the two can never drift apart.

        Under strict validation, a non-``None`` declared default must pass the key's
        ``element_validator`` if it has one, otherwise be a member of the accepted set.
        An ``element_validator`` that itself errors when called with the default is
        reported as having raised (with the original exception chained as the cause),
        which is distinct from the validator running and rejecting the default.
        ``DefaultOptionKeys.required_when`` does NOT exempt this check: it is a
        conditional requirement, so the default still applies on the predicate-false
        branch. The remedies are to add the default to the accepted values or to
        remove the default (a key with no default is required).

        Raises ValueError on violation, otherwise returns None.
        """
        if DefaultOptionKeys.default not in spec:
            return

        default = spec[DefaultOptionKeys.default]
        if default is None:
            return

        if not cls._is_strict_validation(spec):
            return

        def _message(detail: str) -> str:
            return (
                f"{owner} declares default {default!r} for '{key}', "
                f"but {detail} under strict_validation. "
                f"Add the default to the accepted values, or remove the default "
                f"(a key with no default is required)."
            )

        element_validator = cls._get_element_validator(spec)
        if element_validator is not None:
            try:
                verdict = element_validator(default)
            except Exception as exc:
                raise ValueError(
                    _message("the key's element_validator raised an error when called with that default")
                ) from exc
            if not verdict:
                raise ValueError(_message("that default is rejected by the key's element_validator"))
            return

        extracted = cls._extract_property_values(spec)
        try:
            cls._validate_property_value(default, extracted, key, spec)
        except (ValueError, TypeError) as exc:
            detail = f"that default is not one of the accepted values {sorted(extracted, key=repr)}"
            raise ValueError(_message(detail)) from exc

    @classmethod
    def validate_property_mapping_defaults(cls, owner_name: str, property_mapping: dict[str, Any] | None) -> None:
        """Validate a PROPERTY_MAPPING at class-definition time.

        Rejects specs that still carry a removed key (see ``REMOVED_PROPERTY_KEYS``) and
        delegates every declared default to ``check_declared_default``.
        """
        if property_mapping is None:
            return

        for key, spec in property_mapping.items():
            if not isinstance(spec, dict):
                continue
            cls._reject_removed_property_keys(owner_name, key, spec)
            cls.check_declared_default(f"{owner_name}.PROPERTY_MAPPING", key, spec)

    @classmethod
    def _reject_removed_property_keys(cls, owner_name: str, key: str, spec: dict[str, Any]) -> None:
        """Fail loudly on a spec that still uses a removed PROPERTY_MAPPING key.

        The removed keys are no longer reserved metadata, so ``_extract_property_values``
        would absorb a leftover one into a legacy flattened spec's accepted value space,
        silently widening what the feature group matches. Rejecting the spec turns that
        silent corruption into an actionable migration error.
        """
        for stale, replacement in REMOVED_PROPERTY_KEYS.items():
            if stale in spec:
                raise ValueError(
                    f"{owner_name}.PROPERTY_MAPPING['{key}'] uses the removed key '{stale}'. "
                    f"Rename it to '{replacement.value}' (DefaultOptionKeys.{replacement.value})."
                )

    @classmethod
    def _unpack_property_value(cls, found_property_value: Any) -> list[Any]:
        """Unpack an option value into the elements the spec validates.

        The spec declares the arity, not the caller's Python syntax: every sequence
        container (list, tuple, set, frozenset) unpacks element-wise and identically.
        A ``str`` is a scalar, not a sequence of characters, and a ``dict`` is one
        composite value, not a sequence of its keys. Elements keep their real type;
        the only normalization is a ``Feature`` reduced to its name.
        """
        if isinstance(found_property_value, (list, tuple, set, frozenset)):
            elements = list(found_property_value)
        else:
            elements = [found_property_value]

        return [element.name if isinstance(element, Feature) else element for element in elements]

    @classmethod
    def _process_found_property_value(
        cls, found_property_value: Any, property_value: Any, property_name: str, original_property_config: Any
    ) -> list[Any]:
        collected_property_value: list[Any] = []
        for found_property_val in cls._unpack_property_value(found_property_value):
            # Use unified validation function
            cls._validate_property_value(found_property_val, property_value, property_name, original_property_config)

            collected_property_value.append(found_property_val)

        return collected_property_value

    @classmethod
    def _validate_final_properties(
        cls, property_tracker: dict[str, list[Any] | None], property_mapping: dict[str, Any]
    ) -> bool:
        """Validate that all required properties are present.

        Presence is tracked explicitly: ``None`` means the option was absent, while an
        empty list means it was present with zero elements (an empty container), which
        is vacuously valid and still satisfies the required-presence check.
        """
        for key, value in property_tracker.items():
            property_config = property_mapping[key]
            can_skip = cls._can_skip_required_check(property_config)

            if value is None and not can_skip:
                return False
        return True

    @classmethod
    def _validate_options_against_property_mapping(cls, options: Options, property_mapping: dict[str, Any]) -> bool:
        """
        Shared validation logic for both string-based and configuration-based approaches.

        Args:
            options: Options object containing the parameters to validate
            property_mapping: Property mapping with validation rules

        Returns:
            True if validation passes, False otherwise
        """
        # None marks an absent option; a list (possibly empty) marks a present one.
        property_tracker: dict[str, list[Any] | None] = {}
        for key in property_mapping:
            property_tracker[key] = None

        # Process each property in the mapping
        for property_name, property_value in property_mapping.items():
            found_property_value = options.get(property_name)
            property_value = cls._extract_property_values(property_value)

            # Handle missing properties: leave the tracker at None, so the final check
            # decides whether the option was required.
            if found_property_value is None:
                continue

            collected_property_value = cls._process_found_property_value(
                found_property_value, property_value, property_name, property_mapping[property_name]
            )

            if property_tracker[property_name] is not None:
                raise ValueError(f"Feature name has duplicate values for property '{property_name}'.")

            property_tracker[property_name] = collected_property_value
        return cls._validate_final_properties(property_tracker, property_mapping)

    @classmethod
    def match_configuration_feature_chain_parser(
        cls,
        feature_name: str | FeatureName,
        options: Options,
        property_mapping: Optional[dict[str, Any]] = None,
        prefix_patterns: Optional[list[str]] = None,
        pattern: str = CHAIN_SEPARATOR,
    ) -> bool:
        """
        Unified method for matching features using either configuration-based or pattern-based parsing.

        Args:
            feature_name: The feature name to match
            options: Options object containing configuration
            property_mapping: Optional property mapping for configuration-based parsing
            prefix_patterns: Optional prefix patterns for pattern-based parsing
            pattern: Pattern string for pattern-based parsing (defaults to CHAIN_SEPARATOR)

        Returns:
            True if the feature matches either pattern-based or configuration-based parsing, False otherwise
        """

        # string based matching
        if prefix_patterns is not None:
            if cls._match_pattern_based_feature(feature_name, prefix_patterns, pattern):
                return True

        # configuration-based
        if property_mapping is not None:
            return cls._validate_options_against_property_mapping(options, property_mapping)

        # If neither pattern-based nor configuration-based matching succeeded, return False
        return False

    @classmethod
    def extract_in_feature(cls, feature_name: str, suffix_pattern: str) -> str:
        """
        Extract the in_feature from a feature name based on the suffix pattern.

        Args:
            feature_name: The feature name to parse
            suffix_pattern: Regex pattern for the suffix (e.g., r"^.+__([w]+)$")

        Returns:
            The in_feature part of the name

        Raises:
            ValueError: If the feature name doesn't match the expected pattern
        """
        match = re.match(suffix_pattern, feature_name)
        if not match:
            raise ValueError(f"Invalid feature name format: {feature_name}")

        # For L→R: source is everything BEFORE the last CHAIN_SEPARATOR
        suffix_start = feature_name.rfind(CHAIN_SEPARATOR)
        if suffix_start == -1:
            raise ValueError(
                f"Invalid feature name format: {feature_name}. Missing chain separator '{CHAIN_SEPARATOR}'."
            )

        # Return everything BEFORE the last double underscore (the source)
        return feature_name[:suffix_start]
