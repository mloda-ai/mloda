"""
Feature chain parser for handling feature name chaining across feature groups.
"""

from __future__ import annotations

import contextvars
import difflib
import functools
import logging
import re
from collections.abc import Collection
from typing import Any, Optional

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.default_options_key import (
    PROPERTY_SPEC_KEYS,
    REMOVED_PROPERTY_KEYS,
    DefaultOptionKeys,
)
from mloda.core.abstract_plugins.components.utils import safe_field

logger = logging.getLogger(__name__)

# Separator constants for feature name parsing
CHAIN_SEPARATOR = "__"  # Separates chained transformations (source→suffix)
COLUMN_SEPARATOR = "~"  # Separates multi-column output index
INPUT_SEPARATOR = "&"  # Separates multiple input features

# Marks a matcher that already carries the required_when guard, so it is never wrapped twice.
REQUIRED_WHEN_GUARD_FLAG = "_mloda_required_when_guard"

# How many guards the current match call is nested in. A guarded matcher that delegates via super()
# reaches the guard of its parent, and only the outermost one may evaluate the predicates.
# A ContextVar (not a plain global) keeps the count per thread and per async task.
REQUIRED_WHEN_GUARD_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "mloda_required_when_guard_depth", default=0
)


class PropertyValueRejection(ValueError):
    """A present option value that the PROPERTY_MAPPING rejects: a VERDICT, not a crash.

    A ValueError subclass, so every existing ``except ValueError`` keeps working, and a distinct
    type, so a caller can treat it as a non-match without also swallowing the deliberate
    ValueErrors that carry actionable guidance (the forwarded-name mismatch).
    """


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
        prefix_patterns: list[Any],
        pattern: str = CHAIN_SEPARATOR,
    ) -> tuple[str | None, str | None]:
        """Internal method for parsing feature names - used by match_configuration_feature_chain_parser.

        A prefix pattern is anything ``re.match`` accepts: a ``str`` or a compiled ``re.Pattern``.
        """
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
        prefix_patterns: list[Any],
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
        the default will be applied later, or the required_when guard installed
        at class definition will decide.
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

        Raises PropertyValueRejection if validation fails, otherwise returns None.
        """
        if not cls._is_strict_validation(original_property_config):
            return  # No validation needed

        element_validator = cls._get_element_validator(original_property_config)

        if element_validator is not None:
            # A validator that RAISES cannot judge the value, which is a rejection, not a crash:
            # a membership-style validator raises TypeError on an unhashable element, and a
            # str-only one raises AttributeError on an int. Same exceptions as _validate_match_guards.
            try:
                verdict = element_validator(found_property_val)
            except (TypeError, ValueError, AttributeError):
                verdict = False
            if not verdict:
                raise PropertyValueRejection(
                    f"Property value '{found_property_val}' failed validation for '{property_name}'"
                )
        else:
            # Fallback to membership check. An unhashable element (e.g. a dict) can never be
            # a member of the accepted set, so it is a clean rejection, not a TypeError.
            try:
                is_member = found_property_val in property_value
            except TypeError:
                is_member = False
            if not is_member:
                raise PropertyValueRejection(
                    f"Property value '{found_property_val}' not found in mapping for '{property_name}'"
                )

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
    def extract_property_values(cls, property_value: Any) -> Any:
        """Return a spec's declared value space (``allowed_values``), or {} if it declares none.

        Never inferred by subtracting the known keys: the retired flattened form did that, and it
        absorbed every unrecognized key as an accepted value.
        """
        if isinstance(property_value, dict):
            return property_value.get(DefaultOptionKeys.allowed_values, {})
        return property_value

    @classmethod
    def _extract_property_values(cls, property_value: Any) -> Any:
        """Alias kept for existing callers."""
        return cls.extract_property_values(property_value)

    @classmethod
    def check_declared_default(cls, owner: str, key: str, spec: dict[str, Any]) -> None:
        """Check a spec's declared default against its own strict-validation rules.

        Shared by ``validate_property_mapping_defaults`` and ``property_spec`` so the two cannot
        drift. ``required_when`` does NOT exempt the check: the default still applies on the
        predicate-false branch.
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

        The rule ORDER below is load-bearing. Schema before shape: an unknown key means the
        remaining keys cannot be trusted, and a removed key must be reported as a rename rather
        than as some downstream shape error. Shape before the strict and default rules: those
        read the values the shape rule validates, so a malformed one would be mis-diagnosed (a
        non-callable ``element_validator`` gets blamed on the default).
        """
        if property_mapping is None:
            return

        for key, spec in property_mapping.items():
            cls._reject_non_dict_spec(owner_name, key, spec)
            cls._reject_unknown_spec_keys(owner_name, key, spec)
            cls._reject_malformed_spec_values(owner_name, key, spec)
            cls._reject_strict_without_value_space(owner_name, key, spec)
            cls.check_declared_default(f"{owner_name}.PROPERTY_MAPPING", key, spec)

    @classmethod
    def _reject_non_dict_spec(cls, owner_name: str, key: str, spec: Any) -> None:
        """Reject a bare container: it carries no keys, so every rule below would skip it."""
        if isinstance(spec, dict):
            return

        raise ValueError(
            f"{owner_name}.PROPERTY_MAPPING['{key}'] is a {type(spec).__name__}, not a spec dict. "
            f"Declare the accepted values under '{DefaultOptionKeys.allowed_values.value}' inside a spec dict."
        )

    @classmethod
    def _reject_unknown_spec_keys(cls, owner_name: str, key: str, spec: dict[str, Any]) -> None:
        """Reject every key outside the spec schema, reporting them all in one message.

        Removed keys lead, because they are the offenders with an exact remedy, but they never
        hide the rest.
        """
        unknown = [spec_key for spec_key in spec if spec_key not in PROPERTY_SPEC_KEYS]
        if not unknown:
            return

        known = sorted(str(k) for k in PROPERTY_SPEC_KEYS)
        removed = [k for k in unknown if str(k) in REMOVED_PROPERTY_KEYS]
        others = [k for k in unknown if str(k) not in REMOVED_PROPERTY_KEYS]

        faults: list[str] = []
        for spec_key in removed:
            replacement = REMOVED_PROPERTY_KEYS[str(spec_key)].value
            faults.append(f"'{spec_key}' was removed, rename it to '{replacement}' (DefaultOptionKeys.{replacement})")
        for spec_key in others:
            suggestion = difflib.get_close_matches(str(spec_key), known, n=1)
            hint = f", did you mean '{suggestion[0]}'?" if suggestion else ""
            faults.append(f"'{spec_key}' is unknown{hint}")

        raise ValueError(
            f"{owner_name}.PROPERTY_MAPPING['{key}'] has unknown spec key(s): {'; '.join(faults)}. "
            f"A spec may only carry the keys {known}. Accepted VALUES belong "
            f"under '{DefaultOptionKeys.allowed_values.value}'."
        )

    @classmethod
    def _reject_malformed_spec_values(cls, owner_name: str, key: str, spec: dict[str, Any]) -> None:
        """Reject a spec key whose VALUE has the wrong shape.

        ``allowed_values`` must never be a ``str``/``bytes``: ``in`` would silently become a
        SUBSTRING test. It must be a ``Collection`` at all: a generator is truthy even when empty
        and is consumed by the first read, and a scalar makes ``value in 5`` raise a ``TypeError``
        that the match path swallows into a silent reject-everything.
        """
        prefix = f"{owner_name}.PROPERTY_MAPPING['{key}']"

        flag_key = DefaultOptionKeys.strict_validation
        if flag_key in spec and not isinstance(spec[flag_key], bool):
            flag = spec[flag_key]
            raise ValueError(
                f"{prefix} sets '{flag_key.value}' to {flag!r} ({type(flag).__name__}), "
                f"which must be a real bool. A truthy non-bool silently enables strict validation."
            )

        if DefaultOptionKeys.allowed_values in spec:
            allowed_values = spec[DefaultOptionKeys.allowed_values]
            if isinstance(allowed_values, (str, bytes)):
                raise ValueError(
                    f"{prefix} declares '{DefaultOptionKeys.allowed_values.value}' as a "
                    f"{type(allowed_values).__name__} ({allowed_values!r}), which would make membership a "
                    f"SUBSTRING test. Wrap it in a container, for example a one-element tuple: "
                    f"({allowed_values!r},)."
                )
            if not isinstance(allowed_values, Collection):
                raise ValueError(
                    f"{prefix} declares '{DefaultOptionKeys.allowed_values.value}' as a "
                    f"{type(allowed_values).__name__}, which is not a Collection. A value space must be "
                    f"sized and re-iterable (a dict, tuple, list, set or frozenset); a generator is "
                    f"consumed by the first read and a scalar has no members."
                )

        for validator_key in (
            DefaultOptionKeys.element_validator,
            DefaultOptionKeys.required_when,
            DefaultOptionKeys.match_guard,
        ):
            if validator_key in spec and not callable(spec[validator_key]):
                value = spec[validator_key]
                raise ValueError(
                    f"{prefix} declares '{validator_key.value}' as a {type(value).__name__} "
                    f"({value!r}), which is not callable. It must be a callable taking the value "
                    f"and returning a bool."
                )

    @classmethod
    def _reject_strict_without_value_space(cls, owner_name: str, key: str, spec: dict[str, Any]) -> None:
        """Reject strict validation with nothing to validate against: it would reject every value.

        "Non-empty" is a ``len`` test rather than truthiness ON PURPOSE: the shape rule has
        already established the value space is a ``Collection``, and a generator is truthy even
        when it yields nothing.
        """
        if not cls._is_strict_validation(spec):
            return

        if cls._get_element_validator(spec) is not None:
            return

        allowed_values = spec.get(DefaultOptionKeys.allowed_values)
        if allowed_values is not None and len(allowed_values) > 0:
            return

        raise ValueError(
            f"{owner_name}.PROPERTY_MAPPING['{key}'] sets "
            f"{DefaultOptionKeys.strict_validation.value}=True but declares no value space, "
            f"so it would reject every value. Declare a non-empty "
            f"'{DefaultOptionKeys.allowed_values.value}' or an "
            f"'{DefaultOptionKeys.element_validator.value}'."
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
    def _collect_option_value(
        cls, options: Options, property_name: str, property_mapping: dict[str, Any]
    ) -> list[Any] | None:
        """Validate the VALUE of one option and return its elements, or None when it is absent.

        The per-key work both validation paths share, so they cannot drift. Raises
        PropertyValueRejection when a present value is outside the key's declared value space or
        is rejected by its element_validator.
        """
        found_property_value = options.get(property_name)
        if found_property_value is None:
            return None

        property_config = property_mapping[property_name]
        return cls._process_found_property_value(
            found_property_value, cls._extract_property_values(property_config), property_name, property_config
        )

    @classmethod
    def _validate_present_option_values(cls, options: Options, property_mapping: dict[str, Any]) -> None:
        """Validate the values of the options that are present, without enforcing presence.

        Used on the string-named path, where a key like the operation is carried by the feature
        name rather than by the options: an absent option is skipped instead of rejected. Raises
        PropertyValueRejection on a bad value.
        """
        for property_name in property_mapping:
            cls._collect_option_value(options, property_name, property_mapping)

    @classmethod
    def _validate_options_against_property_mapping(cls, options: Options, property_mapping: dict[str, Any]) -> bool:
        """Validate present option values AND enforce required presence (configuration-based path).

        Args:
            options: Options object containing the parameters to validate
            property_mapping: Property mapping with validation rules

        Returns:
            True if validation passes, False when a required option is absent

        Raises:
            PropertyValueRejection: If a present option carries a value the mapping rejects
        """
        # None marks an absent option; a list (possibly empty) marks a present one.
        property_tracker: dict[str, list[Any] | None] = {
            property_name: cls._collect_option_value(options, property_name, property_mapping)
            for property_name in property_mapping
        }
        return cls._validate_final_properties(property_tracker, property_mapping)

    @classmethod
    def match_configuration_feature_chain_parser(
        cls,
        feature_name: str | FeatureName,
        options: Options,
        property_mapping: Optional[dict[str, Any]] = None,
        prefix_patterns: Optional[list[Any]] = None,
        pattern: str = CHAIN_SEPARATOR,
    ) -> bool:
        """
        Unified method for matching features using either configuration-based or pattern-based parsing.

        Both paths validate the VALUES of the options that are present. They differ on required
        PRESENCE only: a name match satisfies the keys the name carries, so presence is enforced
        on the configuration-based path alone.

        This method RAISES on a rejected value, so a caller that overrides
        ``match_feature_group_criteria`` must not call it bare: use
        ``FeatureChainParserMixin.match_parser_criteria``, which turns the rejection into the
        non-match verdict the engine expects.

        Args:
            feature_name: The feature name to match
            options: Options object containing configuration
            property_mapping: Optional property mapping for configuration-based parsing
            prefix_patterns: Optional prefix patterns for pattern-based parsing
            pattern: Pattern string for pattern-based parsing (defaults to CHAIN_SEPARATOR)

        Returns:
            True if the feature matches either pattern-based or configuration-based parsing, False otherwise

        Raises:
            PropertyValueRejection: If a present option carries a value the property mapping rejects
            ValueError: If the feature name matches a prefix pattern but is malformed (no source
                feature), raised by parse_feature_name
        """

        # string based matching
        if prefix_patterns is not None:
            if cls._match_pattern_based_feature(feature_name, prefix_patterns, pattern):
                if property_mapping is not None:
                    cls._validate_present_option_values(options, property_mapping)
                return True

        # configuration-based
        if property_mapping is not None:
            return cls._validate_options_against_property_mapping(options, property_mapping)

        # If neither pattern-based nor configuration-based matching succeeded, return False
        return False

    @staticmethod
    def has_required_when_predicates(property_mapping: dict[str, Any]) -> bool:
        """Return True if any spec in property_mapping declares required_when."""
        for spec in property_mapping.values():
            if isinstance(spec, dict) and DefaultOptionKeys.required_when in spec:
                return True
        return False

    @classmethod
    def prefix_patterns_of(cls, owner: type[Any]) -> list[Any]:
        """Collect the name patterns a class matches on. The single implementation the mixin uses too.

        A pattern is whatever ``re.match`` accepts: a ``str`` or an already compiled ``re.Pattern``.
        Filtering by type would hide a compiled pattern from the guard while the matcher still matches
        on it, and the guard would then reject a feature the matcher accepted.
        """
        patterns: list[Any] = []
        for attribute in ("PREFIX_PATTERN", "SUFFIX_PATTERN"):
            pattern = getattr(owner, attribute, None)
            if pattern is not None:
                patterns.append(pattern)
        return patterns

    @classmethod
    def build_effective_options(
        cls,
        feature_name: str | FeatureName,
        prefix_patterns: list[Any],
        property_mapping: dict[str, Any],
        options: Options,
    ) -> Options:
        """Build effective options by merging string-parsed values with explicit options.

        When a feature is matched by string pattern, the operation_config value extracted
        from the feature name is mapped to the corresponding PROPERTY_MAPPING key. This
        ensures that required_when predicates see values from both sources.

        If the feature is not string-based or no mapping key matches, returns the
        original options unchanged.
        """
        # A matcher may parse the name with its own separator, so CHAIN_SEPARATOR can leave it
        # unparseable. That is no name-parsed value to merge, never an exception out of a matcher.
        nothing_parsed: tuple[str | None, str | None] = (None, None)
        operation_config, _source_feature = safe_field(
            lambda: cls.parse_feature_name(feature_name, prefix_patterns, CHAIN_SEPARATOR),
            nothing_parsed,
            catching=(ValueError,),
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
            if operation_config not in cls._extract_property_values(prop_value):
                continue
            category = cls._determine_parameter_category(prop_key, prop_value, options)
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
    def check_required_when(
        cls,
        owner_name: str,
        feature_name: str | FeatureName,
        prefix_patterns: list[Any],
        property_mapping: dict[str, Any] | None,
        options: Options,
    ) -> bool:
        """Evaluate every required_when predicate of a mapping. False means the feature is not a match."""
        if property_mapping is None or not cls.has_required_when_predicates(property_mapping):
            return True

        effective_options = cls.build_effective_options(feature_name, prefix_patterns, property_mapping, options)
        for key, spec in property_mapping.items():
            if not isinstance(spec, dict):
                continue
            predicate = spec.get(DefaultOptionKeys.required_when)
            if predicate is None:
                continue
            if not callable(predicate):
                logger.warning("required_when for '%s' in %s is not callable, skipping.", key, owner_name)
                continue
            if predicate(effective_options) and effective_options.get(key) is None:
                logger.debug(
                    "Feature group %s requires option '%s' (predicate %s is satisfied) but it was not provided.",
                    owner_name,
                    key,
                    getattr(predicate, "__name__", repr(predicate)),
                )
                return False
        return True

    @staticmethod
    def _resolve_match_arguments(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[str | FeatureName, Any]:
        """Recover (feature_name, options) from a matcher call without assuming an override's parameter names."""
        values = list(args) + list(kwargs.values())

        feature_name = kwargs.get("feature_name", args[0] if args else None)
        if not isinstance(feature_name, str):
            feature_name = next((value for value in values if isinstance(value, str)), "")

        options = kwargs.get("options")
        if not isinstance(options, Options):
            options = next((value for value in values if isinstance(value, Options)), None)

        return feature_name, options

    @classmethod
    def _reject_staticmethod_matcher(cls, owner: type[Any]) -> None:
        """Reject a staticmethod matcher on a class that declares required_when.

        The guard is reinstalled as a classmethod, so the class would be passed as the first
        positional argument: a staticmethod matcher would read ``cls`` as its ``feature_name`` and the
        feature name as its ``options``, and answer a silently wrong verdict. Fail at class definition.
        """
        for klass in owner.__mro__:
            descriptor = klass.__dict__.get("match_feature_group_criteria")
            if descriptor is None:
                continue
            if isinstance(descriptor, staticmethod):
                raise ValueError(
                    f"{owner.__name__} declares required_when in its PROPERTY_MAPPING, but its "
                    f"match_feature_group_criteria is a staticmethod. It must be a classmethod: the "
                    f"required_when guard is installed as a classmethod and passes the class as the first "
                    f"argument, which a staticmethod would misread as the feature name."
                )
            return

    @classmethod
    def install_required_when_guard(cls, owner: type[Any]) -> None:
        """Wrap a class's RESOLVED matcher so its required_when predicates run whatever matcher it kept.

        Called at class-definition time from ``FeatureGroup.__init_subclass__`` and
        ``FeatureChainParserMixin.__init_subclass__``. The predicates cannot live inside one
        matcher: overriding ``match_feature_group_criteria`` is supported, and an override that
        never delegates would silently drop the declared contract. The wrapper stays a
        classmethod, so it reads the PROPERTY_MAPPING and patterns of the class it is called on.

        A class that declares no required_when is left untouched, and an already guarded matcher
        is never wrapped again. Guards do nest (an override may delegate into a guarded parent), so
        only the outermost one evaluates the predicates: exactly once per match call.

        Class definition is the install site, so a PROPERTY_MAPPING mutated, or a matcher replaced,
        AFTER the class body is not seen by the guard.
        """
        property_mapping = getattr(owner, "PROPERTY_MAPPING", None)
        if not isinstance(property_mapping, dict) or not cls.has_required_when_predicates(property_mapping):
            return

        cls._reject_staticmethod_matcher(owner)

        matcher = getattr(owner, "match_feature_group_criteria", None)
        if matcher is None:
            return

        inner: Any = getattr(matcher, "__func__", matcher)
        if getattr(inner, REQUIRED_WHEN_GUARD_FLAG, False):
            return

        @functools.wraps(inner)
        def guarded(guarded_cls: type[Any], *args: Any, **kwargs: Any) -> bool:
            # The outermost guard is the one whose class the matcher was called on, so it is the one
            # whose PROPERTY_MAPPING decides. An inner guard, reached through a delegating super()
            # call, only answers with its matcher's verdict.
            outermost = REQUIRED_WHEN_GUARD_DEPTH.get() == 0
            token = REQUIRED_WHEN_GUARD_DEPTH.set(REQUIRED_WHEN_GUARD_DEPTH.get() + 1)
            try:
                if not inner(guarded_cls, *args, **kwargs):
                    return False

                if not outermost:
                    return True

                feature_name, options = FeatureChainParser._resolve_match_arguments(args, kwargs)
                if options is None:
                    return True

                return FeatureChainParser.check_required_when(
                    guarded_cls.__name__,
                    feature_name,
                    FeatureChainParser.prefix_patterns_of(guarded_cls),
                    getattr(guarded_cls, "PROPERTY_MAPPING", None),
                    options,
                )
            finally:
                REQUIRED_WHEN_GUARD_DEPTH.reset(token)

        setattr(guarded, REQUIRED_WHEN_GUARD_FLAG, True)
        setattr(owner, "match_feature_group_criteria", classmethod(guarded))

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
