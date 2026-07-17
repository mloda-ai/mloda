"""
Feature chain parser for handling feature name chaining across feature groups.
"""

from __future__ import annotations

import contextvars
import functools
import logging
import re
from collections.abc import Callable
from typing import Any, Optional

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.core.abstract_plugins.components.feature_chainer.parsed_feature_name import ParsedFeatureName
from mloda.core.abstract_plugins.components.feature_chainer.property_spec import PropertySpec, is_no_default
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

# Exception classes a user callable raises when it merely cannot judge a value.
_EXPECTED_JUDGMENT_ERRORS: tuple[type[Exception], ...] = (TypeError, ValueError, AttributeError)


def _contained_raise_log_level(exc: BaseException) -> int:
    """DEBUG for expected judgment failures, WARNING for classes that suggest a broken callable."""
    return logging.DEBUG if isinstance(exc, _EXPECTED_JUDGMENT_ERRORS) else logging.WARNING


class PropertyValueRejection(ValueError):
    """An option value the PROPERTY_MAPPING rejects: a verdict, not a crash. Subclasses ValueError so
    existing ``except ValueError`` handlers keep working, while the distinct type lets a caller treat it as a
    non-match without also swallowing the ValueErrors that carry actionable guidance.
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
    def parse_name(
        cls,
        feature_name: FeatureName | str,
        prefix_patterns: list[Any],
        pattern: str = CHAIN_SEPARATOR,
    ) -> ParsedFeatureName:
        """Parse a feature name into structured facts, keeping today's matching semantics.

        A prefix pattern is anything ``re.match`` accepts: a ``str`` or a compiled ``re.Pattern``.
        A matched pattern with nothing before the separator raises the historical ValueError;
        ``match_parser_criteria`` and ``_strict_validation_rejection_reason`` depend on that raise.
        """
        _feature_name: str = feature_name

        parts = _feature_name.rsplit(pattern, 1)
        source_feature = parts[0] if len(parts) > 1 else ""
        operation_part = parts[1] if len(parts) > 1 else parts[0]

        for suffix_pattern in prefix_patterns:
            match = re.match(suffix_pattern, _feature_name)
            if match is None:
                continue

            if len(parts) == 1 or not source_feature:
                raise ValueError(f"Matches the pattern {pattern}, but has no source feature: {_feature_name}")

            return ParsedFeatureName(
                matched=True,
                source_feature=source_feature,
                operation_part=operation_part,
                named_captures=match.groupdict(),
                positional_captures=match.groups(),
            )

        return ParsedFeatureName.no_match()

    @classmethod
    def _legacy_operation_config(cls, parsed: ParsedFeatureName) -> str | None:
        """The value the retired reverse-lookup binding consumes. #772 removes the fabrication."""
        if parsed.positional_captures:
            return parsed.positional_captures[0]
        if parsed.operation_part is None:
            return None
        return parsed.operation_part.split("_")[0]  # fabrication, retired by #772

    @classmethod
    def parse_feature_name(
        cls,
        feature_name: FeatureName | str,
        prefix_patterns: list[Any],
        pattern: str = CHAIN_SEPARATOR,
    ) -> tuple[str | None, str | None]:
        """Legacy adapter over ``parse_name``: returns ``(operation_config, source_feature)``.

        Public API (mloda_plugins call sites and documented examples), so the tuple stays
        byte-for-byte identical to today, including the captureless fabrication and the ValueError.
        """
        parsed = cls.parse_name(feature_name, prefix_patterns, pattern)
        if not parsed.matched:
            return None, None
        return cls._legacy_operation_config(parsed), parsed.source_feature

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
    def _can_skip_required_check(cls, spec: PropertySpec) -> bool:
        """Check if the base parser should treat this property as optional.

        Returns True when the spec DECLARES a default (``NO_DEFAULT`` means it declares none,
        while a declared ``default=None`` marks the key optional with no value to apply) or uses
        conditional requirements (required_when). In both cases the base validation loop should
        not reject the match just because the option is absent; either the default will be applied
        later, or the required_when guard installed at class definition will decide.
        """
        return not is_no_default(spec.default) or spec.required_when is not None

    @classmethod
    def _is_context_parameter(cls, spec: PropertySpec) -> bool:
        """Check if the spec marks the property as a context parameter."""
        return spec.context

    @classmethod
    def _is_strict_validation(cls, spec: PropertySpec) -> bool:
        """Check if the spec requires strict validation (values must be in the value space)."""
        return spec.strict_validation

    @classmethod
    def _get_element_validator(cls, spec: PropertySpec) -> Callable[[Any], Any] | None:
        """Get the spec's per-element validator if present."""
        return spec.element_validator

    @classmethod
    def _validate_property_value(
        cls, found_property_val: Any, property_value: Any, property_name: str, original_property_config: PropertySpec
    ) -> None:
        """
        Unified validation: if strict validation -> apply the element_validator OR check membership.

        Raises PropertyValueRejection if validation fails, otherwise returns None.
        """
        if not cls._is_strict_validation(original_property_config):
            return  # No validation needed

        element_validator = cls._get_element_validator(original_property_config)

        if element_validator is not None:
            # A validator that raises cannot judge the value, so the value is rejected, not the run.
            raised: Exception | None = None
            try:
                verdict = element_validator(found_property_val)
            except Exception as exc:
                level = _contained_raise_log_level(exc)
                if level == logging.DEBUG:
                    logger.debug(
                        "element_validator for '%s' raised %s for value %r; treating value as rejected.",
                        property_name,
                        exc,
                        found_property_val,
                    )
                else:
                    # The raw value stays out of WARNING logs; rerun with debug logging to see it.
                    logger.warning(
                        "element_validator for '%s' raised %s; treating value as rejected.", property_name, exc
                    )
                raised = exc
                verdict = False
            if not verdict:
                raise PropertyValueRejection(
                    f"Property value '{found_property_val}' failed validation for '{property_name}'"
                ) from raised
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
    def _determine_parameter_category(cls, property_name: str, property_value: PropertySpec, options: Options) -> str:
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
    def extract_property_values(cls, spec: PropertySpec) -> Any:
        """Return a spec's declared value space (``allowed_values``), or {} if it declares none."""
        if spec.allowed_values is None:
            return {}
        return spec.allowed_values

    @classmethod
    def _extract_property_values(cls, spec: PropertySpec) -> Any:
        """Alias kept for existing callers."""
        return cls.extract_property_values(spec)

    @classmethod
    def _require_spec(cls, owner_name: str, key: str, spec: Any) -> PropertySpec:
        """Reject anything that is not a ``PropertySpec``.

        The parser entry point is public and takes a mapping straight from a caller, so the type
        rule cannot live at class-definition time alone: an unmigrated dict would otherwise die
        with an AttributeError deep in the match path instead of this ValueError.
        """
        if isinstance(spec, PropertySpec):
            return spec

        raise ValueError(
            f"{owner_name}.PROPERTY_MAPPING['{key}'] is a {type(spec).__name__}, not a PropertySpec. "
            f"Raw dict specs are no longer accepted; construct PropertySpec(...) or use the "
            f"property_spec(...) helper."
        )

    @classmethod
    def validate_property_mapping_defaults(cls, owner_name: str, property_mapping: dict[str, Any] | None) -> None:
        """Validate a PROPERTY_MAPPING at class-definition time.

        Every spec must BE a ``PropertySpec``; a spec validates itself at construction, so the
        only rule left here is the type itself.
        """
        if property_mapping is None:
            return

        for key, spec in property_mapping.items():
            cls._require_spec(owner_name, key, spec)

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
        cls, found_property_value: Any, property_value: Any, property_name: str, original_property_config: PropertySpec
    ) -> list[Any]:
        collected_property_value: list[Any] = []
        for found_property_val in cls._unpack_property_value(found_property_value):
            # Use unified validation function
            cls._validate_property_value(found_property_val, property_value, property_name, original_property_config)

            collected_property_value.append(found_property_val)

        return collected_property_value

    @classmethod
    def _validate_final_properties(
        cls, property_tracker: dict[str, list[Any] | None], property_mapping: dict[str, PropertySpec]
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
        cls, options: Options, property_name: str, property_mapping: dict[str, PropertySpec]
    ) -> list[Any] | None:
        """Validate one option value and return its elements, or None when the option is absent."""
        property_config = property_mapping[property_name]
        found_property_value = options.get(property_name)
        # An opted-in spec treats a present-as-None value as PRESENT, so it flows through validation (#768).
        if found_property_value is None and not (property_config.allow_explicit_none and property_name in options):
            return None
        return cls._process_found_property_value(
            found_property_value, cls._extract_property_values(property_config), property_name, property_config
        )

    @classmethod
    def _validate_present_option_values(cls, options: Options, property_mapping: dict[str, PropertySpec]) -> None:
        """Validate the values of the present options, without enforcing presence of the absent ones."""
        for property_name, spec in property_mapping.items():
            # The entry point is public: a caller may hand over an unmigrated mapping that never
            # passed the class-definition check.
            cls._require_spec(cls.__name__, property_name, spec)

        for property_name in property_mapping:
            cls._collect_option_value(options, property_name, property_mapping)

    @classmethod
    def _validate_options_against_property_mapping(
        cls, options: Options, property_mapping: dict[str, PropertySpec]
    ) -> bool:
        """Validate present option values and enforce required presence. False when a required option is absent.

        Raises:
            PropertyValueRejection: If a present option carries a value the mapping rejects
        """
        for key, spec in property_mapping.items():
            # The entry point is public: a caller may hand over an unmigrated mapping that never
            # passed the class-definition check.
            cls._require_spec(cls.__name__, key, spec)

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
        property_mapping: Optional[dict[str, PropertySpec]] = None,
        prefix_patterns: Optional[list[Any]] = None,
        pattern: str = CHAIN_SEPARATOR,
    ) -> bool:
        """
        Unified method for matching features using either configuration-based or pattern-based parsing.

        Both paths validate the values of the present options; only required presence is enforced on the
        configuration-based path alone. This raises on a rejected value, so an overridden
        ``match_feature_group_criteria`` must reach it through ``FeatureChainParserMixin.match_parser_criteria``.

        Args:
            feature_name: The feature name to match
            options: Options object containing configuration
            property_mapping: Optional property mapping for configuration-based parsing
            prefix_patterns: Optional prefix patterns for pattern-based parsing
            pattern: Pattern string for pattern-based parsing (defaults to CHAIN_SEPARATOR)

        Returns:
            True if the feature matches either pattern-based or configuration-based parsing, False otherwise
        """

        # string based matching. parse_name raises the no-source ValueError exactly as before, contained by
        # match_parser_criteria. Effective options are built from the parse facts here, NOT via the
        # monkeypatchable build_effective_options: a build raise stays owned by check_required_when, the one
        # containment site that knows the owner (see build_effective_options / TestBuildEffectiveOptionsRaiseIsContained).
        if prefix_patterns is not None:
            parsed = cls.parse_name(feature_name, prefix_patterns, pattern)
            if parsed.matched:
                if property_mapping is not None:
                    bindings = cls.bind_name_captures(parsed, property_mapping)
                    effective_options = cls._merge_bindings(options, bindings, property_mapping)
                    cls._validate_present_option_values(effective_options, property_mapping)
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
            if isinstance(spec, PropertySpec) and spec.required_when is not None:
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
    def bind_name_captures(cls, parsed: ParsedFeatureName, property_mapping: dict[str, Any]) -> dict[str, str]:
        """Turn parse facts into PROPERTY_MAPPING bindings by name; documented and deterministic.

        Named captures bind EXCLUSIVELY by name: a capture whose name is a mapping key binds to that
        key, an unmapped name binds nothing, a non-participating (None) capture binds nothing. Only
        when the matched pattern declares no named capture at all does the legacy positional fallback
        bind ``_legacy_operation_config`` to the single key whose ``allowed_values`` already contain
        it (transitional compatibility for unmigrated positional patterns; retired by #772 /
        mloda-registry#327). The fallback binds only a value already accepted, so it never fails strict
        validation.
        """
        if not parsed.matched:
            return {}

        if parsed.named_captures:
            bindings: dict[str, str] = {}
            for name, value in parsed.named_captures.items():
                if value is None or name not in property_mapping:
                    continue
                bindings[name] = value
            return bindings

        legacy_value = cls._legacy_operation_config(parsed)
        if legacy_value is None:
            return {}
        for key, spec in property_mapping.items():
            if not isinstance(spec, PropertySpec):
                continue
            if legacy_value in cls._extract_property_values(spec):
                return {key: legacy_value}
        return {}

    @classmethod
    def _merge_bindings(
        cls, options: Options, bindings: dict[str, str], property_mapping: dict[str, Any] | None
    ) -> Options:
        """Merge name-derived bindings into options; a present option wins, nothing to merge is identity.

        Provenance (inherited_group_keys / inherited_context_keys) and propagate_context_keys survive
        the rebuild, so forwarded-mismatch protection still reads it off the effective options.
        """
        if property_mapping is None or not bindings:
            return options

        merged_group = dict(options.group)
        merged_context = dict(options.context)
        changed = False
        for key, value in bindings.items():
            spec = property_mapping.get(key)
            if not isinstance(spec, PropertySpec):
                continue
            # An explicit option (including an opted-in explicit None, #768) is never overwritten.
            if options.get(key) is not None or (spec.allow_explicit_none and key in options):
                continue
            if cls._determine_parameter_category(key, spec, options) == DefaultOptionKeys.context:
                merged_context[key] = value
            else:
                merged_group[key] = value
            changed = True

        if not changed:
            return options

        effective = Options(
            group=merged_group,
            context=merged_context,
            propagate_context_keys=options.propagate_context_keys,
        )
        effective.inherited_group_keys = options.inherited_group_keys
        effective.inherited_context_keys = options.inherited_context_keys
        effective.last_forwarded_group_keys = options.last_forwarded_group_keys
        return effective

    @classmethod
    def build_effective_options(
        cls,
        feature_name: str | FeatureName,
        prefix_patterns: list[Any],
        property_mapping: dict[str, Any],
        options: Options,
    ) -> Options:
        """Merge every name-derived binding into options so predicates and validation see them.

        Binding is by name (``bind_name_captures``): all captures merge at once, not just the first key.
        A matcher may parse the name with its own separator, so CHAIN_SEPARATOR can leave it unparseable;
        that is no name-parsed value to merge, never an exception out of a matcher. If nothing matches or
        nothing binds, the original options come back by identity.
        """
        parsed = safe_field(
            lambda: cls.parse_name(feature_name, prefix_patterns, CHAIN_SEPARATOR),
            ParsedFeatureName.no_match(),
            catching=(ValueError,),
        )
        if not parsed.matched:
            return options
        bindings = cls.bind_name_captures(parsed, property_mapping)
        return cls._merge_bindings(options, bindings, property_mapping)

    @classmethod
    def _pattern_named_and_total_groups(cls, pattern: Any) -> tuple[frozenset[str], int]:
        """The named group names and total group count of a pattern; an uncompilable pattern reports neither."""
        if isinstance(pattern, re.Pattern):
            return frozenset(pattern.groupindex), pattern.groups
        compiled: re.Pattern[str] | None = safe_field(lambda: re.compile(pattern), None, catching=(re.error, TypeError))
        if compiled is None:
            return frozenset(), 0
        return frozenset(compiled.groupindex), compiled.groups

    @classmethod
    def _str_reachable_values(cls, spec: PropertySpec) -> set[str]:
        """The str members of a spec's value space; only a str can be reverse-looked-up from a capture."""
        reachable: set[str] = set()
        for value in cls._extract_property_values(spec):
            if isinstance(value, str):
                reachable.add(value)
        return reachable

    @classmethod
    def validate_name_binding(cls, owner: type[Any]) -> None:
        """Reject an order-dependent legacy positional binding at class-definition time.

        Fires only when the class relies on the legacy fallback: it declares a capture group, no
        pattern declares a named group, and two keys share a reachable (str) allowed value. Named
        captures make binding explicit, so they are exempt; a captureless pattern has nothing to
        misbind. Called from both FeatureGroup and FeatureChainParserMixin at class definition.
        """
        property_mapping = getattr(owner, "PROPERTY_MAPPING", None)
        if not isinstance(property_mapping, dict):
            return

        patterns = cls.prefix_patterns_of(owner)
        if not patterns:
            return

        has_capture_group = False
        declares_named = False
        for pattern in patterns:
            named, total = cls._pattern_named_and_total_groups(pattern)
            has_capture_group = has_capture_group or total >= 1
            declares_named = declares_named or bool(named)

        if not has_capture_group or declares_named:
            return

        reachable = {
            key: cls._str_reachable_values(spec)
            for key, spec in property_mapping.items()
            if isinstance(spec, PropertySpec)
        }
        keys = list(reachable)
        for i, left in enumerate(keys):
            for right in keys[i + 1 :]:
                overlap = reachable[left] & reachable[right]
                if overlap:
                    raise ValueError(
                        f"{owner.__name__}: PROPERTY_MAPPING keys '{left}' and '{right}' share reachable "
                        f"allowed value(s) {sorted(overlap)}, so a legacy positional capture cannot bind "
                        f"unambiguously. Use named capture groups (?P<key>...) so binding is explicit."
                    )

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

        # Building effective options may itself raise, so a raise here is contained as a non-match too.
        try:
            effective_options = cls.build_effective_options(feature_name, prefix_patterns, property_mapping, options)
        except Exception as exc:
            logger.log(
                _contained_raise_log_level(exc),
                "building effective options for required_when on %s raised %s; treating feature group as a non-match.",
                owner_name,
                exc,
            )
            return False
        for key, spec in property_mapping.items():
            if not isinstance(spec, PropertySpec):
                continue
            # Callability is enforced at PropertySpec construction, so a present predicate is callable.
            predicate = spec.required_when
            if predicate is None:
                continue
            # A predicate that raises cannot judge the value, so the feature group is a non-match, not the run.
            try:
                is_required = bool(predicate(effective_options))
            except Exception as exc:
                logger.log(
                    _contained_raise_log_level(exc),
                    "required_when predicate %s for '%s' raised %s; treating feature group %s as a non-match.",
                    getattr(predicate, "__name__", repr(predicate)),
                    key,
                    exc,
                    owner_name,
                )
                return False
            # An opted-in key present as an explicit None counts as present, so the requirement is met (#768).
            if (
                is_required
                and effective_options.get(key) is None
                and not (spec.allow_explicit_none and key in effective_options)
            ):
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
