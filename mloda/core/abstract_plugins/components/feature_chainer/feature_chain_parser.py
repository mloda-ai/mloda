"""
Feature chain parser for handling feature name chaining across feature groups.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.match_rejection import record_match_rejection
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.core.abstract_plugins.components.feature_chainer.parsed_feature_name import ParsedFeatureName
from mloda.core.abstract_plugins.components.feature_chainer.property_spec import PropertySpec, is_no_default
from mloda.core.abstract_plugins.components.utils import (
    contained_raise_log_level,
    contained_raise_reason,
    escalate_match_abort,
    safe_field,
)

logger = logging.getLogger(__name__)

# Separator constants for feature name parsing
CHAIN_SEPARATOR = "__"  # Separates chained transformations (source→suffix)
COLUMN_SEPARATOR = "~"  # Separates multi-column output index
INPUT_SEPARATOR = "&"  # Separates multiple input features


def option_key_is_present(spec: PropertySpec, key: str, options: Options) -> bool:
    """The single presence decision (#768 matrix): an opted-in explicit None counts as present, a flagless
    present-as-None does not."""
    if spec.allow_explicit_none:
        return key in options
    return options.get(key) is not None


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
        ``match_parser_criteria`` and the mixin's standalone rejection diagnostic depend on that raise.
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
                # Contained: a matched pattern with no source feature is this parser's own name verdict.
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
        """The value the legacy positional reverse-lookup binding consumes: the first positional
        capture, or None. A captureless match fabricates nothing (#772)."""
        if parsed.positional_captures:
            return parsed.positional_captures[0]
        return None

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
    def _validate_property_value(
        cls, found_property_val: Any, property_value: Any, property_name: str, original_property_config: PropertySpec
    ) -> None:
        """
        Unified validation: if strict validation -> apply the element_validator OR check membership.

        Raises PropertyValueRejection if validation fails, otherwise returns None.
        """
        if not original_property_config.strict_validation:
            return  # No validation needed

        element_validator = original_property_config.element_validator

        if element_validator is not None:
            raised: Exception | None = None
            try:
                verdict = element_validator(found_property_val)
            except Exception as exc:  # Swallows: a validator that raises cannot judge the value, so it is rejected.
                # Text, not exc: a retained record must not pin the traceback, its frames and the plugin class.
                level = contained_raise_log_level(exc)
                if level == logging.DEBUG:
                    logger.debug(
                        "element_validator for '%s' %s for value %r; treating value as rejected.",
                        property_name,
                        contained_raise_reason(exc),
                        found_property_val,
                    )
                else:
                    # The raw value stays out of WARNING logs; rerun with debug logging to see it.
                    logger.warning(
                        "element_validator for '%s' %s; treating value as rejected.",
                        property_name,
                        contained_raise_reason(exc),
                    )
                raised = exc
                verdict = False
            if not verdict:
                # Contained: a rejected option value is this candidate's own verdict, recorded as its reason.
                raise PropertyValueRejection(
                    f"Property value '{found_property_val}' failed validation for '{property_name}'"
                ) from raised
        else:
            # Fallback to membership check.
            try:
                is_member = found_property_val in property_value
            # Swallows: an unhashable element can never be a member, so the TypeError is a clean rejection.
            except TypeError:
                is_member = False
            if not is_member:
                # Contained: a rejected option value is this candidate's own verdict, recorded as its reason.
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
            # Marked: Options construction already rejects a key in both categories, so this is a broken invariant.
            raise escalate_match_abort(
                ValueError(
                    f"Parameter '{property_name}' exists in both group and context. "
                    "This is not allowed. Please choose one category."
                )
            )

        if property_name in options.group:
            return DefaultOptionKeys.group
        elif property_name in options.context:
            return DefaultOptionKeys.context
        elif property_value.context:
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
    def _require_spec(cls, owner_name: str, key: str, spec: Any) -> PropertySpec:
        """Reject anything that is not a reader-free ``PropertySpec``; the parser entry point is
        public and takes caller mappings, so neither rule can live at class-definition time alone."""
        if not isinstance(spec, PropertySpec):
            # Contained: a raw dict spec is that candidate's own defect, so the seam reads it as a non-match.
            raise ValueError(
                f"{owner_name}.PROPERTY_MAPPING['{key}'] is a {type(spec).__name__}, not a PropertySpec. "
                f"Raw dict specs are no longer accepted; construct PropertySpec(...) or use the "
                f"property_spec(...) helper."
            )
        if spec.framework_set:
            # Contained: a framework_set spec is that candidate's own defect, so the seam reads it as a non-match.
            raise ValueError(
                f"{owner_name}.PROPERTY_MAPPING['{key}'] declares framework_set=True, which marks a "
                f"reader-surface (BaseInputData PROPERTY_MAPPING) key written by the framework; a "
                f"FeatureGroup's PROPERTY_MAPPING keys are user-set."
            )
        if spec.scalar_only:
            # Contained: a scalar_only spec is that candidate's own defect, so the seam reads it as a non-match.
            raise ValueError(
                f"{owner_name}.PROPERTY_MAPPING['{key}'] declares scalar_only=True, which marks a "
                f"reader-surface (BaseInputData PROPERTY_MAPPING) key rejected outright as a collection; a "
                f"FeatureGroup's PROPERTY_MAPPING keys always unpack element-wise."
            )
        return spec

    @classmethod
    def validate_property_mapping_defaults(cls, owner_name: str, property_mapping: dict[str, Any] | None) -> None:
        """Validate a PROPERTY_MAPPING at class-definition time; the rules live in ``_require_spec``,
        shared with the match-time entry points."""
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
        if not option_key_is_present(property_config, property_name, options):
            return None
        return cls._process_found_property_value(
            found_property_value, cls.extract_property_values(property_config), property_name, property_config
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
    def _name_path_missing_required_keys(
        cls, effective_options: Options, property_mapping: dict[str, PropertySpec]
    ) -> list[str]:
        """The missing required keys on the name path (#769).

        The source key is name-provided (its count is enforced by MIN/MAX_IN_FEATURES), so
        ``in_features`` is excluded. A declared-default or ``required_when`` key is skippable
        (``_can_skip_required_check``); ``deferred_binding`` is the #769 opt-out. A key is absent
        exactly as ``_collect_option_value`` / ``feature_chain_author_guards.check_required_when`` read absence.
        """
        missing: list[str] = []
        for key, spec in property_mapping.items():
            if not isinstance(spec, PropertySpec):
                continue
            if key == DefaultOptionKeys.in_features:
                continue
            if cls._can_skip_required_check(spec):
                continue
            if spec.deferred_binding:
                continue
            absent = not option_key_is_present(spec, key, effective_options)
            if absent:
                missing.append(key)
        return missing

    @staticmethod
    def _presence_rejection_reason(missing: list[str]) -> str:
        """The one formatting of the missing-required-keys reason, shared by the matcher and the diagnostic."""
        return f"required option(s) {', '.join(sorted(missing))} are absent after declared defaults and name bindings"

    @classmethod
    def _check_name_path_required_presence(
        cls,
        owner_name: str | None,
        feature_name: str | FeatureName,
        effective_options: Options,
        property_mapping: dict[str, PropertySpec],
    ) -> bool:
        """Enforce the name-path required-presence rule (#769). False means non-match."""
        missing = cls._name_path_missing_required_keys(effective_options, property_mapping)
        if not missing:
            return True

        if owner_name is not None:
            record_match_rejection(owner_name, cls._presence_rejection_reason(missing))

        owner = owner_name or "A feature group"
        keys = ", ".join(sorted(missing))
        logger.warning(
            "%s did not match feature '%s': required option(s) %s are absent after declared defaults and "
            "name bindings. Provide the option(s), add a named capture (?P<key>...), or set "
            "deferred_binding=True on each key bound outside the name.",
            owner,
            feature_name,
            keys,
        )
        return False

    @classmethod
    def name_path_presence_rejection_reason(
        cls, effective_options: Options, property_mapping: dict[str, PropertySpec]
    ) -> str | None:
        """The reason a name-path candidate was rejected for missing presence (#769); None when nothing is missing.

        Supported diagnostic seam, paired with ``_strict_validation_rejection_reason``:
        mirrors _check_name_path_required_presence so the resolution-failure report explains the
        same non-match the matcher produced.
        """
        missing = cls._name_path_missing_required_keys(effective_options, property_mapping)
        if not missing:
            return None
        return cls._presence_rejection_reason(missing)

    @classmethod
    def match_configuration_feature_chain_parser(
        cls,
        feature_name: str | FeatureName,
        options: Options,
        property_mapping: Optional[dict[str, PropertySpec]] = None,
        prefix_patterns: Optional[list[Any]] = None,
        pattern: str = CHAIN_SEPARATOR,
        owner_name: str | None = None,
    ) -> bool:
        """
        Unified method for matching features using either configuration-based or pattern-based parsing.

        Both paths validate the values of the present options and enforce required presence; the
        string-named path resolves declared defaults and name bindings first (#769). This raises on a
        rejected value, so an overridden ``match_feature_group_criteria`` must reach it through
        ``FeatureChainParserMixin.match_parser_criteria``.

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
        # match_parser_criteria. Effective options are built from the parse facts here, keeping the matcher's
        # own parse containment; a raise out of build_effective_options in the author guards'
        # check_required_when now surfaces as a framework defect (see TestBuildEffectiveOptionsRaiseSurfaces).
        if prefix_patterns is not None:
            parsed = cls.parse_name(feature_name, prefix_patterns, pattern)
            if cls._name_identifies_group(parsed, property_mapping):
                if property_mapping is not None:
                    bindings = cls.bind_name_captures(parsed, property_mapping)
                    effective_options = cls._merge_bindings(options, bindings, property_mapping)
                    cls._validate_present_option_values(effective_options, property_mapping)
                    if not cls._check_name_path_required_presence(
                        owner_name, feature_name, effective_options, property_mapping
                    ):
                        return False
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
            if legacy_value in cls.extract_property_values(spec):
                return {key: legacy_value}
        return {}

    @classmethod
    def _name_identifies_group(cls, parsed: ParsedFeatureName, property_mapping: dict[str, Any] | None) -> bool:
        """True when a matched name string-identifies this group for matching.

        A legacy positional pattern whose only capture is an absent optional first group does NOT
        identify the group: reproduce the pre-#770 gate so required presence still guards it on the
        config path (#769 owns changing that). A named capture that binds a mapping key identifies the
        group even when the legacy operation value is absent, so a named-optional-first pattern gets
        full binding, guard, and forwarded-mismatch visibility. A captureless match is a recognition
        predicate (#772): it identifies the group and binds nothing.
        """
        if not parsed.matched:
            return False
        if property_mapping and cls.bind_name_captures(parsed, property_mapping):
            return True
        if not parsed.positional_captures:
            # Captureless pattern: zero declared groups. It identifies the group as a recognition
            # predicate and binds nothing. #772 stopped fabricating a token here.
            return True
        # A positional group that did not participate (optional-first) still does not identify;
        # #769 owns changing that.
        return cls._legacy_operation_config(parsed) is not None

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
            if option_key_is_present(spec, key, options):
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
