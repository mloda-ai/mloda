"""
Feature chain parser for handling feature name chaining across feature groups.
"""

from __future__ import annotations

import contextvars
import functools
import logging
import re
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

# Marks a matcher that already carries the name-path presence guard, so it is never wrapped twice.
NAME_PATH_PRESENCE_GUARD_FLAG = "_mloda_name_path_presence_guard"

# Marks a class whose captureless diagnostic already ran, so the two __init_subclass__ hooks
# emit it at most once. Checked on the class's OWN dict so a subclass still evaluates fresh.
CAPTURELESS_DIAGNOSTIC_FLAG = "_mloda_captureless_diagnostic_emitted"

# An unrelated feature name used to probe whether a matcher is universal: does it accept a name it
# has no business matching, with empty options? It carries NO chain separator, so no
# PREFIX_PATTERN/SUFFIX_PATTERN can capture it and the resolved matcher falls through to the
# configuration path, where the universal-matcher problem actually lives.
_UNIVERSAL_MATCHER_PROBE_NAME = "mloda_universal_matcher_probe"

# How many guards the current match call is nested in. A guarded matcher that delegates via super()
# reaches the guard of its parent, and only the outermost one may evaluate the predicates.
# A ContextVar (not a plain global) keeps the count per thread and per async task.
REQUIRED_WHEN_GUARD_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "mloda_required_when_guard_depth", default=0
)

# Same nesting rule for the name-path presence guard, tracked independently so the two guards compose.
NAME_PATH_PRESENCE_GUARD_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "mloda_name_path_presence_guard_depth", default=0
)

# Active only while the engine evaluates candidates for one feature; maps owner class name to the first
# structured rejection reason the real match pass produced (os-005 replaces the diagnostic replay).
MATCH_REJECTION_REASONS: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "mloda_match_rejection_reasons", default=None
)

# Exception classes a user callable raises when it merely cannot judge a value.
_EXPECTED_JUDGMENT_ERRORS: tuple[type[Exception], ...] = (TypeError, ValueError, AttributeError)


def _contained_raise_log_level(exc: BaseException) -> int:
    """DEBUG for expected judgment failures, WARNING for classes that suggest a broken callable."""
    return logging.DEBUG if isinstance(exc, _EXPECTED_JUDGMENT_ERRORS) else logging.WARNING


def record_match_rejection(owner_name: str, reason: str) -> None:
    """Record a match rejection; the first reason per owner wins, and outside an active evaluation it is a no-op."""
    reasons = MATCH_REJECTION_REASONS.get()
    if reasons is None:
        return
    reasons.setdefault(owner_name, reason)


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
        exactly as ``_collect_option_value`` / ``check_required_when`` read absence.
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

        Diagnostic-only: mirrors _check_name_path_required_presence so the resolution-failure
        report explains the same non-match the matcher produced.
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
        # own parse containment; a raise out of build_effective_options in check_required_when now surfaces
        # as a framework defect (os-005, see TestBuildEffectiveOptionsRaiseSurfaces).
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
    def _pattern_named_and_total_groups(cls, pattern: Any) -> tuple[frozenset[str], int]:
        """The named group names and total group count of a pattern; an uncompilable pattern reports neither."""
        if isinstance(pattern, re.Pattern):
            return frozenset(pattern.groupindex), pattern.groups
        compiled: re.Pattern[str] | None = safe_field(lambda: re.compile(pattern), None, catching=(re.error, TypeError))
        if compiled is None:
            return frozenset(), 0
        return frozenset(compiled.groupindex), compiled.groups

    @classmethod
    def _flatten_patterns(cls, patterns: list[Any]) -> list[Any]:
        """Flatten one level so a list/tuple pattern attribute contributes its elements as concrete patterns.

        Mirrors how a ``SUFFIX_PATTERN = [regex]`` fixture passes the list straight to ``parse_name`` at
        runtime. A compiled ``re.Pattern`` and a ``str`` stay as-is.
        """
        flattened: list[Any] = []
        for pattern in patterns:
            if isinstance(pattern, (list, tuple)):
                flattened.extend(pattern)
            else:
                flattened.append(pattern)
        return flattened

    @classmethod
    def _str_reachable_values(cls, spec: PropertySpec) -> set[str]:
        """The str members of a spec's value space; only a str can be reverse-looked-up from a capture."""
        reachable: set[str] = set()
        for value in cls.extract_property_values(spec):
            if isinstance(value, str):
                reachable.add(value)
        return reachable

    @classmethod
    def validate_name_binding(cls, owner: type[Any]) -> None:
        """Reject an order-dependent legacy positional binding at class-definition time.

        The check is PER CONCRETE PATTERN: a list/tuple pattern attribute is flattened to its
        elements first. A pattern needs the overlap check only when it declares a capture group
        (``total >= 1``) AND no named group, so it relies on the legacy positional fallback. If any
        such positional-only pattern exists and two keys share a reachable (str) allowed value, the
        binding is order-dependent and rejected. A named-capture pattern is exempt (binding is
        explicit for it); a captureless one has nothing to misbind. Called from both FeatureGroup and
        FeatureChainParserMixin at class definition.
        """
        property_mapping = getattr(owner, "PROPERTY_MAPPING", None)
        if not isinstance(property_mapping, dict):
            return

        patterns = cls.prefix_patterns_of(owner)
        if not patterns:
            return

        needs_overlap_check = False
        for pattern in cls._flatten_patterns(patterns):
            named, total = cls._pattern_named_and_total_groups(pattern)
            if total >= 1 and not named:
                needs_overlap_check = True

        if not needs_overlap_check:
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
    def warn_captureless_without_binding(cls, owner: type[Any]) -> None:
        """Nudge authors of a captureless pattern that carries a PROPERTY_MAPPING (#772).

        A captureless pattern binds no key from the name. If a key must come from the name, add a
        named capture (?P<key>...); if the pattern is only a recognition predicate, set
        RECOGNITION_ONLY_PATTERN = True to declare that intent and silence this diagnostic.
        """
        if owner.__dict__.get(CAPTURELESS_DIAGNOSTIC_FLAG, False):
            return
        if getattr(owner, "RECOGNITION_ONLY_PATTERN", False):
            return
        property_mapping = getattr(owner, "PROPERTY_MAPPING", None)
        if not isinstance(property_mapping, dict) or not property_mapping:
            return
        patterns = cls.prefix_patterns_of(owner)
        if not patterns:
            return
        for pattern in cls._flatten_patterns(patterns):
            _named, total = cls._pattern_named_and_total_groups(pattern)
            if total == 0:
                setattr(owner, CAPTURELESS_DIAGNOSTIC_FLAG, True)
                logger.warning(
                    "%s declares a captureless PREFIX_PATTERN/SUFFIX_PATTERN together with a PROPERTY_MAPPING. "
                    "A captureless pattern binds no key from the feature name. Add a named capture "
                    "(?P<key>...) if a mapping key must be populated from the name, or set "
                    "RECOGNITION_ONLY_PATTERN = True to declare the pattern a recognition-only predicate "
                    "and silence this warning.",
                    owner.__name__,
                )
                return

    @classmethod
    def warn_universal_optional_matcher(cls, owner: type[Any]) -> None:
        """Nudge authors whose all-optional PROPERTY_MAPPING inherits the universal configuration matcher (#771).

        With zero unconditionally required keys, the configuration path matches any feature name given
        empty options. Warn unless the class opts in with ALLOW_UNIVERSAL_MATCHER = True. A key that is
        unconditionally required, or conditionally required via required_when, gates the match, so the
        mapping is not warned. Universality is confirmed behaviorally: the resolved matcher is called
        with an unrelated, separator-free name and empty options, which exempts a genuine custom matcher
        while still catching a pass-through override that delegates to the universal base.
        """
        if getattr(owner, "ALLOW_UNIVERSAL_MATCHER", False):
            return
        property_mapping = getattr(owner, "PROPERTY_MAPPING", None)
        # A None mapping is not a configuration matcher; an EMPTY dict is the strongest universal
        # matcher (it validates vacuously), so it stays in scope.
        if not isinstance(property_mapping, dict):
            return
        for spec in property_mapping.values():
            if not isinstance(spec, PropertySpec):
                continue
            # A required_when key gates the match with a runtime predicate, so the mapping is not a
            # blanket universal matcher. It is also left unprobed: the predicate may reference the
            # class being defined, which is not yet bound to its name during __init_subclass__, so
            # probing it would raise (#771).
            if spec.required_when is not None:
                return
            # A key that declares no default (and, per the check above, no required_when) is
            # unconditionally required and already discriminates on the configuration path.
            if not cls._can_skip_required_check(spec):
                return
        matcher = getattr(owner, "match_feature_group_criteria", None)
        if matcher is None:
            return
        # A matcher that raises on the probe is doing custom work, so it is not treated as universal.
        try:
            universal = bool(matcher(_UNIVERSAL_MATCHER_PROBE_NAME, Options()))
        except Exception as exc:
            err = exc  # rebind: Python clears the "except ... as exc" name at block exit, so the closure needs a stable local
            logger.debug(
                "universal-matcher probe for %s raised %s; treating it as non-universal.",
                owner.__name__,
                safe_field(lambda: str(err), type(err).__name__),
            )
            return
        if not universal:
            return
        logger.warning(
            "%s declares a PROPERTY_MAPPING with no unconditionally required key and inherits the "
            "universal configuration matcher: with empty options it matches any feature name. Add a "
            "required key (a PropertySpec with no default, or a required_when predicate that fires), or "
            "set ALLOW_UNIVERSAL_MATCHER = True to declare the universal match intentional.",
            owner.__name__,
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

        # build_effective_options runs no user callback, so a raise from it is a framework defect (or a user
        # configuration error carrying actionable guidance) and must surface, not read as a non-match (os-005, #763).
        effective_options = cls.build_effective_options(feature_name, prefix_patterns, property_mapping, options)
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
            if is_required and not option_key_is_present(spec, key, effective_options):
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
    def _matcher_is_staticmethod(cls, owner: type[Any]) -> bool:
        """True when the class's resolved matcher is a staticmethod descriptor."""
        for klass in owner.__mro__:
            descriptor = klass.__dict__.get("match_feature_group_criteria")
            if descriptor is not None:
                return isinstance(descriptor, staticmethod)
        return False

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
    def install_name_path_presence_guard(cls, owner: type[Any]) -> None:
        """Wrap a class's RESOLVED matcher so the name-path required-presence rule (#769) survives an override.

        Mirrors ``install_required_when_guard``: installed at class definition from both
        ``__init_subclass__`` hooks, never wrapped twice, and guards nest so only the outermost one
        evaluates. An inner False stands untouched, so the inner path's own presence warning is never
        duplicated. Nesting order relative to the required_when guard is behaviorally irrelevant:
        each guard ANDs its own predicate onto the inner verdict and passes False through unchanged.
        """
        property_mapping = getattr(owner, "PROPERTY_MAPPING", None)
        if not isinstance(property_mapping, dict):
            return
        if not cls.prefix_patterns_of(owner):
            return
        # Same exemptions as the inner rule: with empty options, the missing keys ARE the flaggable ones.
        if not cls._name_path_missing_required_keys(Options(), property_mapping):
            return

        # Wrapping a staticmethod matcher would hide it from _reject_staticmethod_matcher, so the
        # required_when installer's existing definition-time ValueError keeps precedence.
        is_static = cls._matcher_is_staticmethod(owner)
        if is_static and cls.has_required_when_predicates(property_mapping):
            return

        # getattr on a staticmethod returns the plain function, so the __func__ fetch below is a no-op
        # for it and the flag check covers both shapes.
        matcher = getattr(owner, "match_feature_group_criteria", None)
        if matcher is None:
            return

        inner: Any = getattr(matcher, "__func__", matcher)
        if getattr(inner, NAME_PATH_PRESENCE_GUARD_FLAG, False):
            return

        @functools.wraps(inner)
        def guarded(guarded_cls: type[Any], *args: Any, **kwargs: Any) -> bool:
            outermost = NAME_PATH_PRESENCE_GUARD_DEPTH.get() == 0
            token = NAME_PATH_PRESENCE_GUARD_DEPTH.set(NAME_PATH_PRESENCE_GUARD_DEPTH.get() + 1)
            try:
                # A staticmethod inner keeps its calling convention: no cls injected. An inner False
                # stands: the inner default path already warned on its own presence non-match, so
                # passing it through keeps one warning per match call.
                inner_verdict = inner(*args, **kwargs) if is_static else inner(guarded_cls, *args, **kwargs)
                if not inner_verdict:
                    return False

                if not outermost:
                    return True

                feature_name, options = FeatureChainParser._resolve_match_arguments(args, kwargs)
                if options is None:
                    return True

                mapping = getattr(guarded_cls, "PROPERTY_MAPPING", None)
                if not isinstance(mapping, dict):
                    return True
                # Flattened, because a matcher passes a list-valued pattern attribute straight to
                # parse_name, so its ELEMENTS are the concrete patterns. A matcher must never leak
                # an exception, so the parse is contained exactly as in build_effective_options.
                patterns = FeatureChainParser._flatten_patterns(FeatureChainParser.prefix_patterns_of(guarded_cls))
                parsed = safe_field(
                    lambda: FeatureChainParser.parse_name(feature_name, patterns, CHAIN_SEPARATOR),
                    ParsedFeatureName.no_match(),
                    catching=(ValueError,),
                )
                if not FeatureChainParser._name_identifies_group(parsed, mapping):
                    return True
                bindings = FeatureChainParser.bind_name_captures(parsed, mapping)
                effective_options = FeatureChainParser._merge_bindings(options, bindings, mapping)
                return FeatureChainParser._check_name_path_required_presence(
                    guarded_cls.__name__, feature_name, effective_options, mapping
                )
            finally:
                NAME_PATH_PRESENCE_GUARD_DEPTH.reset(token)

        setattr(guarded, NAME_PATH_PRESENCE_GUARD_FLAG, True)
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
