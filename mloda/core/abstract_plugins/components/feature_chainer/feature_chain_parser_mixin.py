"""
Mixin class providing default implementations for feature chain parsing.

Validation Design: ``element_validator`` vs ``match_guard``
==========================================================

``PropertySpec`` carries two callable-valued fields that validate option values.
They serve different purposes and run at different points in the pipeline.

``element_validator`` (``PropertySpec.element_validator``)
  - Requires ``strict_validation=True`` on the same spec.
  - Runs inside ``FeatureChainParser._validate_property_value`` on **both** match paths,
    the configuration-based one and the string-named one. Required *presence* holds on both
    too; a key encoded in the feature name is satisfied by the name binding.
  - Receives **individual parsed elements** after list unpacking
    (``_process_found_property_value`` unpacks a sequence value into a list and
    iterates over each element).
  - On failure: raises ``PropertyValueRejection`` (a ``ValueError``) with an actionable
    message identifying the property name and the rejected element value. A validator that
    raises rather than returning falsy cannot judge the value and is a rejection too.
  - Use case: validating that each individual element satisfies a constraint
    (e.g., ``lambda x: isinstance(x, int) and x > 0``).

``match_guard`` (``PropertySpec.match_guard``)
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

When both are present on the same spec, ``element_validator`` runs
first (during property mapping validation) on each parsed element, then
``match_guard`` runs on the raw value. If ``element_validator`` rejects an
element, the match fails with a ``ValueError`` before ``match_guard`` is
reached.

A guard rejection on a spec that also sets ``strict_validation=True`` is
reportable: the match pass records it as it happens, and the recorded reason
feeds the resolution-failure report. ``_strict_validation_rejection_reason``
remains a standalone diagnostic facade producing the same message. A guard on
a non-strict spec keeps its "not mine" meaning and reports nothing.

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
    PropertyValueRejection,
    _contained_raise_log_level,
    option_key_is_present,
    record_match_rejection,
)
from mloda.core.abstract_plugins.components.feature_chainer.property_spec import PropertySpec
from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.core.abstract_plugins.components.utils import safe_field

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
    - RECOGNITION_ONLY_PATTERN: Optional marker for a recognition-only pattern that binds no key
      from the name (all values come from options); default False (#772)
    - ALLOW_UNIVERSAL_MATCHER: Optional opt-in marking an all-optional PROPERTY_MAPPING as an
      intentional universal configuration matcher; default False (#771)

    PROPERTY_MAPPING supports conditional requirements via ``PropertySpec.required_when``.
    Attach a predicate ``(Options) -> bool`` to any spec. When the predicate returns
    True and the option value is absent, ``match_feature_group_criteria`` rejects the match.
    When the predicate returns False, the option is treated as optional. The predicates are
    enforced by a guard installed on the class at definition time (see
    ``FeatureChainParser.install_required_when_guard``), so overriding
    ``match_feature_group_criteria`` keeps the contract.

    This works for both string-based and configuration-based feature creation. For
    string-based features, the operation value parsed from the feature name is merged
    into effective options before predicate evaluation, so predicates see values from
    both the feature name and explicit options.

    Predicate contract:
    - Signature: ``(Options) -> bool``
    - Must be callable (enforced at ``PropertySpec`` construction)
    - A predicate that raises is contained: the feature group is treated as a non-match
      (unexpected exception classes log a warning)
    - Must be a pure function (no side effects)
    - Non-bool truthy return values are treated as True

    See docs/in_depth/property-mapping.md for full details and examples.
    """

    IN_FEATURE_SEPARATOR: str = INPUT_SEPARATOR
    MIN_IN_FEATURES: int = 1
    MAX_IN_FEATURES: Optional[int] = None
    # A recognition-only pattern binds no key from the name; all values come from options (#772).
    RECOGNITION_ONLY_PATTERN: bool = False
    # An all-optional PROPERTY_MAPPING that inherits the config matcher matches any feature name with
    # empty options; set True to declare that universal match intentional and silence the #771 warning.
    ALLOW_UNIVERSAL_MATCHER: bool = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # The mixin sits first in the MRO of ``class X(FeatureChainParserMixin, FeatureGroup)``,
        # so super() is what lets FeatureGroup's own class-definition validation still run.
        super().__init_subclass__(**kwargs)
        FeatureChainParser.validate_name_binding(cls)
        FeatureChainParser.warn_captureless_without_binding(cls)
        FeatureChainParser.install_name_path_presence_guard(cls)
        FeatureChainParser.install_required_when_guard(cls)
        FeatureChainParser.warn_universal_optional_matcher(cls)

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
        property_mapping = self._get_property_mapping()
        parsed = FeatureChainParser.parse_name(feature_name, prefix_patterns, CHAIN_SEPARATOR)

        # The name is authoritative only when it identifies this group (a captureless recognition match
        # or a participating capture). An optional-first positional group that did not participate does
        # not identify the group, so its source comes from options, not the name (#772 / #769).
        if FeatureChainParser._name_identifies_group(parsed, property_mapping) and parsed.source_feature:
            in_features = parsed.source_feature.split(self.IN_FEATURE_SEPARATOR)
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

        Delegates to match_parser_criteria() and optionally calls _validate_string_match() for custom validation.

        After basic matching succeeds, enforces ``match_guard`` constraints from
        PROPERTY_MAPPING entries. For each spec that defines a
        ``PropertySpec.match_guard`` callable, the guard is called with the raw
        option value. Returning a falsy value causes this method to return False. If
        the guard raises an exception, it is caught and the value is treated as
        invalid. See the module docstring for the full validation design.

        Also enforces MIN_IN_FEATURES / MAX_IN_FEATURES constraints when
        in_features is present in options.

        ``required_when`` is NOT evaluated here. The guard installed at class definition
        runs the predicates after this method (or any override of it) returns True.

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

        result = cls.match_parser_criteria(feature_name, options)

        # On a string match the guards see the EFFECTIVE options (name-derived bindings merged in), so a
        # name-carried value is as visible to match_guard as an explicit one. A no-source ValueError cannot
        # reach here: it would already have made match_parser_criteria a non-match.
        effective_options = options
        if result:
            parsed = FeatureChainParser.parse_name(feature_name, prefix_patterns, CHAIN_SEPARATOR)
            if FeatureChainParser._name_identifies_group(parsed, property_mapping):
                # Bound once and reused: the merge and the guards must see the name-derived value even
                # when the legacy operation value is absent (a named-optional-first pattern).
                bindings = FeatureChainParser.bind_name_captures(parsed, property_mapping or {})
                operation_config = FeatureChainParser._legacy_operation_config(parsed)
                # _validate_string_match needs a str operation, so it stays behind its own gate; the
                # merge and forwarded-mismatch protection do not.
                if operation_config is not None and parsed.source_feature is not None:
                    if not cls._validate_string_match(feature_name, operation_config, parsed.source_feature):
                        return False
                effective_options = FeatureChainParser._merge_bindings(options, bindings, property_mapping)
                cls._validate_forwarded_name_mismatch(feature_name, bindings, options)

        if not cls._validate_match_guards(result, effective_options, property_mapping):
            return False

        if not cls._validate_in_features(result, options):
            return False

        return result

    @classmethod
    def match_parser_criteria(cls, feature_name: str | FeatureName, options: Options) -> bool:
        """Call the parser, turning a rejected option value or a malformed name into a non-match, never an exception.

        The only safe way to reach the parser from an overridden ``match_feature_group_criteria``: an exception out
        of a match hook aborts the identification of the feature for every candidate, not just this one.
        """
        try:
            return FeatureChainParser.match_configuration_feature_chain_parser(
                feature_name,
                options,
                property_mapping=cls._get_property_mapping(),
                prefix_patterns=cls._get_prefix_patterns(),
                owner_name=cls.__name__,
            )
        except PropertyValueRejection as exc:
            record_match_rejection(cls.__name__, str(exc))
            return False
        # Known asymmetry deliberately kept in scope: a config error raised while merging name bindings on the
        # string path (e.g. a key in both group and context) is still contained as a non-match here, while the
        # required_when path surfaces it via build_effective_options (os-005 review note).
        except ValueError:
            return False

    @classmethod
    def _strict_validation_rejection_reason(cls, feature_name: str | FeatureName, options: Options) -> str | None:
        """Return the rejection message that match_feature_group_criteria discards, if any.

        The engine no longer calls this: it renders the reasons the first match pass recorded via
        ``record_match_rejection``. This is a supported diagnostic seam: a stable,
        overridable hook for reproducing a single group's value-rejection reason outside a run. It
        must keep producing the same messages the match pass records.

        Reports two kinds of value rejection, both gated on the feature group OTHERWISE matching
        the feature:

        1. A ValueError raised by option-value validation (a strict_validation rejection). Present
           option values are validated on both match paths, the string-named one included.
        2. A match_guard rejection on a spec that also declares strict_validation, which the match
           path turns into a silent non-match.

        A guard on a non-strict spec means "this feature group does not match", not "this value is
        wrong", so it stays unreported. A ValueError raised while parsing a PREFIX_PATTERN match
        (malformed feature name, no chain separator) is a parse error, not an option-value
        rejection, and is likewise nothing to report. Returns None when nothing was rejected (the
        match succeeded, or the candidate is unrelated). Diagnostic-only: does not affect
        match_feature_group_criteria's behavior.
        """
        property_mapping = cls._get_property_mapping()
        if property_mapping is None:
            return None

        name_matched = False
        effective_options = options
        prefix_patterns = cls._get_prefix_patterns()
        if prefix_patterns:
            try:
                parsed = FeatureChainParser.parse_name(feature_name, prefix_patterns, CHAIN_SEPARATOR)
            except ValueError:
                return None
            name_matched = FeatureChainParser._name_identifies_group(parsed, property_mapping)
            if name_matched:
                bindings = FeatureChainParser.bind_name_captures(parsed, property_mapping)
                effective_options = FeatureChainParser._merge_bindings(options, bindings, property_mapping)

        try:
            if name_matched:
                # The name relates the feature group to the feature, so the values of the present
                # options (name-derived bindings included) are judged here; the name-path presence
                # reason follows below. Judging the effective options keeps the diagnostic in step with the match.
                FeatureChainParser._validate_present_option_values(effective_options, property_mapping)
            else:
                matches_mapping = FeatureChainParser._validate_options_against_property_mapping(
                    options, property_mapping
                )
                # Neither the name nor the option set relates this feature group to the feature: its
                # guards are none of the feature's business.
                if not matches_mapping:
                    return None
        except ValueError as exc:
            return str(exc)

        if name_matched:
            reason = FeatureChainParser.name_path_presence_rejection_reason(effective_options, property_mapping)
            if reason is not None:
                return reason

        rejection = cls._first_rejecting_guard(effective_options, property_mapping)
        if rejection is None:
            return None

        key, value = rejection
        if not property_mapping[key].strict_validation:
            return None
        return f"Property value '{value}' rejected by match_guard for '{key}'"

    @classmethod
    def _validate_forwarded_name_mismatch(
        cls,
        feature_name: str | FeatureName,
        bindings: dict[str, str],
        options: Options,
    ) -> None:
        """Reject consumer-forwarded option values that contradict a name-derived binding.

        Iterates every binding, so a secondary capture is protected exactly like the first one. The
        name-parsed value takes precedence, so a differing forwarded value would be silently ignored.
        Raises ValueError unless MLODA_ALLOW_FORWARDED_NAME_MISMATCH downgrades the error to a warning.
        """
        if not options.inherited_group_keys or not bindings:
            return
        for prop_key, name_value in bindings.items():
            if prop_key not in options.inherited_group_keys:
                continue
            inherited_value = options.get(prop_key)
            if inherited_value is None:
                continue
            # A singleton collection equals its sole element, exactly as _unpack_property_value treats it
            # everywhere else; only a differing or multi-value forward is a real mismatch.
            unpacked = FeatureChainParser._unpack_property_value(inherited_value)
            if len(unpacked) == 1 and str(unpacked[0]) == name_value:
                continue
            message = (
                f"Feature '{feature_name}': option '{prop_key}' was forwarded from a consumer with value "
                f"'{inherited_value}', but the feature name parses to '{name_value}'. The name-parsed value "
                f"takes precedence, so the forwarded value would be silently ignored. Carve the key out with "
                f"forward_group_exclude={{'{prop_key}'}} on the child in the consumer's input_features, or use an "
                f"allowlist / forward_group=False. Set MLODA_ALLOW_FORWARDED_NAME_MISMATCH=1 to downgrade this "
                f"error to a warning."
            )
            if os.environ.get("MLODA_ALLOW_FORWARDED_NAME_MISMATCH", "").lower() in ("1", "true"):
                logger.warning(message)
                continue
            raise ValueError(message)

    @classmethod
    def _first_rejecting_guard(
        cls, options: Options, property_mapping: dict[str, PropertySpec] | None
    ) -> tuple[str, Any] | None:
        """Return the (key, value) of the first match_guard that rejects its option value, or None.

        A guard rejects by returning a falsy value or by raising. Shared by the match decision
        (_validate_match_guards) and the diagnostic (_strict_validation_rejection_reason), so the
        two can never disagree on what a guard rejected.
        """
        if property_mapping is None:
            return None

        for key, mapping_entry in property_mapping.items():
            guard = mapping_entry.match_guard
            if guard is None:
                continue
            value = options.get(key)
            # An opted-in explicit None reaches the guard; every flagless spec still skips a None (#768).
            if not option_key_is_present(mapping_entry, key, options):
                continue
            try:
                rejected = not guard(value)
            except Exception as exc:
                level = _contained_raise_log_level(exc)
                if level == logging.DEBUG:
                    logger.debug("match_guard for '%s' raised %s for value %r", key, exc, value)
                else:
                    # The raw value stays out of WARNING logs; rerun with debug logging to see it.
                    logger.warning("match_guard for '%s' raised %s", key, exc)
                rejected = True
            if rejected:
                return key, value
        return None

    @classmethod
    def _validate_match_guards(
        cls, result: bool, options: Options, property_mapping: dict[str, PropertySpec] | None
    ) -> bool:
        # Enforce match_guard constraints from PROPERTY_MAPPING
        if not result:
            return True

        rejection = cls._first_rejecting_guard(options, property_mapping)
        if rejection is None:
            return True

        key, value = rejection
        logger.debug("match_guard for '%s' rejected value %r", key, value)
        if property_mapping is not None and property_mapping[key].strict_validation:
            record_match_rejection(cls.__name__, f"Property value '{value}' rejected by match_guard for '{key}'")
        return False

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
                    # An in_features shape this matcher cannot count (SourceInputFeature stores join
                    # tuples there) is a NON-MATCH: a group that cannot even count the in_features
                    # cannot consume them. Skipping MIN/MAX here would accept the feature and let the
                    # group win a resolution its own cap says it must lose.
                    in_features: frozenset[Feature] | None = safe_field(
                        options.get_in_features, None, catching=(TypeError,)
                    )
                    if in_features is None:
                        return False
                    count = len(in_features)
                if count < cls.MIN_IN_FEATURES:
                    return False
                if cls.MAX_IN_FEATURES is not None and count > cls.MAX_IN_FEATURES:
                    return False
        return True

    @classmethod
    def _get_prefix_patterns(cls) -> list[Any]:
        """Get prefix/suffix patterns from class attributes.

        Delegates to the guard's own collector, so matcher and guard can never see different patterns.
        """
        return FeatureChainParser.prefix_patterns_of(cls)

    @classmethod
    def _get_property_mapping(cls) -> Optional[dict[str, PropertySpec]]:
        """Get property mapping from class attribute."""
        if hasattr(cls, "PROPERTY_MAPPING"):
            return cast(Optional[dict[str, PropertySpec]], cls.PROPERTY_MAPPING)
        return None

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
        property_mapping = cls._get_property_mapping()

        parsed = FeatureChainParser.parse_name(feature.name, prefix_patterns, CHAIN_SEPARATOR)

        # Same identification gate as input_features: the name owns the source only when it identifies
        # the group (#772 / #769).
        if FeatureChainParser._name_identifies_group(parsed, property_mapping) and parsed.source_feature:
            return parsed.source_feature.split(cls.IN_FEATURE_SEPARATOR)

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
