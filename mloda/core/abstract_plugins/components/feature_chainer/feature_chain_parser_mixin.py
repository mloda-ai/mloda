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

A guard rejection on a spec that also sets ``strict_validation=True``, or that declares
``expected``, is reportable: the match pass records it as it happens, and the recorded
reason feeds the resolution-failure report. ``_strict_validation_rejection_reason``
remains a standalone diagnostic facade producing the same message. A guard on a
non-strict spec with no ``expected`` keeps its "not mine" meaning and reports nothing.

Validators must be pure functions with no side effects. They may be called
multiple times during feature group resolution (once per candidate feature
group). Return values use truthy/falsy semantics: any falsy return (``False``,
``0``, ``""``, ``[]``) is treated as rejection.
"""

from __future__ import annotations

import inspect
import logging
import os
from copy import copy
from collections.abc import Callable, Sequence
from typing import Any, ClassVar, cast

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_author_guards import (
    install_name_path_presence_guard,
    install_required_when_guard,
    validate_name_binding,
    warn_captureless_without_binding,
    warn_missing_in_features_declaration,
    warn_universal_optional_matcher,
)
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import (
    FeatureChainParser,
    INPUT_SEPARATOR,
    PropertyValueRejection,
)
from mloda.core.abstract_plugins.components.feature_chainer.parsed_feature_name import NameResolution
from mloda.core.abstract_plugins.components.match_rejection import NAME_STAGE, record_match_rejection
from mloda.core.abstract_plugins.components.property_spec import PropertySpec, is_no_default
from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.core.abstract_plugins.components.utils import (
    contained_raise_reason,
    escalate_match_abort,
    is_match_abort,
    safe_field,
    safe_value_text,
)

logger = logging.getLogger(__name__)

# The pair every column-wise family needs: the source-feature check and the result writer.
COLUMNWISE_HOOKS: frozenset[str] = frozenset({"_check_source_features_exist", "_add_result_to_data"})

# The pair plus the discovery hook, for a family that resolves column names against the data.
COLUMN_DISCOVERY_HOOKS: frozenset[str] = COLUMNWISE_HOOKS | {"_get_available_columns"}


class FeatureChainParserMixin:
    """
    Mixin providing default implementations for feature chain parsing.

    Beyond the chain-parsing helpers, it also carries the column-wise data hooks
    (see the section at the end of the class).

    Subclasses should define:
    - PREFIX_PATTERN or SUFFIX_PATTERN: Regex patterns for matching
    - PROPERTY_MAPPING: Property validation mapping (see docs/in_depth/property-mapping.md)
    - IN_FEATURE_SEPARATOR: Optional custom separator (default: "&")
    - MIN_IN_FEATURES: Optional minimum in_feature count (default: 1);
      a group with no in_features key in PROPERTY_MAPPING and a minimum of 1 draws a definition-time warning
      (exempt: a name pattern, an input_features override, a custom matcher)
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
    ``feature_chain_author_guards.install_required_when_guard``), so overriding
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

    # Lets the class-definition guards tell a mixin group from a plain one without importing this module.
    IS_CHAIN_PARSER_MIXIN: ClassVar[bool] = True
    IN_FEATURE_SEPARATOR: str = INPUT_SEPARATOR
    MIN_IN_FEATURES: int = 1
    MAX_IN_FEATURES: int | None = None
    # A recognition-only pattern binds no key from the name; all values come from options (#772).
    RECOGNITION_ONLY_PATTERN: bool = False
    # An all-optional PROPERTY_MAPPING that inherits the config matcher matches any feature name once
    # in_features supplies a source; set True to declare that universal match intentional and silence the #771 warning.
    ALLOW_UNIVERSAL_MATCHER: bool = False
    # The column-wise hooks a family's calculate_feature calls; a family base declares it so its
    # framework implementations can be checked against it with missing_columnwise_hooks below.
    REQUIRED_COLUMNWISE_HOOKS: frozenset[str] = frozenset()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # The mixin sits first in the MRO of ``class X(FeatureChainParserMixin, FeatureGroup)``,
        # so super() is what lets FeatureGroup's own class-definition validation still run.
        super().__init_subclass__(**kwargs)
        validate_name_binding(cls)
        warn_captureless_without_binding(cls)
        install_name_path_presence_guard(cls)
        install_required_when_guard(cls)
        warn_universal_optional_matcher(cls)
        warn_missing_in_features_declaration(cls, FeatureChainParserMixin)

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

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        """
        Parse input features from feature name or options.

        First attempts to parse in_features from the feature name string; an agreeing declared in_features
        supplies them instead, keeping its options and feature_group scope.
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
        resolution = self.resolve_feature_name(feature_name)

        # The name is authoritative only when it identifies this group (a captureless recognition match
        # or a participating capture). An optional-first positional group that did not participate does
        # not identify the group, so its source comes from options, not the name (#772 / #769).
        if resolution.owned and resolution.sources:
            self._raise_source_features_reason(feature_name, resolution.sources)
            declared = self._agreeing_declared_in_features(options, list(resolution.sources))
            if declared is not None:
                return {copy(f) for f in declared}  # copies: the engine mutates returned features in place
            return {Feature(n) for n in self.declared_source_names(list(resolution.sources))}

        # Configuration-based fallback using get_in_features()
        in_features = options.get_in_features()
        self._raise_source_features_reason(feature_name, tuple(str(f.name) for f in in_features))
        return {copy(f) for f in in_features}

    @classmethod
    def resolve_feature_name(cls, name: str | FeatureName) -> NameResolution:
        """Resolve ``name`` against this class's patterns, mapping and in_feature separator."""
        return FeatureChainParser.resolve_name(
            name, cls._get_prefix_patterns(), cls._get_property_mapping(), cls.IN_FEATURE_SEPARATOR
        )

    @classmethod
    def source_features_reason(cls, feature_name: str | FeatureName, sources: Sequence[str]) -> str | None:
        """The reason these sources cannot serve the feature: an empty operand first, then the MIN/MAX count."""
        if any(source == "" for source in sources):
            return f"Feature '{feature_name}' has an empty in_feature operand"
        return cls.in_feature_count_reason(feature_name, len(sources))

    @classmethod
    def _raise_source_features_reason(cls, feature_name: str | FeatureName, sources: Sequence[str]) -> None:
        reason = cls.source_features_reason(feature_name, sources)
        if reason is not None:
            # Contained: sources this group cannot serve mean it cannot serve the feature.
            raise ValueError(reason)

    @classmethod
    def in_feature_count_reason(cls, feature_name: str | FeatureName, count: int) -> str | None:
        """The MIN/MAX in_feature error message, or None when the count is in range."""
        if count < cls.MIN_IN_FEATURES:
            return f"Feature '{feature_name}' requires at least {cls.MIN_IN_FEATURES} in_feature(s), but found {count}"
        if cls.MAX_IN_FEATURES is not None and count > cls.MAX_IN_FEATURES:
            return f"Feature '{feature_name}' allows at most {cls.MAX_IN_FEATURES} in_feature(s), but found {count}"
        return None

    @classmethod
    def validate_in_feature_count(cls, feature_name: str | FeatureName, count: int) -> None:
        """Raise ValueError with in_feature_count_reason's message, if any."""
        reason = cls.in_feature_count_reason(feature_name, count)
        if reason is not None:
            # Contained: an in_feature count outside the declared MIN/MAX means this group cannot serve the feature.
            raise ValueError(reason)

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

        Also enforces MIN_IN_FEATURES / MAX_IN_FEATURES, counting the sources the name
        carries when it identifies the group, else the in_features option value. A
        name-carried count outside the range is recorded as a reportable rejection; an
        in_features option value the matcher cannot resolve is a silent non-match. An
        absent in_features counts as zero on the configuration path. This gate runs before ``match_guard``.

        ``required_when`` is NOT evaluated here. The guard installed at class definition
        runs the predicates after this method (or any override of it) returns True.

        For string-parsed features, additionally checks consumer-forwarded option
        values against the value parsed from the feature name: if a PROPERTY_MAPPING
        key arrived via forwarding (it is in ``options.inherited_group_keys`` or
        ``options.inherited_context_keys``) and its value differs from the name-parsed
        value, a ``ValueError`` is raised, because the name-parsed value would
        otherwise silently win. Setting the environment variable
        ``MLODA_ALLOW_FORWARDED_NAME_MISMATCH=1`` downgrades this error to a warning.

        The options view depends on the caller: feature resolution passes declared (pre-default)
        options, filter matching a post-intake merge. See ``FeatureGroup.match_feature_group_criteria``.

        Args:
            feature_name: Feature name to match
            options: Options containing configuration
            data_access_collection: Optional data access collection (unused)

        Returns:
            True if feature matches criteria, False otherwise
        """
        property_mapping = cls._get_property_mapping()

        result = cls.match_parser_criteria(feature_name, options)

        # On a string match the guards see the EFFECTIVE options (name-derived bindings merged in), so a
        # name-carried value is as visible to match_guard as an explicit one. A no-source ValueError cannot
        # reach here: it would already have made match_parser_criteria a non-match.
        effective_options = options
        # The sources input_features would read off the name, so the in_feature gate counts the same ones.
        name_sources: list[str] | None = None
        if result:
            resolution = cls.resolve_feature_name(feature_name)
            parsed = resolution.parsed
            if resolution.owned:
                if resolution.sources:
                    name_sources = list(resolution.sources)
                # Bound once and reused: the merge and the guards must see the name-derived value even
                # when the legacy operation value is absent (a named-optional-first pattern).
                bindings = dict(resolution.bindings)
                operation_config = FeatureChainParser._legacy_operation_config(parsed)
                # _validate_string_match needs a str operation, so it stays behind its own gate; the
                # merge and forwarded-mismatch protection do not.
                if operation_config is not None and parsed.source_feature is not None:
                    if not cls._validate_string_match(feature_name, operation_config, parsed.source_feature):
                        return False
                effective_options = FeatureChainParser._merge_bindings(options, bindings, property_mapping)

        if not cls._validate_in_features(result, options, name_sources, feature_name):
            return False

        if result and resolution.owned:
            cls._validate_forwarded_name_mismatch(feature_name, bindings, options)
            cls._validate_name_agreement(feature_name, bindings, name_sources, options)

        if not cls._validate_match_guards(result, effective_options, property_mapping):
            return False

        return result

    @classmethod
    def match_parser_criteria(cls, feature_name: str | FeatureName, options: Options) -> bool:
        """Call the parser, turning a rejected option value or a malformed name into a non-match, never an exception.

        The preferred way to reach the parser from an overridden ``match_feature_group_criteria``: a raise from a
        match hook is contained as a ``match hook`` near-miss, but a rejection carries a better reason than a crash.
        """
        try:
            return FeatureChainParser.match_configuration_feature_chain_parser(
                feature_name,
                options,
                property_mapping=cls._get_property_mapping(),
                prefix_patterns=cls._get_prefix_patterns(),
                owner_name=cls.__name__,
            )
        # PropertyValueRejection subclasses ValueError, so this handler sits above the abort check and must
        # run it: an unmarked rejection is the parser's non-match verdict, recorded as the candidate's reason.
        except PropertyValueRejection as exc:
            if is_match_abort(exc):
                raise
            record_match_rejection(cls.__name__, str(exc))
            return False
        # A marked raise crosses this containment; an unmarked ValueError stays a non-match.
        except ValueError as exc:
            if is_match_abort(exc):
                raise
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
        2. A match_guard rejection on a spec that also declares strict_validation or ``expected``,
           which the match path would otherwise turn into a silent non-match.

        A guard on a non-strict spec with no ``expected`` means "this feature group does not
        match", not "this value is wrong", so it stays unreported. A ValueError raised while
        parsing a PREFIX_PATTERN match (malformed feature name, no chain separator) is a parse
        error, not an option-value rejection, and is likewise nothing to report. Returns None
        when nothing was rejected (the match succeeded, or the candidate is unrelated).
        Diagnostic-only: does not affect match_feature_group_criteria's behavior.
        """
        property_mapping = cls._get_property_mapping()
        if property_mapping is None:
            return None

        name_matched = False
        effective_options = options
        name_sources: list[str] | None = None
        prefix_patterns = cls._get_prefix_patterns()
        if prefix_patterns:
            try:
                resolution = cls.resolve_feature_name(feature_name)
            except ValueError:
                return None
            name_matched = resolution.owned
            if name_matched:
                if resolution.sources:
                    name_sources = list(resolution.sources)
                effective_options = FeatureChainParser._merge_bindings(
                    options, dict(resolution.bindings), property_mapping
                )

        try:
            if name_matched:
                FeatureChainParser.validate_name_bindings(resolution.bindings, property_mapping)
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

        # Mirrors the matcher's gate order: a name-carried count is reported, the options-path gate is a silent non-match.
        if name_sources is not None:
            reason = cls.source_features_reason(feature_name, name_sources)
            if reason is not None:
                return reason
        elif not cls._validate_in_features(True, options, None, feature_name):
            return None

        rejection = cls._first_rejecting_guard(effective_options, property_mapping)
        if rejection is None:
            return None

        key, value = rejection
        return cls._guard_rejection_reason(key, value, property_mapping[key])

    @classmethod
    def _validate_forwarded_name_mismatch(
        cls,
        feature_name: str | FeatureName,
        bindings: dict[str, str],
        options: Options,
    ) -> None:
        """Reject a value forwarded via either the group or the context path that contradicts a
        name-derived binding.

        Iterates every binding, so a secondary capture is protected exactly like the first one. The
        name-parsed value takes precedence, so a differing forwarded value would be silently ignored.
        Raises ValueError unless MLODA_ALLOW_FORWARDED_NAME_MISMATCH downgrades the error to a warning.
        """
        inherited_keys = options.inherited_group_keys | options.inherited_context_keys
        if not inherited_keys or not bindings:
            return
        for prop_key, name_value in bindings.items():
            if prop_key not in inherited_keys:
                continue
            inherited_value = options.get(prop_key)
            if inherited_value is None:
                continue
            # A singleton collection equals its sole element, exactly as _unpack_property_value treats it
            # everywhere else; only a differing or multi-value forward is a real mismatch.
            unpacked = FeatureChainParser._unpack_property_value(inherited_value)
            if len(unpacked) == 1 and str(unpacked[0]) == name_value:
                continue
            if prop_key in options.inherited_group_keys:
                remedy = (
                    f"Carve the key out with forward_group_exclude={{'{prop_key}'}} on the child in the "
                    f"consumer's input_features, or use an allowlist / forward_group=False."
                )
            else:
                remedy = (
                    f"Adjust inherit_context_keys on the child or propagate_context_keys on the consumer "
                    f"so '{prop_key}' is no longer forwarded."
                )
            message = (
                f"Feature '{feature_name}': option '{prop_key}' was forwarded from a consumer with value "
                f"'{inherited_value}', but the feature name parses to '{name_value}'. The name-parsed value "
                f"takes precedence, so the forwarded value would be silently ignored. {remedy} Set "
                f"MLODA_ALLOW_FORWARDED_NAME_MISMATCH=1 to downgrade this error to a warning."
            )
            if cls._name_mismatch_downgraded(message):
                continue
            # Marked: user misconfiguration; containing it would let a rival group win with the value ignored (#845).
            raise escalate_match_abort(ValueError(message))

    @classmethod
    def _name_mismatch_downgraded(cls, message: str) -> bool:
        """Warn and return True when MLODA_ALLOW_FORWARDED_NAME_MISMATCH downgrades the mismatch error."""
        if os.environ.get("MLODA_ALLOW_FORWARDED_NAME_MISMATCH", "").lower() not in ("1", "true"):
            return False
        logger.warning(message)
        return True

    @classmethod
    def _validate_name_agreement(
        cls,
        feature_name: str | FeatureName,
        bindings: dict[str, str],
        name_sources: list[str] | None,
        options: Options,
    ) -> None:
        """Abort when a declared option or in_features contradicts what the name binds."""
        inherited_keys = options.inherited_group_keys | options.inherited_context_keys
        env_hint = "Set MLODA_ALLOW_FORWARDED_NAME_MISMATCH=1 to downgrade this error to a warning."
        for key, name_value in bindings.items():
            declared = options.get(key)
            if declared is None or key in inherited_keys or not options.is_own(key):
                continue
            unpacked = FeatureChainParser._unpack_property_value(declared)
            if len(unpacked) == 1 and str(unpacked[0]) == name_value:
                continue
            message = (
                f"Feature '{feature_name}': option '{key}' is {safe_value_text(declared)}, but the feature name encodes "
                f"'{name_value}'. The name is authoritative: remove the option or change it to match. {env_hint}"
            )
            if cls._name_mismatch_downgraded(message):
                continue
            # Marked: a declared value contradicting the name is user misconfiguration.
            raise escalate_match_abort(ValueError(message))

        view = cls._declared_in_features_view(options, name_sources)
        if view is None:
            return
        declared_in_features, features, expected = view
        declared_names = None if features is None else [str(f.name) for f in features]
        if declared_names != expected:
            shown = (
                safe_value_text(declared_in_features)
                if declared_names is None
                else [safe_value_text(name) for name in declared_names]
            )
            hints = ""
            if declared_names is not None and sorted(declared_names) == sorted(expected):
                hints += " Order matters: list them in the name's order."
            if any("__" in source for source in name_sources or []):
                hints += (
                    " in_features must list the name's direct sources, not the root source: drop it or make it match."
                )
            message = (
                f"Feature '{feature_name}': in_features is {shown}, "
                f"but the feature name's direct sources are {expected}.{hints} {env_hint}"
            )
            if not cls._name_mismatch_downgraded(message):
                # Marked: a declared in_features contradicting the name is user misconfiguration.
                raise escalate_match_abort(ValueError(message))

    @classmethod
    def _declared_in_features_view(
        cls, options: Options, name_sources: list[str] | None
    ) -> tuple[Any, list[Feature] | None, list[str]] | None:
        """Return (declared value, declared Features or None, expected names) for an own in_features, else None."""
        in_features_key = DefaultOptionKeys.in_features.value
        declared = options.get(in_features_key)
        inherited_keys = options.inherited_group_keys | options.inherited_context_keys
        if name_sources is None or not declared or in_features_key in inherited_keys:
            return None
        if not options.is_own(in_features_key):
            return None
        features = safe_field(lambda: list(options.get_in_features()), None, catching=(TypeError, ValueError))
        expected = cls.declared_source_names(name_sources)
        if isinstance(declared, (set, frozenset)):
            expected = sorted(expected)
        return declared, features, expected

    @classmethod
    def _agreeing_declared_in_features(
        cls, options: Options, name_sources: list[str] | None
    ) -> tuple[Feature, ...] | None:
        """Return the declared in_features when their names equal the name's declared sources, else None."""
        view = cls._declared_in_features_view(options, name_sources)
        if view is None:
            return None
        _, features, expected = view
        if features is None:
            return None
        return tuple(features) if [str(f.name) for f in features] == expected else None

    @classmethod
    def declared_source_names(cls, name_sources: list[str]) -> list[str]:
        """Map the name's sources to the source names this group declares; identity by default."""
        return list(name_sources)

    @classmethod
    def _first_rejecting_guard(
        cls, options: Options, property_mapping: dict[str, PropertySpec] | None
    ) -> tuple[str, Any] | None:
        """The first (key, value) a match_guard rejects, or None; see ``FeatureChainParser._first_rejecting_guard``."""
        return FeatureChainParser._first_rejecting_guard(options, property_mapping, logger)

    @classmethod
    def _guard_rejection_reason(cls, key: str, value: Any, spec: PropertySpec) -> str | None:
        """The reportable reason for a guard rejection, or None; see ``FeatureChainParser._guard_rejection_reason``."""
        return FeatureChainParser._guard_rejection_reason(key, value, spec)

    @classmethod
    def _validate_match_guards(
        cls, result: bool, options: Options, property_mapping: dict[str, PropertySpec] | None
    ) -> bool:
        """Enforce the match_guard constraints once the parser matched; see ``FeatureChainParser``."""
        if not result:
            return True
        return FeatureChainParser._validate_match_guards(cls.__name__, options, property_mapping, logger)

    @classmethod
    def _validate_in_features(
        cls,
        result: bool,
        options: Options,
        name_sources: list[str] | None = None,
        feature_name: str | FeatureName = "",
    ) -> bool:
        # Enforce MIN/MAX_IN_FEATURES on the name-carried sources when there are any, else on the options value
        if not (result and hasattr(cls, "MIN_IN_FEATURES") and hasattr(cls, "MAX_IN_FEATURES")):
            return True

        if name_sources is not None:
            # The name relates this group to the feature, so a count it cannot serve is an actionable
            # near-miss rather than a silent non-match; the option path keeps its "not mine" meaning.
            reason = cls.source_features_reason(feature_name, name_sources)
            if reason is None:
                return True
            record_match_rejection(cls.__name__, reason, stage=NAME_STAGE)
            return False

        in_features_raw = options.get(DefaultOptionKeys.in_features)
        if in_features_raw is None:
            property_mapping = cls._get_property_mapping()
            declared = property_mapping.get(DefaultOptionKeys.in_features.value) if property_mapping else None
            if declared is not None and not is_no_default(declared.default) and declared.default is not None:
                # A declared in_features default is supplied at intake, so the group keeps matching without it.
                return True

        if in_features_raw is None or (
            isinstance(in_features_raw, (list, tuple, set, frozenset)) and not in_features_raw
        ):
            # None counts as absent; absent or empty is zero in_features, a non-match rather than an error.
            count = 0
        else:
            # An in_features value this matcher cannot count is a non-match, not an error:
            # skipping MIN/MAX would let the group win a resolution its own cap says it must lose.
            # The catch is narrow on purpose; another exception class is a defect to surface, not a value.
            try:
                in_features = options.get_in_features()
                count = len(in_features)
            except (TypeError, ValueError) as exc:
                if is_match_abort(exc):
                    raise
                # Text, not exc: a retained record must not pin this frame, its cls and the plugin class.
                logger.debug(
                    "%s cannot resolve in_features value %r: %s",
                    cls.__name__,
                    in_features_raw,
                    contained_raise_reason(exc),
                )
                return False
            if any(str(f.name) == "" for f in in_features):
                return False

        return cls.in_feature_count_reason(feature_name, count) is None

    @classmethod
    def _get_prefix_patterns(cls) -> list[Any]:
        """Get prefix/suffix patterns from class attributes.

        Delegates to the guard's own collector, so matcher and guard can never see different patterns.
        """
        return FeatureChainParser.prefix_patterns_of(cls)

    @classmethod
    def _get_property_mapping(cls) -> dict[str, PropertySpec] | None:
        """Get property mapping from class attribute."""
        if hasattr(cls, "PROPERTY_MAPPING"):
            return cast(dict[str, PropertySpec] | None, cls.PROPERTY_MAPPING)
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
        resolution = cls.resolve_feature_name(feature.name)

        # Same identification gate as input_features: the name owns the source only when it identifies
        # the group (#772 / #769).
        if resolution.owned and resolution.sources:
            return list(resolution.sources)

        # Configuration-based fallback using get_in_features()
        return [str(f.name) for f in feature.options.get_in_features()]

    @classmethod
    def _extract_validated_source_features(cls, feature: Feature) -> list[str]:
        """``_extract_source_features`` plus the empty-operand and MIN/MAX_IN_FEATURES checks of a match."""
        sources = cls._extract_source_features(feature)
        cls._raise_source_features_reason(feature.name, sources)
        return sources

    @classmethod
    def _extract_single_source_feature(cls, feature: Feature) -> str:
        """Single-source counterpart to ``_extract_validated_source_features``; always enforces exactly one result.

        Raises:
            ValueError: if the resolved source count is not exactly one
        """
        source_features = cls._extract_validated_source_features(feature)
        if len(source_features) != 1:
            raise ValueError(
                f"Feature '{feature.name}' resolved {len(source_features)} source feature(s), expected exactly 1"
            )
        return source_features[0]

    @classmethod
    def _extract_operation_and_source_feature(
        cls, feature: Feature, extract_fn: Callable[[Feature], Any], label: str
    ) -> tuple[Any, str]:
        """
        Extract the primary source feature name and an operation parameter from a feature.

        Args:
            feature: The feature to extract parameters from
            extract_fn: Callable that returns the operation-specific value (or None if not found)
            label: Human-readable noun used in the error message when extraction fails

        Returns:
            Tuple of (operation_value, source_feature_name)

        Raises:
            ValueError: if the source count isn't exactly one, or the operation can't be extracted
        """
        source_feature = cls._extract_single_source_feature(feature)
        operation = extract_fn(feature)
        if operation is None:
            raise ValueError(f"Could not extract {label} from: {feature.name}")
        return operation, source_feature

    @classmethod
    def _resolve_operation(
        cls,
        feature_or_name: Any,
        options_or_key: Any,
        config_key: str | None = None,
    ) -> str | None:
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

        The string-based path takes precedence. If the feature name owns the match:
        with named captures, the capture for ``config_key`` is returned; with a
        positional pattern, the first capture. Otherwise, falls back to
        ``options.get(config_key)`` as a string (a singleton collection is unpacked).

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

        resolution = cls.resolve_feature_name(_name)
        if resolution.owned:
            operation_config = resolution.value_for(_key)
            if operation_config is not None:
                return operation_config
        value = _options.get(_key)
        if value is not None:
            unpacked = FeatureChainParser._unpack_property_value(value)
            return str(unpacked[0] if len(unpacked) == 1 else value)
        return None

    # Column-wise data hooks: the concrete compute-framework subclass (pandas, pyarrow, ...) implements
    # these. They are deliberately not @abstractmethod, since this is a plain class where that would be
    # unenforced; the defaults below raise instead, naming the class that skipped the implementation.

    @classmethod
    def _get_available_columns(cls, data: Any) -> set[str]:
        """
        Get the set of available column names from the data.

        Args:
            data: The input data

        Returns:
            Set of column names available in the data
        """
        raise NotImplementedError(f"{cls.__name__} must implement _get_available_columns")

    @classmethod
    def _check_source_features_exist(cls, data: Any, feature_names: list[str]) -> None:
        """
        Check that the resolved source features exist in the data.

        Args:
            data: The input data
            feature_names: Resolved source feature names (may contain ~N suffixes)

        Raises:
            ValueError: When the source features this feature group needs are absent. Whether
                partial presence is accepted is defined by the implementing feature group.
        """
        raise NotImplementedError(f"{cls.__name__} must implement _check_source_features_exist")

    @classmethod
    def _add_result_to_data(cls, data: Any, feature_name: str, result: Any) -> Any:
        """
        Add the result to the data.

        Args:
            data: The input data
            feature_name: The name of the feature to add
            result: The result to add

        Returns:
            The updated data
        """
        raise NotImplementedError(f"{cls.__name__} must implement _add_result_to_data")


def _resolved_hook(owner: type[Any], hook_name: str) -> Any:
    """The plain function behind a hook attribute, or None when the owner has no such hook."""
    attribute = inspect.getattr_static(owner, hook_name, None)
    return getattr(attribute, "__func__", attribute)


def _hook_is_implemented(owner: type[Any], hook_name: str) -> bool:
    """True when ``cls._hook(...)`` reaches an own implementation rather than the raising default.

    A plain function is unreachable: the ``cls`` slot would eat the data argument.
    """
    if not isinstance(inspect.getattr_static(owner, hook_name, None), (classmethod, staticmethod)):
        return False
    return _resolved_hook(owner, hook_name) is not _resolved_hook(FeatureChainParserMixin, hook_name)


def declared_columnwise_hooks(owner: type[Any]) -> frozenset[str]:
    """The column-wise hooks a class declares required; empty for a class that declares none."""
    return frozenset(getattr(owner, "REQUIRED_COLUMNWISE_HOOKS", frozenset()))


def missing_columnwise_hooks(owner: type[Any]) -> list[str]:
    """The declared hooks the class does not implement in a reachable shape, sorted.

    Assert this empty in a plugin repo's own suite to catch a skipped hook there instead of mid-run.
    A family base correctly reports all of them: it declares the contract its subclasses implement.
    """
    return sorted(hook for hook in declared_columnwise_hooks(owner) if not _hook_is_implemented(owner, hook))
