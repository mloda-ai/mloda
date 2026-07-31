"""Class-definition-time guards for feature chaining: author validation, authoring diagnostics, and the
matcher guards the two ``__init_subclass__`` hooks install.

Imports ``feature_chain_parser``; the parser never imports this module, which keeps the split acyclic.

Depends on these parser-private names, so renaming one of them is a cross-module break:
``FeatureChainParser._can_skip_required_check``, ``._check_name_path_required_presence``, ``._merge_bindings``,
``._name_identifies_group``, and ``._name_path_missing_required_keys``.
"""

from __future__ import annotations

import contextvars
import functools
import logging
import re
from typing import Any

from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import (
    CHAIN_SEPARATOR,
    FeatureChainParser,
    option_key_is_present,
    record_match_rejection,
)
from mloda.core.abstract_plugins.components.feature_chainer.parsed_feature_name import ParsedFeatureName
from mloda.core.abstract_plugins.components.feature_chainer.property_spec import PropertySpec
from mloda.core.abstract_plugins.components.utils import contained_raise_log_level, escalate_match_abort, safe_field

logger = logging.getLogger(__name__)

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


def _pattern_named_and_total_groups(pattern: Any) -> tuple[frozenset[str], int]:
    """The named group names and total group count of a pattern; an uncompilable pattern reports neither."""
    if isinstance(pattern, re.Pattern):
        return frozenset(pattern.groupindex), pattern.groups
    compiled: re.Pattern[str] | None = safe_field(lambda: re.compile(pattern), None, catching=(re.error, TypeError))
    if compiled is None:
        return frozenset(), 0
    return frozenset(compiled.groupindex), compiled.groups


def _flatten_patterns(patterns: list[Any]) -> list[Any]:
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


def _str_reachable_values(spec: PropertySpec) -> set[str]:
    """The str members of a spec's value space; only a str can be reverse-looked-up from a capture."""
    reachable: set[str] = set()
    for value in FeatureChainParser.extract_property_values(spec):
        if isinstance(value, str):
            reachable.add(value)
    return reachable


def validate_name_binding(owner: type[Any]) -> None:
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

    patterns = FeatureChainParser.prefix_patterns_of(owner)
    if not patterns:
        return

    needs_overlap_check = False
    for pattern in _flatten_patterns(patterns):
        named, total = _pattern_named_and_total_groups(pattern)
        if total >= 1 and not named:
            needs_overlap_check = True

    if not needs_overlap_check:
        return

    reachable = {
        key: _str_reachable_values(spec) for key, spec in property_mapping.items() if isinstance(spec, PropertySpec)
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


def warn_captureless_without_binding(owner: type[Any]) -> None:
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
    patterns = FeatureChainParser.prefix_patterns_of(owner)
    if not patterns:
        return
    for pattern in _flatten_patterns(patterns):
        _named, total = _pattern_named_and_total_groups(pattern)
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


def warn_universal_optional_matcher(owner: type[Any]) -> None:
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
        if not FeatureChainParser._can_skip_required_check(spec):
            return
    matcher = getattr(owner, "match_feature_group_criteria", None)
    if matcher is None:
        return
    # A matcher that raises on the probe is doing custom work, so it is not treated as universal.
    try:
        universal = bool(matcher(_UNIVERSAL_MATCHER_PROBE_NAME, Options()))
    except Exception as exc:
        # rebind: Python clears the "except ... as exc" name at block exit, so the closure needs a stable local
        err = exc
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


def warn_missing_columnwise_hooks(owner: type[Any]) -> None:
    """Nudge an author whose framework-bound class leaves a required column-wise hook on the raising default.

    ``compute_framework_rule`` in the class's OWN __dict__ is the static marker of a framework-bound
    implementation. It is read, never called: running author code at class-definition time is not this
    guard's business. A family base declares the requirement for its children without that marker, so
    it stays silent, and only the hooks actually left unimplemented are named.
    """
    if "compute_framework_rule" not in owner.__dict__:
        return
    # Local import: feature_chain_parser_mixin imports this module, so a module-scope import would cycle.
    from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import (
        missing_columnwise_hooks,
    )

    missing = missing_columnwise_hooks(owner)
    if not missing:
        return
    logger.warning(
        "%s binds a compute framework but leaves %s on the raising default of FeatureChainParserMixin, "
        "so a run reaches the hook and fails there. Implement them on this class, or narrow "
        "REQUIRED_COLUMNWISE_HOOKS if the family does not call them.",
        owner.__name__,
        ", ".join(missing),
    )


def check_required_when(
    owner_name: str,
    feature_name: str | FeatureName,
    prefix_patterns: list[Any],
    property_mapping: dict[str, Any] | None,
    options: Options,
) -> bool:
    """Evaluate every required_when predicate of a mapping. False means the feature is not a match."""
    if property_mapping is None or not FeatureChainParser.has_required_when_predicates(property_mapping):
        return True

    # build_effective_options runs no user callback, so a raise from it is a framework defect (or a user
    # configuration error carrying actionable guidance) and must surface, not read as a non-match (#763).
    # Marked so it survives the match seam, which otherwise contains every raise (#845).
    try:
        effective_options = FeatureChainParser.build_effective_options(
            feature_name, prefix_patterns, property_mapping, options
        )
    except Exception as exc:
        escalate_match_abort(exc)
        raise
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
                contained_raise_log_level(exc),
                "required_when predicate %s for '%s' raised %s; treating feature group %s as a non-match.",
                getattr(predicate, "__name__", repr(predicate)),
                key,
                exc,
                owner_name,
            )
            return False
        # An opted-in key present as an explicit None counts as present, so the requirement is met (#768).
        if is_required and not option_key_is_present(spec, key, effective_options):
            predicate_name = getattr(predicate, "__name__", repr(predicate))
            logger.debug(
                "Feature group %s requires option '%s' (predicate %s is satisfied) but it was not provided.",
                owner_name,
                key,
                predicate_name,
            )
            # Same diagnostic seam as the sibling presence rules, so the resolution-failure report can
            # explain this non-match. The engine re-keys the harvest by candidate, so the reason itself
            # names the class that declared the requirement.
            record_match_rejection(
                owner_name,
                f"required option '{key}' is absent, but {owner_name} declares it required "
                f"(required_when predicate {predicate_name} is satisfied)",
            )
            return False
    return True


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


def _matcher_is_staticmethod(owner: type[Any]) -> bool:
    """True when the class's resolved matcher is a staticmethod descriptor."""
    for klass in owner.__mro__:
        descriptor = klass.__dict__.get("match_feature_group_criteria")
        if descriptor is not None:
            return isinstance(descriptor, staticmethod)
    return False


def _reject_staticmethod_matcher(owner: type[Any]) -> None:
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


def install_required_when_guard(owner: type[Any]) -> None:
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
    if not isinstance(property_mapping, dict) or not FeatureChainParser.has_required_when_predicates(property_mapping):
        return

    _reject_staticmethod_matcher(owner)

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

            feature_name, options = _resolve_match_arguments(args, kwargs)
            if options is None:
                return True

            return check_required_when(
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


def install_name_path_presence_guard(owner: type[Any]) -> None:
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
    if not FeatureChainParser.prefix_patterns_of(owner):
        return
    # Same exemptions as the inner rule: with empty options, the missing keys ARE the flaggable ones.
    if not FeatureChainParser._name_path_missing_required_keys(Options(), property_mapping):
        return

    # Wrapping a staticmethod matcher would hide it from _reject_staticmethod_matcher, so the
    # required_when installer's existing definition-time ValueError keeps precedence.
    is_static = _matcher_is_staticmethod(owner)
    if is_static and FeatureChainParser.has_required_when_predicates(property_mapping):
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

            feature_name, options = _resolve_match_arguments(args, kwargs)
            if options is None:
                return True

            mapping = getattr(guarded_cls, "PROPERTY_MAPPING", None)
            if not isinstance(mapping, dict):
                return True
            # Flattened, because a matcher passes a list-valued pattern attribute straight to
            # parse_name, so its ELEMENTS are the concrete patterns. A matcher must never leak
            # an exception, so the parse is contained as in build_effective_options; re.error is
            # additionally caught here because a malformed pattern must degrade to a non-match
            # of the name path, never veto the inner verdict (#868).
            patterns = _flatten_patterns(FeatureChainParser.prefix_patterns_of(guarded_cls))
            parsed = safe_field(
                lambda: FeatureChainParser.parse_name(feature_name, patterns, CHAIN_SEPARATOR),
                ParsedFeatureName.no_match(),
                catching=(ValueError, re.error),
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
