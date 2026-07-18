import inspect
from dataclasses import dataclass, field, replace
from difflib import get_close_matches
from typing import Any, Literal, Optional

from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.domain import Domain
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import PropertyValueRejection
from mloda.core.abstract_plugins.components.utils import safe_field
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.link import Link

import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandidateFrameworks:
    """One candidate's own accessible frameworks, split by the match-time capability hook."""

    supported: frozenset[type[ComputeFramework]] = frozenset()
    rejected: frozenset[type[ComputeFramework]] = frozenset()


@dataclass(frozen=True)
class RenderFacts:
    """Facts captured during the decision pass so rendering needs no provider hook.

    The empty instance is the success value: the winning path captures nothing.
    """

    environment_empty: bool = False
    domains: dict[type[FeatureGroup], str] = field(default_factory=dict)
    concrete_frameworks: tuple[str, ...] = ()
    value_rejections: tuple[tuple[str, str], ...] = ()
    known_names: tuple[str, ...] = ()
    # Render-only split over the DECLARED available frameworks, which is wider than the run's own
    # candidate_frameworks split: a framework the run did not enable still renders as supported.
    capability_frameworks: dict[type[FeatureGroup], CandidateFrameworks] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationResult:
    """Non-raising result of matching a feature against accessible plugins."""

    identified: FeatureGroupEnvironmentMapping
    criteria_matched: set[type[FeatureGroup]] = field(default_factory=set)
    abstract_matched: set[type[FeatureGroup]] = field(default_factory=set)
    candidate_frameworks: dict[type[FeatureGroup], CandidateFrameworks] = field(default_factory=dict)
    facts: RenderFacts = field(default_factory=RenderFacts)

    @property
    def failure_kind(self) -> Literal["multiple", "abstract_only", "none"] | None:
        # "none" means no winner in the identified mapping, not that nothing matched: an all-rejected
        # concrete group still yields "none" with a non-empty criteria_matched.
        n = len(self.identified)
        if n == 1:
            return None
        if n > 1:
            return "multiple"
        if self.abstract_matched:
            return "abstract_only"
        return "none"


@dataclass(frozen=True)
class ResolutionRecord:
    """One feature's identification during planning: its name, whether it was requested, and its EvaluationResult."""

    feature_name: str
    requested: bool
    result: EvaluationResult


class FeatureResolutionError(ValueError):
    """Typed resolution failure carrying the feature name and the EvaluationResult of its single pass."""

    def __init__(self, message: str, feature_name: str, result: EvaluationResult) -> None:
        super().__init__(message)
        self.feature_name = feature_name
        self.result = result

    def __reduce__(self) -> tuple[type["FeatureResolutionError"], tuple[str, str, EvaluationResult]]:
        # The default reduction reconstructs from args=(message,) and drops the two extra constructor arguments.
        return type(self), (str(self), self.feature_name, self.result)


def matches_feature_group_scope(feature_group: type[FeatureGroup], scope: str | type[FeatureGroup]) -> bool:
    """Is the candidate inside the requested scope, for both the class-object and the string form.

    The string form matches the named class and its subclasses by walking the candidate's ancestry
    (MRO), so a config that can only carry a name keeps the same subclass-preferring semantics. The
    root FeatureGroup base is excluded from that walk because every candidate carries it, which would
    make it a wildcard.
    """
    if isinstance(scope, type):
        return issubclass(feature_group, scope)
    # Name first: get_class_name() is @final and just returns __name__, while issubclass() on an ABCMeta
    # class is the expensive check, so the name gate keeps it off nearly every MRO entry.
    return any(
        ancestor.__name__ == scope and ancestor is not FeatureGroup and issubclass(ancestor, FeatureGroup)
        for ancestor in feature_group.__mro__
    )


def scope_callout(scope: str | type[FeatureGroup] | None) -> str | None:
    """Render the shared scope callout, or None when the scope is unset."""
    if scope is None:
        return None
    scope_name = scope.get_class_name() if isinstance(scope, type) else scope
    return f"Scoped to feature group: '{scope_name}'."


TROUBLESHOOTING_URL = "https://mloda-ai.github.io/mloda/in_depth/troubleshooting/feature-group-resolution-errors/"


def _candidate_sort_key(feature_group: type[FeatureGroup]) -> tuple[str, str]:
    """Sort candidates by name, then module: two candidates may share a name across modules."""
    return feature_group.__name__, feature_group.__module__


def _as_str(value: Any) -> str:
    """Return `value` unchanged, raising TypeError on a non-str, so the guarded read that wraps this degrades.

    A hook's annotation is a promise, not a guarantee. A non-str name reaching difflib raises there instead,
    far from the plugin to blame, so every captured name is validated where it is read.
    """
    if not isinstance(value, str):
        raise TypeError(f"expected str, got {type(value).__name__}")
    return value


def _supported_feature_names(feature_group: type[FeatureGroup]) -> set[str]:
    """Best-effort name catalog of one candidate. A malformed entry costs that candidate its whole catalog."""
    return safe_field(
        lambda: {_as_str(name) for name in feature_group.feature_names_supported()},
        set(),
        field=f"{feature_group.get_class_name()}.feature_names_supported",
    )


def _prefix_name(feature_group: type[FeatureGroup]) -> str:
    """Best-effort prefix of one candidate."""
    return safe_field(lambda: _as_str(feature_group.prefix()), "", field=f"{feature_group.get_class_name()}.prefix")


def _supports_framework(
    feature_group: type[FeatureGroup], feature: Feature, compute_framework: type[ComputeFramework]
) -> bool | None:
    """Best-effort capability answer. None when the hook raised, leaving the pair undecided for rendering."""
    return safe_field(
        lambda: feature_group.supports_compute_framework(feature.name, feature.options, compute_framework),
        None,
        field=f"{feature_group.get_class_name()}.supports_compute_framework",
    )


def _render_multiple(result: EvaluationResult, feature: Feature, callout: str | None) -> str:
    # Every identified candidate gets a line; only a candidate with a captured domain gets the suffix.
    lines = "\n".join(
        f"  - {fg.__name__} ({fg.__module__})"
        + (f" [domain: {result.facts.domains[fg]}]" if fg in result.facts.domains else "")
        for fg in sorted(result.identified, key=_candidate_sort_key)
    )
    scope_line = f"{callout}\n" if callout else ""
    return (
        f"Multiple feature groups found for feature '{str(feature.name)}':\n"
        f"{lines}\n"
        f"{scope_line}"
        f"For troubleshooting guide, see: {TROUBLESHOOTING_URL}"
    )


def _render_capability(result: EvaluationResult, feature: Feature) -> str | None:
    """One line per candidate that rejects a framework, each naming only its OWN frameworks."""
    rejecting = {fg: cfw for fg, cfw in result.facts.capability_frameworks.items() if cfw.rejected}
    if not rejecting:
        return None

    lines = []
    for fg in sorted(rejecting, key=_candidate_sort_key):
        candidate_frameworks = rejecting[fg]
        rejected_names = sorted(fw.get_class_name() for fw in candidate_frameworks.rejected)
        line = f"  - {fg.__name__}: {rejected_names}."
        if candidate_frameworks.supported:
            supported_names = sorted(fw.get_class_name() for fw in candidate_frameworks.supported)
            line += f" Supported on: {supported_names}."
        lines.append(line)

    body = "\n".join(lines)
    return (
        f"Unsupported compute framework(s) for feature '{str(feature.name)}':\n"
        f"{body}\n"
        "Pin the feature to a supported compute framework or override supports_compute_framework."
    )


def _render_abstract_only(result: EvaluationResult, feature: Feature) -> str:
    feature_name = str(feature.name)
    if not result.facts.concrete_frameworks:
        return (
            f"No feature groups found for feature name: '{feature_name}'. "
            f"Only abstract feature group base(s) matched, which cannot be instantiated; "
            f"no concrete implementation is available or enabled."
        )

    framework_names = sorted(result.facts.concrete_frameworks)
    return (
        f"No feature groups found for feature name: '{feature_name}'. "
        f"Its concrete implementations require compute framework(s) {framework_names}, "
        f"none of which are available or enabled for this run."
    )


def _render_none(result: EvaluationResult, feature: Feature, callout: str | None) -> str:
    feature_name = str(feature.name)
    msg = f"No feature groups found for feature name: '{feature_name}'."

    if callout:
        msg += f" {callout}"

    if result.facts.environment_empty:
        return msg + "\nNo plugins are loaded. Did you call PluginLoader.all()?"

    if result.facts.value_rejections:
        lines = "\n".join(f"  - {class_name}: {reason}" for class_name, reason in sorted(result.facts.value_rejections))
        msg += f"\nFeature group(s) rejected an option value while matching '{feature_name}':\n{lines}"

    similar = get_close_matches(feature_name, list(result.facts.known_names), n=5, cutoff=0.5)
    if similar:
        msg += f"\nDid you mean one of: {similar}?"

    pointer_args = "name, options=..., feature_group=..." if callout else "name, options=..."
    msg += (
        f"\nUse resolve_feature({pointer_args}) to debug feature resolution."
        f"\nFor troubleshooting guide, see: {TROUBLESHOOTING_URL}"
    )
    return msg


def render_resolution_failure(result: EvaluationResult, feature: Feature) -> str | None:
    """Project a failed EvaluationResult into its message. Pure: reads only the result and the Feature.

    Calls no provider-overridable hook, so every fact it needs was captured by evaluate(). The
    forwarding hint is dropped: it needs a speculative second match, which is not a projection.
    """
    kind = result.failure_kind
    if kind is None:
        return None

    callout = scope_callout(feature.feature_group_scope)

    if kind == "multiple":
        return _render_multiple(result, feature, callout)

    capability_message = _render_capability(result, feature)
    if capability_message is not None:
        return f"{capability_message} {callout}" if callout else capability_message

    if result.abstract_matched:
        abstract_message = _render_abstract_only(result, feature)
        return f"{abstract_message} {callout}" if callout else abstract_message

    return _render_none(result, feature, callout)


class IdentifyFeatureGroupClass:
    _criteria_matched_feature_groups: set[type[FeatureGroup]]
    _abstract_matched_feature_groups: set[type[FeatureGroup]]
    _candidate_frameworks: dict[type[FeatureGroup], CandidateFrameworks]
    _data_access_collection: Optional[DataAccessCollection]
    # Per-evaluation memos of the hooks more than one reader wants. evaluate() builds a fresh instance, so
    # they are scoped to one resolution attempt and never cache across runs.
    _domain_outcomes: dict[type[FeatureGroup], tuple[Optional[Domain], Optional[Exception]]]
    _declared_frameworks: dict[type[FeatureGroup], frozenset[type[ComputeFramework]]]
    result: EvaluationResult

    def __init__(
        self,
        feature: Feature,
        accessible_plugins: FeatureGroupEnvironmentMapping,
        links: Optional[set[Link]],
        data_access_collection: Optional[DataAccessCollection] = None,
    ):
        result = self.evaluate(feature, accessible_plugins, links, data_access_collection)
        # The message is a projection of the pass that just ran: no second evaluation, no re-asked hook.
        message = render_resolution_failure(result, feature)
        if message is not None:
            raise FeatureResolutionError(message, str(feature.name), result)
        self.feature_group_compute_framework_mapping = result.identified
        self.result = result

    @classmethod
    def evaluate(
        cls,
        feature: Feature,
        accessible_plugins: FeatureGroupEnvironmentMapping,
        links: Optional[set[Link]],
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> EvaluationResult:
        """Run the matching/filter logic without raising, returning a structured result."""
        self = cls.__new__(cls)
        self._criteria_matched_feature_groups = set()
        self._abstract_matched_feature_groups = set()
        self._candidate_frameworks = {}
        self._domain_outcomes = {}
        self._declared_frameworks = {}
        self._data_access_collection = data_access_collection
        identified = self._filter_loop(feature, accessible_plugins, links, data_access_collection)
        result = EvaluationResult(
            identified=identified,
            criteria_matched=self._criteria_matched_feature_groups,
            abstract_matched=self._abstract_matched_feature_groups,
            candidate_frameworks=self._candidate_frameworks,
        )
        if result.failure_kind is not None:
            result = replace(result, facts=self._capture_render_facts(result, feature, accessible_plugins))
        # A captured exception pins its traceback, whose frames pin this instance: a refcount cycle that would
        # keep both alive until a gc pass. Dropping the outcomes here makes the memo's lifetime what it claims.
        self._domain_outcomes.clear()
        return result

    def _capture_render_facts(
        self,
        result: EvaluationResult,
        feature: Feature,
        accessible_plugins: FeatureGroupEnvironmentMapping,
    ) -> RenderFacts:
        """Capture what rendering needs, following the branch precedence the message builders use today.

        Only reached when the pass has no single winner, so the success path stays hook-free. Every provider
        hook here is best-effort: a raising one degrades its own fact, never this call or a sibling's fact.
        """
        environment_empty = not accessible_plugins

        if result.failure_kind == "multiple":
            return RenderFacts(environment_empty=environment_empty, domains=self._capture_domains(result))

        # Capability rejections win over the abstract-only fallback, and both over the ordinary none.
        capability_frameworks = self._capture_capability_frameworks(result, feature)
        if any(candidate.rejected for candidate in capability_frameworks.values()):
            return RenderFacts(environment_empty=environment_empty, capability_frameworks=capability_frameworks)

        if result.abstract_matched:
            return RenderFacts(
                environment_empty=environment_empty,
                concrete_frameworks=self._concrete_implementation_frameworks(result, accessible_plugins),
            )

        return RenderFacts(
            environment_empty=environment_empty,
            value_rejections=self._capture_value_rejections(feature, accessible_plugins),
            known_names=self._capture_known_names(accessible_plugins),
        )

    def _domain_outcome(self, feature_group: type[FeatureGroup]) -> tuple[Optional[Domain], Optional[Exception]]:
        """Memoized get_domain() OUTCOME, value or raise, so one candidate's hook runs once per evaluation.

        The outcome rather than the value, because the two readers disagree on error semantics: the decision
        filter re-raises, the render capture degrades. Caching successes only would re-call a raising hook.

        Unlike safe_field, this retains the exception object, not str(exc): re-raising it needs the object.
        That pins a traceback and its frames, so evaluate() clears this memo before returning rather than
        leaving the cycle for the collector.
        """
        if feature_group not in self._domain_outcomes:
            try:
                self._domain_outcomes[feature_group] = (feature_group.get_domain(), None)
            except Exception as exc:  # noqa: BLE001  (outcome capture; each reader decides how to react)
                self._domain_outcomes[feature_group] = (None, exc)
        return self._domain_outcomes[feature_group]

    def _domain_name(self, feature_group: type[FeatureGroup]) -> str | None:
        """Best-effort domain name. None when get_domain() raised or returned no Domain: renders without a suffix."""
        field = f"{feature_group.get_class_name()}.get_domain"
        domain, error = self._domain_outcome(feature_group)
        # error, not domain, is what tells a raise apart from a malformed return: both leave domain unusable.
        if error is not None:
            logger.warning("Degraded field '%s': %s: %s", field, type(error).__name__, str(error))
            return None
        if not isinstance(domain, Domain):
            # Annotated to return a Domain; a provider that returns something else costs only its own suffix.
            logger.warning("Degraded field '%s': expected Domain, got %s", field, type(domain).__name__)
            return None
        return domain.name

    def _declared_frameworks_of(self, feature_group: type[FeatureGroup]) -> frozenset[type[ComputeFramework]]:
        """Memoized compute_framework_definition(), which drives compute_framework_rule(): once per candidate.

        Best-effort, and both readers guard it identically, so the value alone is enough to cache.
        """
        if feature_group not in self._declared_frameworks:
            self._declared_frameworks[feature_group] = safe_field(
                lambda: frozenset(feature_group.compute_framework_definition()),
                frozenset(),
                field=f"{feature_group.get_class_name()}.compute_framework_definition",
            )
        return self._declared_frameworks[feature_group]

    def _declared_framework_names(self, feature_group: type[FeatureGroup]) -> set[str]:
        """Best-effort names of every framework one candidate declares, available or not, as the message wants them.

        Guards the projection, not just the declaration read: a declaration holding something that is not a
        ComputeFramework costs the whole candidate its names, well-formed entries included, as before the memo.
        """
        return safe_field(
            lambda: {_as_str(cfw.get_class_name()) for cfw in self._declared_frameworks_of(feature_group)},
            set(),
            field=f"{feature_group.get_class_name()}.compute_framework_definition",
        )

    def _available_declared_frameworks(self, feature_group: type[FeatureGroup]) -> frozenset[type[ComputeFramework]]:
        """Best-effort render universe of one candidate: the frameworks it declares that are available."""
        return safe_field(
            lambda: frozenset(cfw for cfw in self._declared_frameworks_of(feature_group) if cfw.is_available()),
            frozenset(),
            field=f"{feature_group.get_class_name()}.is_available",
        )

    def _capture_domains(self, result: EvaluationResult) -> dict[type[FeatureGroup], str]:
        """Domain name of every identified candidate, skipping the ones whose get_domain() raised."""
        domains: dict[type[FeatureGroup], str] = {}
        for feature_group in result.identified:
            domain = self._domain_name(feature_group)
            if domain is not None:
                domains[feature_group] = domain
        return domains

    def _capture_capability_frameworks(
        self, result: EvaluationResult, feature: Feature
    ) -> dict[type[FeatureGroup], CandidateFrameworks]:
        """Split each criteria-matched candidate's declared available frameworks, the universe the message uses.

        Starts from the decision split, which already answered the enabled frameworks, and only asks about the
        rest: the capability hook is asked at most once per (candidate, framework) pair per evaluation.
        """
        capability_frameworks: dict[type[FeatureGroup], CandidateFrameworks] = {}
        for feature_group in result.criteria_matched:
            decided = result.candidate_frameworks.get(feature_group, CandidateFrameworks())
            declared = self._available_declared_frameworks(feature_group)
            supported = set(decided.supported & declared)
            rejected = set(decided.rejected & declared)

            for cfw in declared - decided.supported - decided.rejected:
                verdict = _supports_framework(feature_group, feature, cfw)
                if verdict is None:
                    continue
                if verdict:
                    supported.add(cfw)
                else:
                    rejected.add(cfw)

            capability_frameworks[feature_group] = CandidateFrameworks(
                supported=frozenset(supported), rejected=frozenset(rejected)
            )
        return capability_frameworks

    def _concrete_implementation_frameworks(
        self, result: EvaluationResult, accessible_plugins: FeatureGroupEnvironmentMapping
    ) -> tuple[str, ...]:
        """Frameworks declared by the accessible concrete subclasses of an abstract-matched base."""
        frameworks: set[str] = set()
        for candidate in accessible_plugins:
            if inspect.isabstract(candidate):
                continue
            if not any(issubclass(candidate, abstract_fg) for abstract_fg in result.abstract_matched):
                continue
            frameworks.update(self._declared_framework_names(candidate))
        return tuple(sorted(frameworks))

    def _capture_value_rejections(
        self, feature: Feature, accessible_plugins: FeatureGroupEnvironmentMapping
    ) -> tuple[tuple[str, str], ...]:
        reasons: list[tuple[str, str]] = []
        for feature_group in accessible_plugins:
            reason = self._scoped_value_rejection_reason(feature_group, feature)
            if reason is not None:
                reasons.append((feature_group.get_class_name(), reason))
        return tuple(sorted(reasons))

    def _scoped_value_rejection_reason(self, feature_group: type[FeatureGroup], feature: Feature) -> str | None:
        """Best-effort rejection reason of one candidate, guarded together with the filters it runs."""
        return safe_field(
            lambda: self._in_scope_value_rejection_reason(feature_group, feature),
            None,
            field=f"{feature_group.get_class_name()}._strict_validation_rejection_reason",
        )

    def _in_scope_value_rejection_reason(self, feature_group: type[FeatureGroup], feature: Feature) -> str | None:
        """The candidate's rejection reason, or None when the domain or scope filter rules it out.

        The reason is validated here, inside the caller's guard: a non-str one only detonates later, in the
        sorted() over reasons, where two same-named candidates make it a str/int comparison.
        """
        if not self._filter_feature_group_by_domain(feature_group, feature):
            return None
        if not self._filter_feature_group_by_scope(feature_group, feature):
            return None
        reason = self._value_rejection_reason(feature_group, feature)
        return None if reason is None else _as_str(reason)

    def _capture_known_names(self, accessible_plugins: FeatureGroupEnvironmentMapping) -> tuple[str, ...]:
        known_names: list[str] = []
        for fg_class in accessible_plugins:
            # get_class_name() is @final, so it cannot raise on a provider's behalf and needs no guard.
            known_names.append(fg_class.get_class_name())
            known_names.extend(_supported_feature_names(fg_class))
            prefix = _prefix_name(fg_class)
            if prefix:
                known_names.append(prefix)
        return tuple(known_names)

    def _filter_loop(
        self,
        feature: Feature,
        accessible_plugins: FeatureGroupEnvironmentMapping,
        links: Optional[set[Link]],
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> FeatureGroupEnvironmentMapping:
        _identified_feature_groups: FeatureGroupEnvironmentMapping = {}

        for feature_group, compute_frameworks in accessible_plugins.items():
            if not self._filter_feature_group_by_criteria(feature_group, feature, data_access_collection):
                continue

            if not self._filter_feature_group_by_domain(feature_group, feature):
                continue

            if not self._filter_feature_group_by_scope(feature_group, feature):
                continue

            # Abstract bases can match name+domain+scope but cannot be instantiated; never let one win.
            if inspect.isabstract(feature_group):
                self._abstract_matched_feature_groups.add(feature_group)
                continue

            self._criteria_matched_feature_groups.add(feature_group)

            supported_frameworks = {
                cfw
                for cfw in compute_frameworks
                if feature_group.supports_compute_framework(feature.name, feature.options, cfw)
            }

            # The split the capability hook just produced over this candidate's own accessible frameworks:
            # keeping it costs no extra hook call. frozenset() first: callers may pass any iterable.
            self._candidate_frameworks[feature_group] = CandidateFrameworks(
                supported=frozenset(supported_frameworks),
                rejected=frozenset(compute_frameworks) - frozenset(supported_frameworks),
            )

            if not self._filter_feature_group_by_framework(supported_frameworks, feature):
                continue

            if not self._filter_feature_group_by_links(feature_group, links):
                continue

            if supported_frameworks:
                _identified_feature_groups[feature_group] = supported_frameworks

        _identified_feature_groups = self.filter_subclasses(_identified_feature_groups)
        return _identified_feature_groups

    def _filter_feature_group_by_links(self, feature_group: type[FeatureGroup], links: Optional[set[Link]]) -> bool:
        # Case index columns not given, so no validation possible
        if feature_group.index_columns() is None:
            return True

        # Case no links given, so no validation possible
        if links is None:
            return True

        # Validate that at least one index is supported by the feature group
        for link in links:
            if feature_group.supports_index(link.left_index):
                return True

            if feature_group.supports_index(link.right_index):
                return True

        return False

    def _filter_feature_group_by_criteria(
        self,
        feature_group: type[FeatureGroup],
        feature: Feature,
        data_access_collection: Optional[DataAccessCollection],
    ) -> bool:
        """A rejected option value is a non-match, whoever calls the parser: a candidate that overrides the match
        hook and calls FeatureChainParser directly must not take the whole filter loop down. Only the rejection is
        caught, so a plain ValueError (the forwarded-name-mismatch guidance) still reaches the user.
        """
        try:
            return feature_group.match_feature_group_criteria(feature.name, feature.options, data_access_collection)
        except PropertyValueRejection as exc:
            logger.debug("%s rejected an option value while matching '%s': %s", feature_group, feature.name, exc)
            return False

    def _filter_feature_group_by_domain(self, feature_group: type[FeatureGroup], feature: Feature) -> bool:
        """Decision-side domain gate: unguarded, so a raising get_domain() still fails the engine loudly."""
        if not feature.domain:
            return True
        domain, error = self._domain_outcome(feature_group)
        if error is not None:
            raise error
        return domain == feature.domain

    def _filter_feature_group_by_scope(self, feature_group: type[FeatureGroup], feature: Feature) -> bool:
        scope = feature.feature_group_scope
        return scope is None or matches_feature_group_scope(feature_group, scope)

    def _filter_feature_group_by_framework(
        self,
        compute_frameworks: set[type[ComputeFramework]],
        feature: Feature,
    ) -> bool:
        if feature.compute_frameworks is None:
            return True

        if len(feature.compute_frameworks) > 1:
            raise ValueError(f"Feature should only have one compute framework when set by user {feature.name}.")

        return feature.get_compute_framework() in compute_frameworks

    def _value_rejection_reason(self, feature_group: type[FeatureGroup], feature: Feature) -> Optional[str]:
        """The candidate's own message for rejecting an option VALUE, if it has one."""
        rejection_check = getattr(feature_group, "_strict_validation_rejection_reason", None)
        if rejection_check is None:
            return None
        reason: Optional[str] = rejection_check(feature.name, feature.options)
        return reason

    def get(self) -> tuple[type[FeatureGroup], set[type[ComputeFramework]]]:
        return next(iter(self.feature_group_compute_framework_mapping.items()))

    def filter_subclasses(
        self, _identified_feature_groups: FeatureGroupEnvironmentMapping
    ) -> FeatureGroupEnvironmentMapping:
        """
        This functionality ensures that only subclass feature groups are kept.
        """
        fgs_to_pop: set[type[FeatureGroup]] = set()

        for i_feature_group, i_compute_frameworks in _identified_feature_groups.items():
            for o_feature_group, o_compute_frameworks in _identified_feature_groups.items():
                if i_compute_frameworks != o_compute_frameworks:
                    continue

                if i_feature_group == o_feature_group:
                    continue

                if issubclass(i_feature_group, o_feature_group):
                    fgs_to_pop.add(o_feature_group)

        for fg in fgs_to_pop:
            _identified_feature_groups.pop(fg)

        return _identified_feature_groups
