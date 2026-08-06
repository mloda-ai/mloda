import functools
import inspect
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import replace
from typing import Optional

from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping

# Not a re-export facade: every import here is used by this module and ruff F401 fails any added just to re-export.
from mloda.core.prepare.resolution_types import (
    CandidateFrameworks,
    Elimination,
    EliminationStage,
    EvaluationResult,
    NAME_INDEPENDENT_STAGES,
    PARTIAL_RECORDS_CAP,
    RenderFacts,
    ResolutionRecord,
)
from mloda.core.prepare.resolution_failure_renderer import (
    render_resolution_failure,
    _prefix_name,
    _supported_feature_names,
)
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.domain import Domain
from mloda.core.abstract_plugins.components.match_rejection import (
    INPUT_DATA_OWNED_STAGE,
    INPUT_DATA_STAGE,
    MatchRejection,
)
from mloda.core.abstract_plugins.components.match_hook import probe_match_criteria
from mloda.core.abstract_plugins.components.utils import (
    as_str,
    contained_raise_log_level,
    contained_raise_reason,
    safe_field,
)
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.link import Link

import logging

logger = logging.getLogger(__name__)


class FeatureResolutionError(ValueError):
    """Typed resolution failure carrying the feature name, the EvaluationResult of its single pass,
    and the records resolved before the failure, capped at PARTIAL_RECORDS_CAP."""

    def __init__(
        self,
        message: str,
        feature_name: str,
        result: EvaluationResult,
        partial_records: Sequence[ResolutionRecord] = (),
    ) -> None:
        super().__init__(message)
        self.feature_name = feature_name
        self.result = result
        # Cap then snapshot: slicing before deepcopy keeps the copied payload bounded on huge requests.
        self.partial_records: tuple[ResolutionRecord, ...] = tuple(
            deepcopy(record) for record in partial_records[-PARTIAL_RECORDS_CAP:]
        )

    def __reduce__(
        self,
    ) -> tuple[type["FeatureResolutionError"], tuple[str, str, EvaluationResult, tuple[ResolutionRecord, ...]]]:
        # The default reduction reconstructs from args=(message,) and drops the extra constructor arguments.
        return type(self), (str(self), self.feature_name, self.result, self.partial_records)


class ComputeFrameworkPinError(ValueError):
    """User pinned more than one compute framework; validated before matching (#851)."""


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


class IdentifyFeatureGroupClass:
    _criteria_matched_feature_groups: set[type[FeatureGroup]]
    _abstract_matched_feature_groups: set[type[FeatureGroup]]
    _candidate_frameworks: dict[type[FeatureGroup], CandidateFrameworks]
    _match_rejections: dict[type[FeatureGroup], MatchRejection]
    _matcher_errors: dict[type[FeatureGroup], str]
    _eliminations: dict[type[FeatureGroup], Elimination]
    _data_access_collection: Optional[DataAccessCollection]
    # Per-evaluation memos of the hooks more than one reader wants. evaluate() builds a fresh instance, so
    # they are scoped to one resolution attempt and never cache across runs.
    _domain_outcomes: dict[type[FeatureGroup], tuple[Optional[Domain], Optional[Exception]]]
    _links_outcomes: dict[type[FeatureGroup], tuple[Optional[bool], Optional[Exception]]]
    _declared_frameworks: dict[type[FeatureGroup], frozenset[type[ComputeFramework]]]
    _supported_names: dict[type[FeatureGroup], frozenset[str]]
    _prefixes: dict[type[FeatureGroup], str]

    def __init__(self, data_access_collection: Optional[DataAccessCollection] = None) -> None:
        self._criteria_matched_feature_groups = set()
        self._abstract_matched_feature_groups = set()
        self._candidate_frameworks = {}
        # Reasons the first match pass recorded, keyed by candidate class.
        self._match_rejections = {}
        # Contained matcher raises as text, keyed by candidate class: never the exception object.
        self._matcher_errors = {}
        self._eliminations = {}
        self._domain_outcomes = {}
        self._links_outcomes = {}
        self._declared_frameworks = {}
        self._supported_names = {}
        self._prefixes = {}
        self._data_access_collection = data_access_collection

    @staticmethod
    def _validate_single_framework_pin(feature: Feature) -> None:
        """Raise if the user pinned more than one compute framework; this misuse is only a programmer error."""
        pinned = feature.compute_frameworks
        if pinned is not None and len(pinned) > 1:
            names = ", ".join(sorted(cfw.get_class_name() for cfw in pinned))
            raise ComputeFrameworkPinError(
                f"Feature '{feature.name}' is pinned to more than one compute framework ({names}); "
                f"pin at most one compute framework."
            )

    @classmethod
    def evaluate(
        cls,
        feature: Feature,
        accessible_plugins: FeatureGroupEnvironmentMapping,
        links: Optional[set[Link]],
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> EvaluationResult:
        """Run the matching/filter logic without raising, returning a structured result."""
        # Pre-matching guard: a >1 pin fires regardless of whether any candidate matches (the old check
        # sat inside the filter loop, so it never ran when the name matched nothing).
        cls._validate_single_framework_pin(feature)
        self = cls(data_access_collection)
        try:
            identified = self._filter_loop(feature, accessible_plugins, links, data_access_collection)
            result = EvaluationResult(
                identified=identified,
                criteria_matched=self._criteria_matched_feature_groups,
                abstract_matched=self._abstract_matched_feature_groups,
                candidate_frameworks=self._candidate_frameworks,
                eliminations=self._eliminations,
            )
            if result.failure_kind is not None:
                # Every elimination (value_rejection included) was already recorded during the single filter pass;
                # this only captures the non-elimination facts the messages still need.
                result = replace(result, facts=self._capture_render_facts(result, accessible_plugins, feature, links))
        finally:
            # A captured exception pins its traceback, whose frames pin this instance: a refcount cycle that would
            # keep both alive until a gc pass. Dropping the outcomes makes each memo's lifetime what it claims,
            # in a finally because a re-raising gate or an escalated match abort leaves without a return.
            self._domain_outcomes.clear()
            self._links_outcomes.clear()
        return result

    def _capture_render_facts(
        self,
        result: EvaluationResult,
        accessible_plugins: FeatureGroupEnvironmentMapping,
        feature: Feature,
        links: Optional[set[Link]],
    ) -> RenderFacts:
        """Capture the non-elimination facts the messages still need. Only reached when the pass has no winner.

        The renderer alone owns which message wins, so this does not mirror its branch order: the four cheap
        facts are captured whatever the failure kind is. domains feeds the multiple message, concrete_frameworks
        the abstract_only message, and known_names, eliminated_hints and dead_only_names the none message.
        dead_only_names is the one exception, gated on its own kind: its sweep retests the links gate over every
        accessible plugin, which on a linked run costs provider hooks per candidate for a fact only the none
        message reads. Every provider hook here is best-effort: a raising one degrades its own fact, never this
        call or a sibling's fact.
        """
        return RenderFacts(
            domains=self._capture_domains(result),
            concrete_frameworks=self._concrete_implementation_frameworks(result, accessible_plugins),
            known_names=self._capture_known_names(accessible_plugins),
            eliminated_hints=self._capture_eliminated_hints(result),
            dead_only_names=(
                self._capture_dead_only_names(result, accessible_plugins, feature, links)
                if result.failure_kind == "none"
                else frozenset()
            ),
        )

    def _capture_eliminated_hints(self, result: EvaluationResult) -> frozenset[str]:
        """Class name and prefix of every eliminated near-miss, so the none message can suppress a
        'Did you mean' suggestion that merely echoes a candidate its near-miss block already names."""
        hints: set[str] = set()
        for feature_group in result.eliminations:
            hints.add(feature_group.get_class_name())
            prefix = self._prefix_of(feature_group)
            if prefix:
                hints.add(prefix)
        return frozenset(hints)

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
            lambda: {as_str(cfw.get_class_name()) for cfw in self._declared_frameworks_of(feature_group)},
            set(),
            field=f"{feature_group.get_class_name()}.compute_framework_definition",
        )

    def _capture_domains(self, result: EvaluationResult) -> dict[type[FeatureGroup], str]:
        """Domain name of every identified candidate, skipping the ones whose get_domain() raised."""
        domains: dict[type[FeatureGroup], str] = {}
        for feature_group in result.identified:
            domain = self._domain_name(feature_group)
            if domain is not None:
                domains[feature_group] = domain
        return domains

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

    def _supported_names_of(self, feature_group: type[FeatureGroup]) -> frozenset[str]:
        """Memoized feature_names_supported(): the name catalog and the dead-name capture share one call.

        Degrades exactly as _supported_feature_names does: a raising hook costs that candidate its whole catalog.
        """
        if feature_group not in self._supported_names:
            self._supported_names[feature_group] = frozenset(_supported_feature_names(feature_group))
        return self._supported_names[feature_group]

    def _prefix_of(self, feature_group: type[FeatureGroup]) -> str:
        """Memoized prefix(), read by the name catalog and by the live side of the dead-name difference."""
        if feature_group not in self._prefixes:
            self._prefixes[feature_group] = _prefix_name(feature_group)
        return self._prefixes[feature_group]

    def _catalog_names_of(self, feature_group: type[FeatureGroup]) -> list[str]:
        """One candidate's whole contribution to the name catalog, in capture order."""
        # get_class_name() is @final, so it cannot raise on a provider's behalf and needs no guard.
        names = [feature_group.get_class_name(), *self._supported_names_of(feature_group)]
        prefix = self._prefix_of(feature_group)
        if prefix:
            names.append(prefix)
        return names

    def _capture_known_names(self, accessible_plugins: FeatureGroupEnvironmentMapping) -> tuple[str, ...]:
        known_names: list[str] = []
        for fg_class in accessible_plugins:
            known_names.extend(self._catalog_names_of(fg_class))
        return tuple(known_names)

    def _fails_name_blind_gate(
        self, feature_group: type[FeatureGroup], feature: Feature, links: Optional[set[Link]]
    ) -> bool:
        """Capture-side retest of the three name-blind gates, scope then domain then links, that never raises.

        Links last because it is the costliest: the only one of the three reading a provider hook that neither
        the catalog nor a sibling fact already needs.
        """
        if not self._filter_feature_group_by_scope(feature_group, feature):
            return True
        if self._fails_domain_gate(feature_group, feature):
            return True
        if links is None:
            # Without links the gate passes every candidate, so it decides nothing: returning before the memo
            # is what keeps a link-free run from reading index_columns() at all.
            return False
        # The gate reads two hooks and the guard cannot tell which raised, so the report names the pair rather
        # than the one it starts with. The fallback leaves the gate undecided: an unreadable index is not a lost
        # gate, so the candidate stays live.
        return not safe_field(
            lambda: self._filter_feature_group_by_links(feature_group, links),
            True,
            field=f"{feature_group.get_class_name()}.index_columns/supports_index",
        )

    def _fails_domain_gate(self, feature_group: type[FeatureGroup], feature: Feature) -> bool:
        """Capture-side retest of the domain gate alone, which only fires for a domain-carrying request."""
        if feature.domain is None:
            return False
        domain, error = self._domain_outcome(feature_group)
        # _domain_name is what reports either degrade, and both readers share one memo, so reading through it
        # keeps this at a single get_domain() call per candidate.
        if self._domain_name(feature_group) is None:
            # A raise leaves the gate undecided, so nothing is known and the candidate stays live. A malformed
            # return is decided: the gate compares it and drops the candidate, for every name it declares.
            return error is None
        # The gate's own comparison, never a name one: a Domain subclass with a custom __eq__ must not pass the
        # gate and fail this retest.
        return domain != feature.domain

    def _is_dead(
        self,
        result: EvaluationResult,
        feature_group: type[FeatureGroup],
        compute_frameworks: set[type[ComputeFramework]],
        feature: Feature,
        links: Optional[set[Link]],
    ) -> bool:
        """No name at all can resolve to this candidate: it has no framework left, or it lost at a name-blind gate.

        The framework set is tested before the elimination lookup because it kills the candidate without one:
        a record only exists for the name that was asked about, while an empty set routes EVERY name to
        frameworks_not_enabled.
        """
        if not compute_frameworks:
            return True
        # _filter_loop parks an abstract base in abstract_matched and never in the identified mapping, whatever
        # name it is asked about.
        if inspect.isabstract(feature_group):
            return True
        # A candidate that never matched the requested name carries no record at all, so the name-blind gates
        # are retested directly here: scope, domain and links, the empty framework set above being the fourth.
        if self._fails_name_blind_gate(feature_group, feature, links):
            return True
        elimination = result.eliminations.get(feature_group)
        return elimination is not None and elimination.stage in NAME_INDEPENDENT_STAGES

    def _capture_dead_only_names(
        self,
        result: EvaluationResult,
        accessible_plugins: FeatureGroupEnvironmentMapping,
        feature: Feature,
        links: Optional[set[Link]],
    ) -> frozenset[str]:
        """Catalog names of dead candidates that no live candidate owns and no live candidate's prefix covers.

        A difference, not a per-candidate drop: one accessible group that is still alive keeps its name
        suggestible, whatever a dead sibling also declares. Prefixes because the default matcher owns names
        by class-name prefix too, so a live group covers names it never declares. The matcher's data-access
        branches stay out: deciding one needs the speculative second match the renderer's contract refuses.
        """
        dead: set[str] = set()
        live: set[str] = set()
        live_prefixes: set[str] = set()
        for feature_group, compute_frameworks in accessible_plugins.items():
            if self._is_dead(result, feature_group, compute_frameworks, feature, links):
                # The whole catalog, as the live branch collects it: the default matcher owns a candidate's
                # class name and its class-name prefix too, and a dead candidate resolves neither.
                dead.update(self._catalog_names_of(feature_group))
                continue
            live.update(self._catalog_names_of(feature_group))
            # Non-empty only: _prefix_name degrades a raising prefix() to "", which every name starts with.
            prefix = self._prefix_of(feature_group)
            if prefix:
                live_prefixes.add(prefix)
        return frozenset(name for name in dead - live if not any(name.startswith(prefix) for prefix in live_prefixes))

    def _filter_loop(
        self,
        feature: Feature,
        accessible_plugins: FeatureGroupEnvironmentMapping,
        links: Optional[set[Link]],
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> FeatureGroupEnvironmentMapping:
        _identified_feature_groups: FeatureGroupEnvironmentMapping = {}

        for feature_group, compute_frameworks in accessible_plugins.items():
            # A criteria non-match records a value_rejection only when the first pass recorded a reason for it:
            # a plain name mismatch is not a near-miss, but a value the candidate declined (with a reportable
            # reason) is. The criteria call above just recorded any rejection under this candidate's window, so
            # this reads it back for a criteria-FAILING candidate only; a matched/winning/abstract candidate is
            # never probed. Recorded regardless of domain/scope or of the overall outcome (a sibling may win).
            if not self._filter_feature_group_by_criteria(feature_group, feature, data_access_collection):
                # A contained matcher raise is always a near-miss: the raise says nothing about name ownership.
                # Deliberate precedence: a contained crash outranks a recorded decline for the same candidate.
                matcher_error = self._matcher_errors.get(feature_group)
                if matcher_error is not None:
                    self._record_elimination(feature_group, "matcher_error", matcher_error)
                    continue
                rejection = self._value_rejection(feature_group)
                if rejection is not None:
                    # The stage is a free-form hint; only the two input-data stages are engine-known and both
                    # surface as the public "input_data" elimination stage, the rest fall back.
                    stage: EliminationStage = (
                        "input_data"
                        if rejection.stage in (INPUT_DATA_STAGE, INPUT_DATA_OWNED_STAGE)
                        else "value_rejection"
                    )
                    self._record_elimination(feature_group, stage, rejection.reason)
                continue

            if not self._filter_feature_group_by_domain(feature_group, feature):
                self._record_elimination(feature_group, "domain", self._domain_reason(feature_group, feature))
                continue

            if not self._filter_feature_group_by_scope(feature_group, feature):
                self._record_elimination(feature_group, "scope", "outside the requested feature group scope")
                continue

            # Abstract bases can match name+domain+scope but cannot be instantiated; never let one win, and
            # never record one as a near-miss: the abstract_only message owns them.
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

            # Decide the empty-supported case FIRST so a pin over an empty supported set reports the deeper
            # capability / not-enabled reason rather than framework_pin. The identification decision (all three
            # gates must hold) is order-independent, so this reordering only changes which reason is recorded.
            if not supported_frameworks:
                if compute_frameworks:
                    rejected_names = sorted(cfw.get_class_name() for cfw in compute_frameworks)
                    self._record_elimination(
                        feature_group, "capability", f"supports_compute_framework rejected {rejected_names}"
                    )
                else:
                    self._record_elimination(
                        feature_group,
                        "frameworks_not_enabled",
                        "none of its compute frameworks are enabled for this run",
                    )
                continue

            if not self._filter_feature_group_by_framework(supported_frameworks, feature):
                pin_name = feature.get_compute_framework().get_class_name()
                supported_names = sorted(cfw.get_class_name() for cfw in supported_frameworks)
                self._record_elimination(
                    feature_group,
                    "framework_pin",
                    f"pinned compute framework '{pin_name}' is not among its supported {supported_names}",
                )
                continue

            if not self._filter_feature_group_by_links(feature_group, links):
                self._record_elimination(feature_group, "links", "no index column matches the run's links")
                continue

            _identified_feature_groups[feature_group] = supported_frameworks

        _identified_feature_groups = self.filter_subclasses(_identified_feature_groups)
        return _identified_feature_groups

    def _record_elimination(self, feature_group: type[FeatureGroup], stage: EliminationStage, reason: str) -> None:
        """Record the first gate a non-winning name-matching candidate failed; one entry per candidate."""
        self._eliminations.setdefault(feature_group, Elimination(stage=stage, reason=reason))

    def _value_rejection(self, feature_group: type[FeatureGroup]) -> Optional[MatchRejection]:
        """The MatchRejection the first match pass recorded for this candidate class, if any.

        The candidate's criteria match records its own rejection under a per-candidate window; this
        only reads that record back, so no rejection hook is ever reran on the failure path.
        """
        return self._match_rejections.get(feature_group)

    def _domain_reason(self, feature_group: type[FeatureGroup], feature: Feature) -> str:
        """Reason wording for a candidate dropped at the domain gate, which only fires for a domain-carrying request."""
        assert feature.domain is not None  # the domain gate only drops a candidate when the request carries a domain
        requested = feature.domain.name
        candidate_domain = self._domain_name(feature_group)
        if candidate_domain is None:
            return f"does not declare the requested domain '{requested}'"
        return f"declares domain '{candidate_domain}', but the run requested '{requested}'"

    def _filter_feature_group_by_links(self, feature_group: type[FeatureGroup], links: Optional[set[Link]]) -> bool:
        """Decision-side links gate: unguarded, so a raising index hook still fails the engine loudly."""
        supported, error = self._links_outcome(feature_group, links)
        if error is not None:
            raise error
        # None is the memo's unreadable marker, never a verdict, so an outcome without an error always has one.
        assert supported is not None
        return supported

    def _links_outcome(
        self, feature_group: type[FeatureGroup], links: Optional[set[Link]]
    ) -> tuple[Optional[bool], Optional[Exception]]:
        """Memoized links-gate OUTCOME, verdict or raise, so one candidate's index hooks run once per evaluation.

        The outcome rather than the verdict, for the reason _domain_outcome caches one: the decision filter
        re-raises, the render capture degrades. The candidate alone keys it, because links is one value for the
        whole evaluation. Retains the exception object, so evaluate() clears this memo as it clears that one.

        The verdict is None, not False, when the gate raised: unreadable is not lost, and a reader that skipped
        the error check would otherwise read a raise as a candidate that failed the gate.
        """
        if feature_group not in self._links_outcomes:
            try:
                self._links_outcomes[feature_group] = (self._links_gate(feature_group, links), None)
            except Exception as exc:  # noqa: BLE001  (outcome capture; each reader decides how to react)
                self._links_outcomes[feature_group] = (None, exc)
        return self._links_outcomes[feature_group]

    @staticmethod
    def _links_gate(feature_group: type[FeatureGroup], links: Optional[set[Link]]) -> bool:
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
        """A raise out of the match hook is a non-match for that candidate only, not a run-wide abort (#845).

        The shared probe owns the per-candidate window and the containment; this seam keeps only its own
        policy: the option rollback on a contained raise and the per-candidate recording, never as an
        exception object whose traceback would pin the plugin class.

        Mark-or-contain policy: see call_match_hook.
        """
        # Shallow copies, taken per candidate so an earlier match's write survives a later candidate's raise.
        group_before = dict(feature.options.group)
        context_before = dict(feature.options.context)
        probe = probe_match_criteria(feature_group, feature.name, feature.options, data_access_collection)
        if probe.matcher_error is not None or probe.value_rejection is not None:
            # Only the contained branch rolls back: a matcher that returns True keeps its write,
            # which is how a matched reader is linked through mloda.
            feature.options.group.clear()
            feature.options.group.update(group_before)
            feature.options.context.clear()
            feature.options.context.update(context_before)
        if probe.value_rejection is not None:
            exc = probe.value_rejection
            # Text, not exc: a retained record must not pin the traceback, its frames and the plugin class.
            logger.debug(
                "%s rejected an option value while matching '%s': %s",
                feature_group.get_class_name(),
                feature.name,
                safe_field(functools.partial(str, exc), type(exc).__name__),
            )
        elif probe.matcher_error is not None:
            reason = contained_raise_reason(probe.matcher_error)
            logger.log(
                contained_raise_log_level(probe.matcher_error),
                "%s %s while matching '%s'; treating it as a non-match.",
                feature_group.get_class_name(),
                reason,
                feature.name,
            )
            self._matcher_errors[feature_group] = reason
        if not probe.matched and probe.rejection is not None:
            # Everything recorded during this candidate's window belongs to this candidate, whatever
            # owner name an inner delegation stamped.
            self._match_rejections[feature_group] = probe.rejection
        return probe.matched

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
        # Cardinality (<=1) is validated up front in evaluate(), so no >1 pin reaches here.
        if feature.compute_frameworks is None:
            return True

        return feature.get_compute_framework() in compute_frameworks

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


def evaluate_and_render(
    feature: Feature,
    accessible_plugins: FeatureGroupEnvironmentMapping,
    links: Optional[set[Link]] = None,
    data_access_collection: Optional[DataAccessCollection] = None,
) -> tuple[EvaluationResult, str | None]:
    """One resolution pass plus its failure message; the message is None iff the feature resolved."""
    # Unguarded: ComputeFrameworkPinError is a misuse validated before matching, so it escapes unconverted.
    result = IdentifyFeatureGroupClass.evaluate(feature, accessible_plugins, links, data_access_collection)
    return result, render_resolution_failure(result, feature)


def resolve_or_raise(
    feature: Feature,
    accessible_plugins: FeatureGroupEnvironmentMapping,
    links: Optional[set[Link]] = None,
    data_access_collection: Optional[DataAccessCollection] = None,
    partial_records: Sequence[ResolutionRecord] = (),
) -> EvaluationResult:
    """Evaluate one feature and raise the typed FeatureResolutionError on failure."""
    result, message = evaluate_and_render(feature, accessible_plugins, links, data_access_collection)
    if message is not None:
        # The constructor does the cap-then-deepcopy snapshot, so the records are forwarded as they are.
        raise FeatureResolutionError(message, str(feature.name), result, partial_records=partial_records)
    return result
