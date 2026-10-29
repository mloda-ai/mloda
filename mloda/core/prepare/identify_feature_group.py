import inspect
import os
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from copy import copy, deepcopy
from difflib import get_close_matches
from dataclasses import dataclass, replace
from typing import Any, cast

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
    rejection_elimination_stage,
)
from mloda.core.prepare.resolution_failure_renderer import (
    render_resolution_failure,
    _candidate_sort_key,
    _prefix_name,
    _supported_feature_names,
)
from mloda.core.abstract_plugins.components.credential_scrub import scrub_credentials
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.declared_attributes import DeclarationRequirement
from mloda.core.abstract_plugins.components.input_data.claim_route import (
    ChainCheck,
    ChainCheckNeeded,
    DataAccessReader,
    chain_check_installed,
    chain_check_scope,
    chain_check_uses,
    computed_elsewhere,
    feature_group_scope,
    open_claim_scope,
    pin_report_scope,
)
from mloda.core.abstract_plugins.components.input_data.source_feature_group import (
    SourceFeatureGroup,
    SourcePointerError,
    concrete_source_groups,
)
from mloda.core.abstract_plugins.components.domain import Domain
from mloda.core.abstract_plugins.components.options import Options, reject_reserved_reader_key
from mloda.core.abstract_plugins.components.match_rejection import MatchRejection
from mloda.core.abstract_plugins.components.match_hook import CriteriaProbeOutcome, probe_match_criteria
from mloda.core.abstract_plugins.components.input_data.match_cache import run_match_cache
from mloda.core.abstract_plugins.components.validators.link_validator import LinkValidator
from mloda.core.prepare.input_chain import (
    EnvironmentKey,
    InputChainOutcome,
    cached_input_fact,
    declared_inputs,
    input_chain_state,
    input_features_collection,
    normalize_input_name,
    resolution_environment,
    resolution_key,
)
from mloda.core.abstract_plugins.components.utils import (
    as_str,
    contained_raise_log_level,
    escalate_match_abort,
    is_deferred_match_abort,
    is_match_abort,
    contained_raise_reason,
    get_all_subclasses,
    safe_exc_str,
    safe_field,
)
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.link import Link
from mloda.core.abstract_plugins.plugin_loader.plugin_loader import PluginLoader
from mloda.core.abstract_plugins.implementation import current_implementations, loaded_implementations
from mloda.core.abstract_plugins.operation_feature_group import OperationFeatureGroup

import logging


def _operation_implementation_reason(feature_group: type[FeatureGroup]) -> str | None:
    """Why an operation group has no accessible implementation, else None."""
    if not issubclass(feature_group, OperationFeatureGroup):
        return None
    if not any(issubclass(feature_group, impl.feature_group) for impl in loaded_implementations()):
        return "no Implementation of it is loaded; import its implementation modules or call PluginLoader.all()"
    if not any(issubclass(feature_group, impl.feature_group) for impl in current_implementations()):
        return "its implementations are all disabled or denied for this run"
    return None


logger = logging.getLogger(__name__)


InputResolver = Callable[[Feature, set[Link] | None, int], EvaluationResult]


def _first_line(text: str) -> str:
    lines = text.strip().splitlines()
    return lines[0] if lines else ""


@dataclass(frozen=True)
class _CriteriaAttempt:
    probe: CriteriaProbeOutcome
    options: Options
    open_claim: bool
    pending_pins: tuple[tuple[str, Any], ...]
    consulted: bool | None = None


def _unique_refusals(refusals: Sequence[Exception]) -> tuple[Exception, ...]:
    seen: set[int] = set()
    unique: list[Exception] = []
    for refusal in refusals:
        if id(refusal) not in seen:
            seen.add(id(refusal))
            unique.append(refusal)
    return tuple(unique)


class FeatureResolutionError(ValueError):
    """Typed resolution failure carrying the feature name, the EvaluationResult of its single pass,
    and the records resolved before the failure, capped at PARTIAL_RECORDS_CAP."""

    def __init__(
        self,
        message: str,
        feature_name: str,
        result: EvaluationResult,
        partial_records: Sequence[ResolutionRecord] = (),
        refusals: Sequence[Exception] = (),
    ) -> None:
        super().__init__(message)
        self.refusals: tuple[Exception, ...] = tuple(refusals)
        if self.refusals:
            self.__cause__ = self.refusals[0]
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
    # Name first: __name__ is an attribute read no plugin can raise out of, while issubclass() on an ABCMeta
    # class is the expensive check, so the name gate keeps it off nearly every MRO entry.
    return any(
        ancestor.__name__ == scope and ancestor is not FeatureGroup and issubclass(ancestor, FeatureGroup)
        for ancestor in feature_group.__mro__
    )


def validate_single_framework_pin(feature: Feature) -> None:
    """Raise if the user pinned more than one compute framework; this misuse is only a programmer error."""
    pinned = feature.compute_frameworks
    if pinned is not None and len(pinned) > 1:
        names = ", ".join(sorted(cfw.get_class_name() for cfw in pinned))
        raise ComputeFrameworkPinError(
            f"Feature '{feature.name}' is pinned to more than one compute framework ({names}); "
            f"pin at most one compute framework."
        )


_NOT_ACCESSIBLE = "it is not among this run's accessible feature groups"


def _pinned_path(feature: Feature, collection: DataAccessCollection | None) -> str | None:
    if collection is None or collection.column_to_file is None:
        return None
    handle = collection.column_to_file.get(str(feature.name))
    return None if handle is None else collection.files.get(handle)


def _pin_conflict(
    feature: Feature, collection: DataAccessCollection | None, loaded: dict[str, type[SourceFeatureGroup]]
) -> str | None:
    from mloda.core.abstract_plugins.components.input_data.file_source_fg import _FileSourceFG

    pinned = _pinned_path(feature, collection)
    if pinned is None:
        return None
    pinned_abs = os.path.abspath(pinned)
    for key in sorted(k for k in feature.options.keys() if isinstance(k, str)):
        group = loaded.get(key)
        if group is None:
            continue
        value = feature.options[key]
        if issubclass(group, _FileSourceFG) and isinstance(value, (str, os.PathLike)):
            pointer = os.path.abspath(value)
            if pointer == pinned_abs:
                continue
            return (
                f"options point {key} at {pointer}, but column_to_file pins '{feature.name}' to {pinned_abs}; "
                "drop the pointer or the pin."
            )
        return (
            f"options point at {key}, but column_to_file pins '{feature.name}' to {pinned_abs}; "
            "drop the pointer or the pin."
        )
    return None


def validate_source_pointers(
    feature: Feature,
    accessible_plugins: FeatureGroupEnvironmentMapping,
    excluded: Mapping[type[FeatureGroup], str] | None,
    data_access_collection: DataAccessCollection | None,
) -> None:
    """Raise SourcePointerError for an unreachable pointer or scope, or a pointer against a pin."""
    scope = feature.feature_group_scope
    keys = sorted(key for key in feature.options.keys() if isinstance(key, str))
    if not keys and scope is None:
        return
    excluded = excluded or {}
    loaded = concrete_source_groups()
    accessible_sources = [fg for fg in accessible_plugins if issubclass(fg, SourceFeatureGroup)]
    for key in keys:
        group = loaded.get(key)
        if group is None:
            near = [name for name in loaded if name.casefold() == key.casefold()]
            if len(near) == 1:
                raise SourcePointerError(f"options key {key} names no source group; did you mean {near[0]}?")
        if group is not None and not any(key in fg.pointer_keys() for fg in accessible_sources):
            reason = excluded.get(group, _NOT_ACCESSIBLE)
            raise SourcePointerError(f"options point at {key}, but {key} is not accessible in this run: {reason}.")
    if scope is not None:
        scope_name = scope if isinstance(scope, str) else scope.__name__
        group = loaded.get(scope_name)
        if group is not None and not any(matches_feature_group_scope(fg, scope) for fg in accessible_plugins):
            reason = excluded.get(group, _NOT_ACCESSIBLE)
            raise SourcePointerError(
                f"feature_group={scope_name}, but {scope_name} is not accessible in this run: {reason}."
            )
        for key in keys:
            if key in loaded and not any(
                matches_feature_group_scope(fg, scope) and key in fg.pointer_keys() for fg in accessible_sources
            ):
                raise SourcePointerError(
                    f"feature_group={scope_name} and an options pointer at {key} name different groups; "
                    "drop one of them."
                )
    conflict = _pin_conflict(feature, data_access_collection, loaded)
    if conflict is not None:
        raise SourcePointerError(conflict)


class IdentifyFeatureGroupClass:
    _criteria_matched_feature_groups: set[type[FeatureGroup]]
    _abstract_matched_feature_groups: set[type[FeatureGroup]]
    _candidate_frameworks: dict[type[FeatureGroup], CandidateFrameworks]
    _match_rejections: dict[type[FeatureGroup], MatchRejection]
    _matcher_errors: dict[type[FeatureGroup], str]
    _eliminations: dict[type[FeatureGroup], Elimination]
    _data_access_collection: DataAccessCollection | None
    # Per-evaluation memos of the hooks more than one reader wants. evaluate() builds a fresh instance, so
    # they are scoped to one resolution attempt and never cache across runs.
    _domain_outcomes: dict[type[FeatureGroup], tuple[Domain | None, Exception | None]]
    _links_outcomes: dict[type[FeatureGroup], tuple[bool | None, Exception | None]]
    _declared_frameworks: dict[type[FeatureGroup], frozenset[type[ComputeFramework]]]
    _supported_names: dict[type[FeatureGroup], frozenset[str]]
    _prefixes: dict[type[FeatureGroup], str]
    _declarations: dict[type, tuple[dict[str, str | int | float | bool], str | None]]
    _replaced: set[type[FeatureGroup]]
    _matched_options: dict[type[FeatureGroup], Options]
    _output_parts: dict[type[FeatureGroup], tuple[str, str]]
    _deferred_aborts: dict[type[FeatureGroup], Exception]
    _input_data_matches: dict[type[FeatureGroup], tuple[DataAccessReader, Any]]
    _open_claims: set[type[FeatureGroup]]
    _reported_pins: set[str]
    _partial_records: Sequence[ResolutionRecord]
    _excluded: Mapping[type[FeatureGroup], str]

    def __init__(self, data_access_collection: DataAccessCollection | None = None) -> None:
        self._excluded = {}
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
        self._declarations = {}
        self._replaced = set()
        self._matched_options = {}
        self._output_parts = {}
        self._deferred_aborts = {}
        self._input_data_matches = {}
        self._open_claims = set()
        self._reported_pins = set()
        self._partial_records = ()
        self._data_access_collection = data_access_collection
        self._accessible_plugins: FeatureGroupEnvironmentMapping = {}
        self._links: set[Link] | None = None
        self._environment: EnvironmentKey = ()
        self._input_resolver: InputResolver | None = None
        self._input_depth = 0
        self._input_refusals: dict[type[FeatureGroup], tuple[Exception, ...]] = {}

    @classmethod
    def evaluate(
        cls,
        feature: Feature,
        accessible_plugins: FeatureGroupEnvironmentMapping,
        links: set[Link] | None,
        data_access_collection: DataAccessCollection | None = None,
        *,
        excluded: Mapping[type[FeatureGroup], str] | None = None,
    ) -> EvaluationResult:
        """Run the matching/filter logic returning a structured result; only request misuse raises."""
        return cls._evaluate(feature, accessible_plugins, links, data_access_collection, excluded=excluded)[0]

    @classmethod
    def _evaluate(
        cls,
        feature: Feature,
        accessible_plugins: FeatureGroupEnvironmentMapping,
        links: set[Link] | None,
        data_access_collection: DataAccessCollection | None = None,
        *,
        excluded: Mapping[type[FeatureGroup], str] | None = None,
        partial_records: Sequence[ResolutionRecord] = (),
        _input_resolver: InputResolver | None = None,
        _input_depth: int = 0,
    ) -> tuple[EvaluationResult, tuple[Exception, ...]]:
        """evaluate plus the extender refusals of the refused candidates, kept out of the result."""
        with run_match_cache():
            return cls._evaluate_once(
                feature,
                accessible_plugins,
                links,
                data_access_collection,
                excluded=excluded,
                partial_records=partial_records,
                _input_resolver=_input_resolver,
                _input_depth=_input_depth,
            )

    @classmethod
    def _evaluate_once(
        cls,
        feature: Feature,
        accessible_plugins: FeatureGroupEnvironmentMapping,
        links: set[Link] | None,
        data_access_collection: DataAccessCollection | None = None,
        *,
        excluded: Mapping[type[FeatureGroup], str] | None = None,
        partial_records: Sequence[ResolutionRecord] = (),
        _input_resolver: InputResolver | None = None,
        _input_depth: int = 0,
    ) -> tuple[EvaluationResult, tuple[Exception, ...]]:
        refusals: tuple[Exception, ...] = ()
        reject_reserved_reader_key(feature.options)
        # Pre-matching guard: a >1 pin fires regardless of whether any candidate matches (the old check
        # sat inside the filter loop, so it never ran when the name matched nothing).
        validate_single_framework_pin(feature)
        validate_source_pointers(feature, accessible_plugins, excluded, data_access_collection)
        self = cls(data_access_collection)
        self._partial_records = partial_records
        self._excluded = excluded or {}
        self._input_resolver = _input_resolver
        self._input_depth = _input_depth
        try:
            identified = self._filter_loop(feature, accessible_plugins, links, data_access_collection)
            # A single survivor means every dropped candidate is its ancestor (issubclass is transitive).
            specialized_from = tuple(sorted(self._replaced, key=_candidate_sort_key)) if len(identified) == 1 else ()
            if specialized_from:
                logger.debug(
                    "Feature %s: %s replaced %s",
                    feature.name,
                    ", ".join(c.__qualname__ for c in identified),
                    ", ".join(c.__qualname__ for c in specialized_from),
                )
            result = EvaluationResult(
                identified=identified,
                criteria_matched=self._criteria_matched_feature_groups,
                abstract_matched=self._abstract_matched_feature_groups,
                candidate_frameworks=self._candidate_frameworks,
                eliminations=self._eliminations,
                specialized_from=specialized_from,
                output_parts={fg: self._output_parts[fg] for fg in identified if fg in self._output_parts},
            )
            if result.failure_kind is not None:
                # Every elimination (value_rejection included) was already recorded during the single filter pass;
                # this only captures the non-elimination facts the messages still need.
                result = replace(result, facts=self._capture_render_facts(result, accessible_plugins, feature, links))
                if result.failure_kind != "multiple":
                    collected: list[Exception] = []
                    for fg in sorted(
                        self._match_rejections.keys() | self._input_refusals.keys(), key=_candidate_sort_key
                    ):
                        rejection = self._match_rejections.get(fg)
                        if fg not in self._matcher_errors and rejection is not None and rejection.refusal is not None:
                            collected.append(rejection.refusal)
                        collected.extend(self._input_refusals.get(fg, ()))
                    refusals = _unique_refusals(collected)
        finally:
            self._match_rejections.clear()
            # A captured exception pins its traceback, whose frames pin this instance: a refcount cycle that would
            # keep both alive until a gc pass. Dropping the outcomes and option forks (user values) makes each
            # memo's lifetime what it claims,
            # in a finally because a re-raising gate or an escalated match abort leaves without a return.
            self._domain_outcomes.clear()
            self._links_outcomes.clear()
            self._matched_options.clear()
            self._deferred_aborts.clear()
            self._input_data_matches.clear()
            self._open_claims.clear()
            self._input_refusals.clear()
        return result, refusals

    def _capture_render_facts(
        self,
        result: EvaluationResult,
        accessible_plugins: FeatureGroupEnvironmentMapping,
        feature: Feature,
        links: set[Link] | None,
    ) -> RenderFacts:
        """Capture the non-elimination facts the messages still need. Only reached when the pass has no winner.

        The renderer alone owns which message wins, so this does not mirror its branch order: the five cheap
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
            skipped_plugins=tuple(sorted(PluginLoader.skipped_plugins().items())),
            sources=self._capture_sources(result),
            scope_suggestions=self._capture_scope_suggestions(result, accessible_plugins, feature),
        )

    @staticmethod
    def _capture_scope_suggestions(
        result: EvaluationResult, accessible_plugins: FeatureGroupEnvironmentMapping, feature: Feature
    ) -> tuple[str, ...]:
        """Close names for a string scope that no accessible candidate has in its MRO; probes no candidate."""
        scope = feature.feature_group_scope
        if result.failure_kind != "none" or not isinstance(scope, str):
            return ()
        if any(matches_feature_group_scope(fg, scope) for fg in accessible_plugins):
            return ()
        names = {
            ancestor.__name__
            for fg in accessible_plugins
            for ancestor in fg.__mro__
            if ancestor is not FeatureGroup and issubclass(ancestor, FeatureGroup)
        }
        return tuple(get_close_matches(scope, sorted(names), n=3, cutoff=0.6))

    def _capture_sources(self, result: EvaluationResult) -> dict[type[FeatureGroup], str]:
        """Credential-free 'Reader: identity' of every identified candidate that matched a data source."""
        sources: dict[type[FeatureGroup], str] = {}
        for feature_group in result.identified:
            pair = self._input_data_matches.get(feature_group)
            if pair is None:
                continue
            reader, access = pair
            sources[feature_group] = safe_field(
                lambda: scrub_credentials(f"{reader.data_access_name()}: {reader.data_access_identity(access)}"),
                "",
                field=f"{getattr(reader, '__name__', 'reader')}.data_access_identity",
            )
        return {fg: text for fg, text in sources.items() if text}

    def _capture_eliminated_hints(self, result: EvaluationResult) -> frozenset[str]:
        """Class name and prefix of every eliminated near-miss, so the none message can suppress a
        'Did you mean' suggestion that merely echoes a candidate its near-miss block already names."""
        hints: set[str] = set()
        for feature_group in result.eliminations:
            # __name__, unlike the catalog: this suppresses what the near-miss block renders, which is __name__.
            hints.add(feature_group.__name__)
            prefix = self._prefix_of(feature_group)
            if prefix:
                hints.add(prefix)
        return frozenset(hints)

    def _domain_outcome(self, feature_group: type[FeatureGroup]) -> tuple[Domain | None, Exception | None]:
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
        field = f"{feature_group.__name__}.get_domain"
        domain, error = self._domain_outcome(feature_group)
        # error, not domain, is what tells a raise apart from a malformed return: both leave domain unusable.
        if error is not None:
            logger.warning("Degraded field '%s': %s", field, contained_raise_reason(error))
            return None
        if not isinstance(domain, Domain):
            # Annotated to return a Domain; a provider that returns something else costs only its own suffix.
            logger.warning("Degraded field '%s': expected Domain, got %s", field, type(domain).__name__)
            return None
        try:
            return domain.name
        except Exception as exc:  # noqa: BLE001  (plugin-owned property read; same contract as the two guards above)
            logger.warning("Degraded field '%s': %s", field, contained_raise_reason(exc))
            return None

    def _declared_frameworks_of(self, feature_group: type[FeatureGroup]) -> frozenset[type[ComputeFramework]]:
        """Memoized compute_framework_definition(), which drives compute_framework_rule(): once per candidate.

        Best-effort, and both readers guard it identically, so the value alone is enough to cache.
        """
        if feature_group not in self._declared_frameworks:
            self._declared_frameworks[feature_group] = safe_field(
                lambda: frozenset(feature_group.compute_framework_definition()),
                frozenset(),
                field=f"{feature_group.__name__}.compute_framework_definition",
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
            field=f"{feature_group.__name__}.compute_framework_definition",
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
        # get_class_name(), because that is the name the default matcher accepts, so a renaming override stays
        # reachable. Guarded, and falling back to __name__: a placeholder would seed a name no candidate carries.
        names = [
            safe_field(
                lambda: as_str(feature_group.get_class_name()),
                feature_group.__name__,
                field=f"{feature_group.__name__}.get_class_name",
            ),
            *self._supported_names_of(feature_group),
        ]
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
        self, feature_group: type[FeatureGroup], feature: Feature, links: set[Link] | None
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
            field=f"{feature_group.__name__}.index_columns/supports_index",
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
            # A raising Domain.name is also undecided: error is None yet domain is still a genuine Domain.
            return error is None and not isinstance(domain, Domain)
        # The gate's own comparison, never a name one: a Domain subclass with a custom __eq__ must not pass the
        # gate and fail this retest.
        return domain != feature.domain

    def _is_dead(
        self,
        result: EvaluationResult,
        feature_group: type[FeatureGroup],
        compute_frameworks: set[type[ComputeFramework]],
        feature: Feature,
        links: set[Link] | None,
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
        links: set[Link] | None,
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

    def _collect_candidates(
        self,
        feature: Feature,
        candidates: list[tuple[type[FeatureGroup], set[type[ComputeFramework]]]],
        links: set[Link] | None,
        data_access_collection: DataAccessCollection | None,
        identified: FeatureGroupEnvironmentMapping,
    ) -> None:
        request_key = resolution_key(feature)
        for feature_group, compute_frameworks in candidates:
            # An out-of-scope candidate is skipped silently: never probed, no elimination recorded.
            if not self._filter_feature_group_by_scope(feature_group, feature):
                continue
            requirement = self._declaration_requirement(feature_group, feature)
            # A criteria non-match records a value_rejection only when the first pass recorded a reason for it:
            # a plain name mismatch is not a near-miss, but a value the candidate declined (with a reportable
            # reason) is. The criteria call above just recorded any rejection under this candidate's window, so
            # this reads it back for a criteria-FAILING candidate only; a matched/winning/abstract candidate is
            # never probed. Recorded regardless of domain or of the overall outcome (a sibling may win).
            try:
                criteria_matched = self._filter_feature_group_by_criteria(
                    feature_group, feature, data_access_collection, request_key
                )
            except Exception as exc:  # noqa: BLE001  (only a marked abort of a gated-out candidate is absorbed)
                if not is_match_abort(exc):
                    raise
                if not self._filter_feature_group_by_domain(feature_group, feature):
                    self._record_elimination(feature_group, "domain", self._domain_reason(feature_group, feature))
                elif is_deferred_match_abort(exc):
                    self._deferred_aborts[feature_group] = exc
                    self._record_elimination(feature_group, "input_data", safe_exc_str(exc))
                else:
                    raise
                continue
            if not criteria_matched:
                # A contained matcher raise is always a near-miss: the raise says nothing about name ownership.
                # Deliberate precedence: a contained crash outranks a recorded decline for the same candidate.
                matcher_error = self._matcher_errors.get(feature_group)
                if matcher_error is not None:
                    self._record_elimination(feature_group, "matcher_error", matcher_error)
                    continue
                rejection = self._value_rejection(feature_group)
                if rejection is not None:
                    # The shared mapper owns the projection, so both seams spell the taxonomy the same way.
                    self._record_elimination(
                        feature_group, rejection_elimination_stage(rejection.stage), rejection.reason
                    )
                continue

            if not self._filter_feature_group_by_domain(feature_group, feature):
                self._record_elimination(feature_group, "domain", self._domain_reason(feature_group, feature))
                continue

            # Abstract bases can match name+domain but cannot be instantiated; never let one win, and
            # never record one as a near-miss: the abstract_only message owns them.
            if inspect.isabstract(feature_group):
                self._abstract_matched_feature_groups.add(feature_group)
                continue

            self._criteria_matched_feature_groups.add(feature_group)
            options = self._matched_options[feature_group]

            matched_part = self._output_parts.get(feature_group)
            requested_part = matched_part[1] if matched_part is not None else None
            supported_frameworks = {
                cfw
                for cfw in compute_frameworks
                if feature_group.supports_compute_framework(
                    self._hook_name(feature_group, feature), options, cfw, part=requested_part
                )
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
                    if get_all_subclasses(ComputeFramework):
                        not_enabled_reason = (
                            self._excluded.get(feature_group)
                            or _operation_implementation_reason(feature_group)
                            or "none of its compute frameworks are enabled for this run"
                        )
                    else:
                        not_enabled_reason = (
                            "no compute framework plugin is loaded; import one or call PluginLoader.all()"
                        )
                    self._record_elimination(feature_group, "frameworks_not_enabled", not_enabled_reason)
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

            if requirement is not None:
                reader = self._written_reader(feature_group)
                unmet = requirement.unmet_reason((reader or feature_group).__name__, reader)
                if unmet is not None:
                    self._record_elimination(feature_group, "declarations", unmet)
                    continue

            identified[feature_group] = supported_frameworks

    def _filter_loop(
        self,
        feature: Feature,
        accessible_plugins: FeatureGroupEnvironmentMapping,
        links: set[Link] | None,
        data_access_collection: DataAccessCollection | None = None,
    ) -> FeatureGroupEnvironmentMapping:
        self._accessible_plugins = accessible_plugins
        self._links = links
        self._environment = resolution_environment(accessible_plugins, links, data_access_collection, self._excluded)
        identified: FeatureGroupEnvironmentMapping = {}
        items = list(accessible_plugins.items())
        others = [item for item in items if not issubclass(item[0], SourceFeatureGroup)]
        sources = [item for item in items if issubclass(item[0], SourceFeatureGroup)]
        # Non-source candidates go first so a searched source route can step aside for a name one of them computes.
        self._collect_candidates(feature, others, links, data_access_collection, identified)
        if sources:
            computing = [fg for fg in identified if fg not in self._output_parts]
            narrowed = self.filter_subclasses({fg: identified[fg] for fg in computing})
            check = ChainCheck(str(feature.name)) if narrowed else None
            while True:
                try:
                    with chain_check_scope(check) if check is not None else nullcontext():
                        self._collect_candidates(feature, sources, links, data_access_collection, identified)
                    break
                except ChainCheckNeeded as needed:
                    if needed.check is not check or check is None:
                        raise
                    # Outside the scope, so the chain check stacks like an eager one and a cycle cannot see it unanswered.
                    check.answer = any(self._computes(fg, feature) for fg in sorted(narrowed, key=_candidate_sort_key))
        _identified_feature_groups = identified

        candidates = set(_identified_feature_groups)
        _identified_feature_groups = self.filter_subclasses(_identified_feature_groups)
        self._replaced = candidates - set(_identified_feature_groups)
        whole = [
            fg
            for fg in _identified_feature_groups
            if fg not in self._output_parts and fg.matches_whole_name(feature.name)
        ]
        if whole:
            for fg in [fg for fg in _identified_feature_groups if fg in self._output_parts]:
                del _identified_feature_groups[fg]
                self._record_elimination(
                    fg, "name", "matched only as an output part; another group matches the whole name"
                )
        if len(_identified_feature_groups) > 1:
            self._filter_input_chains(feature, _identified_feature_groups)
        self._drop_open_claims(feature, _identified_feature_groups)
        self._raise_unmasked_deferred_abort(feature, _identified_feature_groups)
        if len(_identified_feature_groups) == 1:
            winner = next(iter(_identified_feature_groups))
            winner_options = self._matched_options[winner]
            pair = self._input_data_matches.get(winner)
            # A winner that wrote no reader inherits the pair of its nearest replaced ancestor that recorded one.
            if pair is None:
                for klass in winner.__mro__:
                    if klass in self._replaced and klass in self._input_data_matches:
                        pair = self._input_data_matches[klass]
                        break
            feature.options._adopt(winner_options)
            feature.input_data_match = pair
        return _identified_feature_groups

    def _hook_name(self, feature_group: type[FeatureGroup], feature: Feature) -> str:
        """The name plugin hooks see: the base feature when the candidate matched a declared output part."""
        part = self._output_parts.get(feature_group)
        return part[0] if part else feature.name

    def _raise_unmasked_deferred_abort(self, feature: Feature, identified: FeatureGroupEnvironmentMapping) -> None:
        """Re-raise a pointed group's deferred abort unless another candidate computes the feature."""
        # A deferring group with a surviving strict subclass is subclass takeover: its abort is dropped.
        remaining = [
            fg
            for fg in sorted(self._deferred_aborts, key=_candidate_sort_key)
            if not any(survivor is not fg and issubclass(survivor, fg) for survivor in identified)
        ]
        if not remaining:
            return
        # A failed competing chain was already dropped and cannot unmask the abort.
        computing = any(
            self._declares_inputs(fg, feature)
            for fg in sorted(self._criteria_matched_feature_groups, key=_candidate_sort_key)
            if fg not in self._input_refusals and fg not in self._replaced
        )
        reading = [
            fg
            for fg in sorted(identified, key=_candidate_sort_key)
            if fg().is_root(self._matched_options[fg], self._hook_name(fg, feature))
        ]
        for fg in remaining:
            if not computing or any(not issubclass(survivor, fg) for survivor in reading):
                abort = self._deferred_aborts[fg]
                refusal = abort.__cause__
                raise escalate_match_abort(
                    FeatureResolutionError(
                        safe_exc_str(abort),
                        str(feature.name),
                        EvaluationResult(
                            identified={},
                            criteria_matched=self._criteria_matched_feature_groups,
                            abstract_matched=self._abstract_matched_feature_groups,
                            candidate_frameworks=self._candidate_frameworks,
                            eliminations=self._eliminations,
                        ),
                        partial_records=self._partial_records,
                        refusals=(refusal,) if isinstance(refusal, Exception) else (),
                    )
                )

    def _computing_groups(
        self,
        feature: Feature,
        identified: FeatureGroupEnvironmentMapping,
        excluding: frozenset[type[FeatureGroup]] = frozenset(),
    ) -> list[type[FeatureGroup]]:
        # A candidate eliminated after the chain filter is not probed; it only masks the claim by declaring inputs.
        return [
            fg
            for fg in sorted(
                self._criteria_matched_feature_groups - excluding - self._replaced, key=_candidate_sort_key
            )
            if (
                self._computes(fg, feature)
                if fg in identified or fg in self._input_refusals
                else self._declares_inputs(fg, feature)
            )
        ]

    def _drop_open_claims(self, feature: Feature, identified: FeatureGroupEnvironmentMapping) -> None:
        """Drop claimants that matched through an open route alone when another candidate computes the feature."""
        claimants = frozenset(fg for fg in self._open_claims if fg in identified)
        computing = self._computing_groups(feature, identified, claimants) if claimants else []
        if not computing:
            return
        names = ", ".join(fg.__name__ for fg in computing)
        for fg in sorted(claimants, key=_candidate_sort_key):
            del identified[fg]
            self._record_elimination(
                fg,
                "input_data",
                f"claimed '{feature.name}' only through an open route, but {names} computes it; "
                f"pass feature_group={fg.__name__} to read it from this source",
            )

    def _declares_inputs(self, feature_group: type[FeatureGroup], feature: Feature) -> bool:
        """A pointed source's missing output must still defer to the consumer that declares its inputs."""
        try:
            return (
                declared_inputs(
                    self._environment,
                    feature_group,
                    self._matched_options[feature_group]._fork(),
                    FeatureName(self._hook_name(feature_group, feature)),
                )
                is not None
            )
        except Exception as exc:  # noqa: BLE001  (an unmarked declaration failure does not compute the name)
            if is_match_abort(exc):
                raise
            return False

    def _computes(self, feature_group: type[FeatureGroup], feature: Feature) -> bool:
        """Only a non-root whose entire input chain resolves can make a source step aside."""
        outcome = self._input_chain(feature_group, feature)
        return outcome.viable and outcome.computing

    def _filter_input_chains(self, feature: Feature, identified: FeatureGroupEnvironmentMapping) -> None:
        for feature_group in sorted(tuple(identified), key=_candidate_sort_key):
            outcome = self._input_chain(feature_group, feature)
            if not outcome.viable:
                del identified[feature_group]
                self._record_elimination(feature_group, "inputs", outcome.reason)
                self._input_refusals[feature_group] = outcome.refusals

    @staticmethod
    def _input_failure(error: Exception) -> InputChainOutcome:
        if isinstance(error, FeatureResolutionError):
            reason = _first_line(str(error))
            if error.refusals:
                reason = f"{reason} ({contained_raise_reason(error.refusals[0])})"
            return InputChainOutcome(False, True, reason, error.refusals)
        return InputChainOutcome(False, True, contained_raise_reason(error), ())

    def _contain_own_raise(
        self, exc: Exception, feature_group: type[FeatureGroup], feature: Feature
    ) -> InputChainOutcome:
        """Contain a resolution failure or an unmarked raise of the candidate's own hooks; other aborts propagate."""
        if not isinstance(exc, FeatureResolutionError):
            if is_match_abort(exc):
                raise exc
            logger.log(
                contained_raise_log_level(exc),
                "%s %s while resolving its inputs for '%s'; treating it as a non-match.",
                safe_field(lambda: feature_group.get_class_name(), "<unnamed feature group>"),
                contained_raise_reason(exc),
                feature.name,
            )
        return self._input_failure(exc)

    def _input_chain(self, feature_group: type[FeatureGroup], feature: Feature) -> InputChainOutcome:
        """Resolve a candidate's input chain without adding graph nodes."""
        attempted = copy(feature)
        attempted.options = self._matched_options.get(feature_group, feature.options)._fork()
        output_part = self._output_parts.get(feature_group)
        feature_key = resolution_key(attempted)
        key = feature_group, feature_key, output_part
        state = input_chain_state(self._environment)
        if key in state.completed:
            return state.completed[key]
        if key in state.active:
            return InputChainOutcome(True, True, pending=frozenset({key}))
        state.active.add(key)
        state.active_features.setdefault(feature_key, set()).add(key)
        try:
            failure: InputChainOutcome | None = None
            entries: tuple[Feature | str, ...] | None = None
            try:
                normalize_input_name(self._environment, attempted, feature_group(), output_part)
                try:
                    entries = declared_inputs(self._environment, feature_group, attempted.options, attempted.name)
                except NotImplementedError as exc:
                    if is_match_abort(exc):
                        raise
            except Exception as exc:  # noqa: BLE001  (only the candidate's own naming/declaration raises are contained)
                failure = self._contain_own_raise(exc, feature_group, feature)
            if failure is not None:
                outcome = failure
            elif entries is None:
                outcome = InputChainOutcome(True, False)
            else:
                inputs = input_features_collection(feature_group, entries, attempted)
                links = None if self._links is None else set(self._links)
                for input_feature in inputs:
                    if input_feature.link is not None:
                        links = {input_feature.link} if links is None else links | {input_feature.link}
                if links is not None:
                    LinkValidator.validate_links(links)
                pending: set[Any] = set()
                outcome = InputChainOutcome(True, True)
                for input_feature in inputs:
                    input_feature.options.lock_own_keys()
                    input_feature.resolving_consumer = feature_group.get_class_name()
                    # Relative paths belong to the candidate, not the outer request that first filled the memo.
                    input_feature.resolving_path = (str(attempted.name),)
                    child = self._probe_input(input_feature, links)
                    if not child.viable:
                        outcome = InputChainOutcome(
                            False, True, f"input '{input_feature.name}': {child.reason}", child.refusals
                        )
                        break
                    pending.update(child.pending)
                if outcome.viable:
                    outcome = replace(outcome, pending=frozenset(pending))
        finally:
            state.active.remove(key)
            state.active_features[feature_key].remove(key)
            if not state.active_features[feature_key]:
                del state.active_features[feature_key]
        # A provisional back-edge success must not outlive a later failure elsewhere in that cycle.
        if not outcome.pending:
            state.completed[key] = outcome
        return outcome

    def _probe_input(self, feature: Feature, links: set[Link] | None) -> InputChainOutcome:
        environment = resolution_environment(
            self._accessible_plugins, links, self._data_access_collection, self._excluded
        )
        active = input_chain_state(environment).active_features.get(resolution_key(feature))
        if active:
            # Do not re-run name disambiguation for a back edge; the outer attempt owns that choice.
            return InputChainOutcome(True, True, pending=frozenset(active))
        try:
            if self._input_resolver is None:
                result, refusals = type(self)._evaluate(
                    feature,
                    self._accessible_plugins,
                    links,
                    self._data_access_collection,
                    excluded=self._excluded,
                    _input_depth=self._input_depth + 1,
                )
                message = render_resolution_failure(result, feature)
                if message is not None:
                    return InputChainOutcome(False, True, _first_line(message), refusals)
            else:
                result = self._input_resolver(feature, links, self._input_depth + 1)
            winner = next(iter(result.identified))
            child = type(self)(self._data_access_collection)
            child._accessible_plugins = self._accessible_plugins
            child._excluded = self._excluded
            child._links = links
            child._environment = environment
            child._input_resolver = self._input_resolver
            child._input_depth = self._input_depth + 1
            child._matched_options[winner] = feature.options
            child._output_parts = result.output_parts
            return child._input_chain(winner, feature)
        except FeatureResolutionError as exc:
            return self._input_failure(exc)

    def _declaration_requirement(
        self, feature_group: type[FeatureGroup], feature: Feature
    ) -> DeclarationRequirement | None:
        """The consumer's requirement with this candidate's own declarations; None when the request carries none."""
        required = feature.required_declarations
        if not required:
            return None
        consumer = feature.resolving_consumer or f"request for '{feature.name}'"
        return DeclarationRequirement(consumer, required, feature_group, self._declarations)

    def _written_reader(self, feature_group: type[FeatureGroup]) -> type | None:
        """The reader of the (ReaderClass, data_access) pair this candidate's criteria match wrote, if it wrote one."""
        matched = self._input_data_matches.get(feature_group)
        return cast("type | None", matched[0]) if matched else None

    def _record_elimination(self, feature_group: type[FeatureGroup], stage: EliminationStage, reason: str) -> None:
        """Record the first gate a non-winning name-matching candidate failed; one entry per candidate."""
        self._eliminations.setdefault(feature_group, Elimination(stage=stage, reason=reason))

    def _value_rejection(self, feature_group: type[FeatureGroup]) -> MatchRejection | None:
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

    def _filter_feature_group_by_links(self, feature_group: type[FeatureGroup], links: set[Link] | None) -> bool:
        """Decision-side links gate: unguarded, so a raising index hook still fails the engine loudly."""
        supported, error = self._links_outcome(feature_group, links)
        if error is not None:
            raise error
        # None is the memo's unreadable marker, never a verdict, so an outcome without an error always has one.
        assert supported is not None
        return supported

    def _links_outcome(
        self, feature_group: type[FeatureGroup], links: set[Link] | None
    ) -> tuple[bool | None, Exception | None]:
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
    def _links_gate(feature_group: type[FeatureGroup], links: set[Link] | None) -> bool:
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
        data_access_collection: DataAccessCollection | None,
        request_key: Any = None,
    ) -> bool:
        """A raise out of the match hook is a non-match for that candidate only, not a run-wide abort (#845).

        The shared probe owns the per-candidate window and the containment. This seam runs each probe on its own
        fork of the options, so a contained raise leaves nothing behind, and records the per-candidate outcome as
        text, never as an exception object whose traceback would pin the plugin class.

        Mark-or-contain policy: see call_match_hook.
        """

        name = str(feature.name)

        def compute() -> _CriteriaAttempt:
            options = feature.options._fork()
            uses = chain_check_uses(name)
            with (
                feature_group_scope(feature.feature_group_scope),
                open_claim_scope() as open_claim,
                pin_report_scope(self._reported_pins) as pending_pins,
            ):
                probe = probe_match_criteria(feature_group, feature.name, options, data_access_collection, True)
            consulted = computed_elsewhere(name) if chain_check_uses(name) > uses else None
            return _CriteriaAttempt(probe, options, bool(open_claim), tuple(pending_pins), consulted)

        if request_key is None:
            request_key = resolution_key(feature)
        key = (feature_group, request_key, chain_check_installed(name), frozenset(self._reported_pins))
        attempt = cached_input_fact(self._environment, "criteria", key, compute)
        if attempt.consulted is not None and attempt.consulted != computed_elsewhere(name):
            attempt = cached_input_fact(self._environment, "criteria", (*key, not attempt.consulted), compute)
        probe = attempt.probe
        pending_pins = attempt.pending_pins
        if not probe.matched and probe.matcher_error is None:
            self._reported_pins |= {
                path for path, record in pending_pins if record is not None and record is probe.rejection
            }
        if probe.matched and attempt.open_claim:
            self._open_claims |= {feature_group}
        # The pair carries credentials: it never stays in options, only in this evaluation's per-candidate map.
        written = probe.written
        if probe.matched:
            self._matched_options[feature_group] = attempt.options._fork()
            if probe.part is not None:
                self._output_parts[feature_group] = probe.part
            if written is not None:
                self._input_data_matches[feature_group] = written
        if probe.value_rejection is not None:
            exc = probe.value_rejection
            # Text, not exc: a retained record must not pin the traceback, its frames and the plugin class.
            logger.debug(
                "%s rejected an option value while matching '%s': %s",
                # A plugin-owned read past the hook call's containment, so it degrades instead of escaping the seam.
                safe_field(lambda: feature_group.get_class_name(), "<unnamed feature group>"),
                feature.name,
                scrub_credentials(safe_exc_str(exc)),
            )
        elif probe.matcher_error is not None:
            reason = contained_raise_reason(probe.matcher_error)
            logger.log(
                contained_raise_log_level(probe.matcher_error),
                "%s %s while matching '%s'; treating it as a non-match.",
                safe_field(lambda: feature_group.get_class_name(), "<unnamed feature group>"),
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
        """Decision-side domain gate: unguarded, so a raising get_domain() or Domain.name fails the engine loudly."""
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
        """Drop every candidate that another candidate subclasses, whatever their compute-framework sets."""
        fgs_to_pop: set[type[FeatureGroup]] = set()

        for i_feature_group in _identified_feature_groups:
            for o_feature_group in _identified_feature_groups:
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
    links: set[Link] | None = None,
    data_access_collection: DataAccessCollection | None = None,
    *,
    excluded: Mapping[type[FeatureGroup], str] | None = None,
) -> tuple[EvaluationResult, str | None]:
    """One resolution pass plus its failure message; the message is None iff the feature resolved."""
    # Unguarded: ComputeFrameworkPinError is a misuse validated before matching, so it escapes unconverted.
    result = IdentifyFeatureGroupClass.evaluate(
        feature, accessible_plugins, links, data_access_collection, excluded=excluded
    )
    return result, render_resolution_failure(result, feature)


def resolve_or_raise(
    feature: Feature,
    accessible_plugins: FeatureGroupEnvironmentMapping,
    links: set[Link] | None = None,
    data_access_collection: DataAccessCollection | None = None,
    partial_records: Sequence[ResolutionRecord] = (),
    *,
    excluded: Mapping[type[FeatureGroup], str] | None = None,
    _input_resolver: InputResolver | None = None,
    _input_depth: int = 0,
) -> EvaluationResult:
    """Evaluate one feature and raise the typed FeatureResolutionError on failure."""
    result, refusals = IdentifyFeatureGroupClass._evaluate(
        feature,
        accessible_plugins,
        links,
        data_access_collection,
        excluded=excluded,
        partial_records=partial_records,
        _input_resolver=_input_resolver,
        _input_depth=_input_depth,
    )
    message = render_resolution_failure(result, feature)
    if message is not None:
        # The constructor does the cap-then-deepcopy snapshot, so the records are forwarded as they are.
        raise FeatureResolutionError(
            message, str(feature.name), result, partial_records=partial_records, refusals=refusals
        )
    return result
