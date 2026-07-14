from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import PropertyValueRejection
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.link import Link
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.resolve.environment import (
    EnvironmentProvenance,
    FeatureGroupRecord,
    FrameworkRecord,
    ResolutionEnvironmentSnapshot,
    _fingerprint,
)
from mloda.core.resolve.identity import PluginIdentity
from mloda.core.resolve.outcome import (
    CandidateEvaluation,
    CandidateStatus,
    FrameworkEvaluation,
    FrameworkStatus,
    PluginFailure,
    Rejection,
    RejectionReason,
    ResolutionOutcome,
    ResolutionStatus,
    ResolvedCandidate,
)
from mloda.core.resolve.request import ResolutionRequestSnapshot


logger = logging.getLogger(__name__)


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


def snapshot_from_mapping(
    mapping: FeatureGroupEnvironmentMapping, strict_mode: str = "off"
) -> ResolutionEnvironmentSnapshot:
    """Accessible-only snapshot over an engine mapping.

    Enabled and requested frameworks are the union of the mapping's framework sets; every
    record carries ACCESSIBLE provenance because the engine mapping only holds survivors.
    ``strict_mode`` is caller-provided metadata; the mapping already encodes the survivors
    of any strict filtering, so the snapshot never re-reads run configuration from the env.
    """
    accessible_entries: list[tuple[type[FeatureGroup], tuple[type[ComputeFramework], ...]]] = []
    records: list[FeatureGroupRecord] = []
    framework_union: set[type[ComputeFramework]] = set()
    for feature_group in sorted(mapping, key=PluginIdentity.from_class):
        frameworks = tuple(sorted(mapping[feature_group], key=PluginIdentity.from_class))
        framework_union.update(frameworks)
        accessible_entries.append((feature_group, frameworks))
        records.append(
            FeatureGroupRecord(
                identity=PluginIdentity.from_class(feature_group),
                provenance=EnvironmentProvenance.ACCESSIBLE,
                frameworks=tuple(
                    FrameworkRecord(
                        identity=PluginIdentity.from_class(framework),
                        provenance=EnvironmentProvenance.ACCESSIBLE,
                    )
                    for framework in frameworks
                ),
            )
        )
    enabled = tuple(sorted(PluginIdentity.from_class(framework) for framework in framework_union))
    record_tuple = tuple(records)
    return ResolutionEnvironmentSnapshot(
        records=record_tuple,
        accessible=tuple(accessible_entries),
        strict_mode=strict_mode,
        requested_frameworks=enabled,
        enabled_frameworks=enabled,
        fingerprint=_fingerprint(strict_mode, enabled, enabled, record_tuple),
    )


@dataclass
class _CandidateState:
    """Mutable working state for one candidate; frozen into a CandidateEvaluation at the end."""

    feature_group: type[FeatureGroup]
    identity: PluginIdentity
    status: CandidateStatus
    frameworks: tuple[FrameworkEvaluation, ...] = ()
    rejections: list[Rejection] = field(default_factory=list)
    shadowed_by: PluginIdentity | None = None
    failure: PluginFailure | None = None
    failures: tuple[PluginFailure, ...] = ()

    def supported_frameworks(self) -> tuple[type[ComputeFramework], ...]:
        return tuple(
            evaluation.framework for evaluation in self.frameworks if evaluation.status is FrameworkStatus.SUPPORTED
        )

    def freeze(self) -> CandidateEvaluation:
        return CandidateEvaluation(
            feature_group=self.feature_group,
            identity=self.identity,
            status=self.status,
            frameworks=self.frameworks,
            rejections=tuple(self.rejections),
            shadowed_by=self.shadowed_by,
            failure=self.failure,
        )


class FeatureGroupResolver:
    """Authoritative per-feature resolution: one request against one environment snapshot.

    resolve() never raises for resolution failures; provider exceptions become fail-closed
    FAILED outcomes. Filter order follows the decided contract in
    docs/docs/in_depth/feature-group-resolution-contract.md.
    """

    def resolve(
        self, request: ResolutionRequestSnapshot, environment: ResolutionEnvironmentSnapshot
    ) -> ResolutionOutcome:
        request_failure = self._validate_request(request)
        if request_failure is not None:
            return ResolutionOutcome(
                status=ResolutionStatus.FAILED,
                winner=None,
                candidates=(),
                failures=(request_failure,),
                environment_fingerprint=environment.fingerprint,
            )

        feature_name = FeatureName(request.feature_name)
        states = [
            self._evaluate_candidate(feature_group, frameworks, request, feature_name)
            for feature_group, frameworks in sorted(
                environment.accessible, key=lambda entry: PluginIdentity.from_class(entry[0])
            )
        ]

        self._apply_shadowing(states)

        failures = tuple(failure for state in states for failure in state.failures)
        survivors = [state for state in states if state.status is CandidateStatus.SURVIVOR]

        winner: ResolvedCandidate | None = None
        if failures:
            status = ResolutionStatus.FAILED
        elif len(survivors) == 1:
            status = ResolutionStatus.RESOLVED
            winning = survivors[0]
            winning.status = CandidateStatus.WINNER
            winner = ResolvedCandidate(
                feature_group=winning.feature_group,
                identity=winning.identity,
                compute_frameworks=winning.supported_frameworks(),
            )
        elif survivors:
            status = ResolutionStatus.AMBIGUOUS
        else:
            status = ResolutionStatus.NOT_FOUND

        return ResolutionOutcome(
            status=status,
            winner=winner,
            candidates=tuple(state.freeze() for state in states),
            failures=failures,
            environment_fingerprint=environment.fingerprint,
        )

    def _validate_request(self, request: ResolutionRequestSnapshot) -> PluginFailure | None:
        if len(request.pinned_frameworks) > 1:
            return PluginFailure(
                plugin=PluginIdentity.from_class(request.pinned_frameworks[0]),
                stage="validate_request",
                category="ValueError",
                message=f"Feature should only have one compute framework when set by user {request.feature_name}.",
            )
        return None

    def _evaluate_candidate(
        self,
        feature_group: type[FeatureGroup],
        frameworks: tuple[type[ComputeFramework], ...],
        request: ResolutionRequestSnapshot,
        feature_name: FeatureName,
    ) -> _CandidateState:
        state = _CandidateState(
            feature_group=feature_group,
            identity=PluginIdentity.from_class(feature_group),
            status=CandidateStatus.REJECTED,
        )

        scope = request.feature_group_scope
        if scope is not None and not matches_feature_group_scope(feature_group, scope):
            state.rejections.append(Rejection(RejectionReason.SCOPE))
            return state

        # Abstract classification: recorded here, rejected after criteria so the ABSTRACT
        # reason keeps meaning "a genuine match that can never be instantiated".
        is_abstract = inspect.isabstract(feature_group)

        if request.domain is not None:
            try:
                domain_name = feature_group.get_domain().name
            except Exception as exc:  # Fail closed: a raising domain hook is decision-relevant.
                return self._fail_state(state, "get_domain", exc)
            if domain_name != request.domain:
                state.rejections.append(Rejection(RejectionReason.DOMAIN))
                return state

        try:
            matched = feature_group.match_feature_group_criteria(
                feature_name, request.options, request.data_access_collection
            )
        except PropertyValueRejection as exc:
            logger.debug("%s rejected an option value while matching '%s': %s", feature_group, feature_name, exc)
            state.rejections.append(Rejection(RejectionReason.VALUE_REJECTION, detail=str(exc)))
            return state
        except Exception as exc:  # Fail closed: any provider exception is decision-relevant.
            return self._fail_state(state, "match_feature_group_criteria", exc)
        if not matched:
            state.rejections.append(Rejection(RejectionReason.CRITERIA))
            return state

        if is_abstract:
            state.rejections.append(Rejection(RejectionReason.ABSTRACT))
            return state

        if not frameworks:
            state.rejections.append(Rejection(RejectionReason.NO_ACCESSIBLE_FRAMEWORK))
            return state

        remaining = list(frameworks)
        evaluations: list[FrameworkEvaluation] = []
        if request.pinned_frameworks:
            pinned = request.pinned_frameworks
            for framework in frameworks:
                if framework not in pinned:
                    evaluations.append(
                        FrameworkEvaluation(
                            framework=framework,
                            identity=PluginIdentity.from_class(framework),
                            status=FrameworkStatus.PIN_EXCLUDED,
                        )
                    )
            remaining = [framework for framework in frameworks if framework in pinned]
            if not remaining:
                state.frameworks = tuple(evaluations)
                state.rejections.append(Rejection(RejectionReason.FRAMEWORK_PIN))
                return state

        links_supported, link_failure = self._supports_request_links(feature_group, request.links, state.identity)
        if link_failure is not None:
            state.frameworks = tuple(evaluations)
            return self._record_failure(state, link_failure)
        if not links_supported:
            # Capability is never consulted for a link-rejected candidate; the surviving
            # frameworks carry NOT_EVALUATED because they never reached that evaluation.
            for framework in remaining:
                evaluations.append(
                    FrameworkEvaluation(
                        framework=framework,
                        identity=PluginIdentity.from_class(framework),
                        status=FrameworkStatus.NOT_EVALUATED,
                    )
                )
            state.frameworks = tuple(evaluations)
            state.rejections.append(Rejection(RejectionReason.LINK_INDEX))
            return state

        failures: list[PluginFailure] = []
        supported = False
        for framework in remaining:
            framework_identity = PluginIdentity.from_class(framework)
            try:
                accepts = feature_group.supports_compute_framework(feature_name, request.options, framework)
            except Exception as exc:  # Fail closed: a raising capability hook is decision-relevant.
                failure = PluginFailure(
                    plugin=state.identity,
                    stage="supports_compute_framework",
                    category=type(exc).__name__,
                    message=str(exc),
                )
                failures.append(failure)
                evaluations.append(
                    FrameworkEvaluation(
                        framework=framework,
                        identity=framework_identity,
                        status=FrameworkStatus.HOOK_FAILED,
                        failure=failure,
                    )
                )
                continue
            if accepts:
                supported = True
                evaluations.append(
                    FrameworkEvaluation(
                        framework=framework, identity=framework_identity, status=FrameworkStatus.SUPPORTED
                    )
                )
            else:
                evaluations.append(
                    FrameworkEvaluation(
                        framework=framework, identity=framework_identity, status=FrameworkStatus.CAPABILITY_REJECTED
                    )
                )

        state.frameworks = tuple(evaluations)
        if failures:
            state.status = CandidateStatus.FAILED
            state.failure = failures[0]
            state.failures = tuple(failures)
            return state
        if not supported:
            state.rejections.append(Rejection(RejectionReason.CAPABILITY))
            return state

        state.status = CandidateStatus.SURVIVOR
        return state

    def _fail_state(self, state: _CandidateState, stage: str, exc: Exception) -> _CandidateState:
        """Freeze a raising provider hook into the candidate's fail-closed FAILED state."""
        failure = PluginFailure(
            plugin=state.identity,
            stage=stage,
            category=type(exc).__name__,
            message=str(exc),
        )
        return self._record_failure(state, failure)

    def _record_failure(self, state: _CandidateState, failure: PluginFailure) -> _CandidateState:
        state.status = CandidateStatus.FAILED
        state.failure = failure
        state.failures = (failure,)
        return state

    def _supports_request_links(
        self, feature_group: type[FeatureGroup], links: tuple[Link, ...], identity: PluginIdentity
    ) -> tuple[bool, PluginFailure | None]:
        """Link/index compatibility verdict, plus a failure when an index hook raised.

        index_columns() runs BEFORE the empty-links short-circuit on purpose: a raising
        index declaration fails closed even for a link-less request.
        """
        try:
            index_columns = feature_group.index_columns()
        except Exception as exc:  # Fail closed: a raising index declaration is decision-relevant.
            return False, PluginFailure(
                plugin=identity, stage="index_columns", category=type(exc).__name__, message=str(exc)
            )
        if index_columns is None:
            return True, None
        if not links:
            return True, None
        for link in links:
            for index in (link.left_index, link.right_index):
                try:
                    supported = feature_group.supports_index(index)
                except Exception as exc:  # Fail closed: a raising index capability hook is decision-relevant.
                    return False, PluginFailure(
                        plugin=identity, stage="supports_index", category=type(exc).__name__, message=str(exc)
                    )
                if supported:
                    return True, None
        return False, None

    def _apply_shadowing(self, states: list[_CandidateState]) -> None:
        """Framework-aware subclass preference: a child shadows its parent only when both
        survived every filter and their supported framework sets are equal."""
        survivors = [state for state in states if state.status is CandidateStatus.SURVIVOR]
        for parent in survivors:
            parent_supported = set(parent.supported_frameworks())
            for child in survivors:
                if child is parent:
                    continue
                if not issubclass(child.feature_group, parent.feature_group):
                    continue
                if set(child.supported_frameworks()) != parent_supported:
                    continue
                parent.status = CandidateStatus.SHADOWED
                parent.shadowed_by = child.identity
                parent.rejections.append(Rejection(RejectionReason.SUBCLASS_SHADOWED))
                break
