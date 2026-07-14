"""Filter-order and exactly-once hook tests for FeatureGroupResolver (issue #722 Stage 3a).

These tests define ``mloda.core.resolve.outcome`` and ``mloda.core.resolve.resolver``,
which do not exist yet: every test in this module fails on import until the Green
phase implements them. They pin the two deliberate order changes of the decided
contract (docs/docs/in_depth/feature-group-resolution-contract.md):

- Scope, domain, and abstract classification run BEFORE criteria, so irrelevant
  provider code never observes a structurally excluded request.
- The framework pin runs BEFORE capability hooks, so a broken hook on a
  pinned-away framework can never be engine-fatal.

Plus the exactly-once discipline: each provider hook fires at most once per
candidate, or per candidate/framework pair, on the success path.
"""

from typing import ClassVar

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.domain import Domain
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.resolve.identity import PluginIdentity
from mloda.core.resolve.outcome import (
    CandidateEvaluation,
    CandidateStatus,
    FrameworkEvaluation,
    FrameworkStatus,
    RejectionReason,
    ResolutionOutcome,
    ResolutionStatus,
)
from mloda.core.resolve.request import ResolutionRequestSnapshot
from mloda.core.resolve.resolver import FeatureGroupResolver, snapshot_from_mapping


SCOPE_ORDER_FEATURE = "probe722f_scope_order"
DOMAIN_ORDER_FEATURE = "probe722f_domain_order"
PIN_CAPABILITY_FEATURE = "probe722f_pin_capability"
HOOK_ONCE_FEATURE = "probe722f_hook_once"

DOMAIN_ORDER_MATCH = "probe722f_order_dom_a"
DOMAIN_ORDER_OTHER = "probe722f_order_dom_b"

PINNED_AWAY_ERROR_TEXT = "broken pinned-away capability 722f"


class CfwOrder722F(ComputeFramework):
    """Framework for the scope-order and domain-order probes."""


class CfwPinKeep722F(ComputeFramework):
    """The framework the pin-before-capability request pins."""


class CfwPinDrop722F(ComputeFramework):
    """The pinned-away framework whose capability hook would raise."""


class CfwHookA722F(ComputeFramework):
    """First framework of the hook-once family."""


class CfwHookB722F(ComputeFramework):
    """Second framework of the hook-once family."""


class _OrderProbeBase722F(FeatureGroup):
    """Shared probe base: never matches anything itself."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return False

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None


class ScopeOrderProbe722F(_OrderProbeBase722F):
    """Counting criteria probe: records every criteria call for its own feature name."""

    criteria_calls: ClassVar[list[str]] = []

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwOrder722F}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        # Gated on the probe name so process-global scans in other tests never count.
        if str(feature_name) == SCOPE_ORDER_FEATURE:
            cls.criteria_calls.append(str(feature_name))
            return True
        return False


class DomainOrderProbe722F(_OrderProbeBase722F):
    """Counting criteria probe living in its own domain."""

    criteria_calls: ClassVar[list[str]] = []

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwOrder722F}

    @classmethod
    def get_domain(cls) -> Domain:
        return Domain(DOMAIN_ORDER_MATCH)

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        # Gated on the probe name so process-global scans in other tests never count.
        if str(feature_name) == DOMAIN_ORDER_FEATURE:
            cls.criteria_calls.append(str(feature_name))
            return True
        return False


class PinCapabilityProbe722F(_OrderProbeBase722F):
    """Declares a pinned-kept and a pinned-away framework; the hook RAISES for the pinned-away one."""

    capability_calls: ClassVar[list[str]] = []

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwPinKeep722F, CfwPinDrop722F}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == PIN_CAPABILITY_FEATURE

    @classmethod
    def supports_compute_framework(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        compute_framework: type[ComputeFramework],
    ) -> bool:
        # Gated on the probe name so process-global scans in other tests never trip the raise.
        if str(feature_name) != PIN_CAPABILITY_FEATURE:
            return True
        cls.capability_calls.append(compute_framework.__name__)
        if compute_framework is CfwPinDrop722F:
            raise RuntimeError(PINNED_AWAY_ERROR_TEXT)
        return True


class HookOnceParentProbe722F(_OrderProbeBase722F):
    """Counting parent of the hook-once family; counters are shared and keyed by class name."""

    criteria_calls: ClassVar[dict[str, int]] = {}
    capability_calls: ClassVar[dict[str, list[str]]] = {}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwHookA722F, CfwHookB722F}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        # Gated on the probe name so process-global scans in other tests never count.
        if str(feature_name) == HOOK_ONCE_FEATURE:
            cls.criteria_calls[cls.__name__] = cls.criteria_calls.get(cls.__name__, 0) + 1
            return True
        return False

    @classmethod
    def supports_compute_framework(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        compute_framework: type[ComputeFramework],
    ) -> bool:
        if str(feature_name) == HOOK_ONCE_FEATURE:
            cls.capability_calls.setdefault(cls.__name__, []).append(compute_framework.__name__)
        return True


class HookOnceChildProbe722F(HookOnceParentProbe722F):
    """Counting child with the identical framework set; shadows the parent."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve(mapping: FeatureGroupEnvironmentMapping, feature: Feature) -> ResolutionOutcome:
    request = ResolutionRequestSnapshot.from_feature(feature)
    return FeatureGroupResolver().resolve(request, snapshot_from_mapping(mapping))


def _candidate(outcome: ResolutionOutcome, feature_group: type[FeatureGroup]) -> CandidateEvaluation:
    identity = PluginIdentity.from_class(feature_group)
    matches = [candidate for candidate in outcome.candidates if candidate.identity == identity]
    assert len(matches) == 1, f"expected exactly one evaluation for {identity.render()}, got {len(matches)}"
    return matches[0]


def _framework_evaluation(candidate: CandidateEvaluation, framework: type[ComputeFramework]) -> FrameworkEvaluation:
    matches = [evaluation for evaluation in candidate.frameworks if evaluation.framework is framework]
    assert len(matches) == 1, f"expected exactly one evaluation for {framework.__name__}, got {len(matches)}"
    return matches[0]


def _reasons(candidate: CandidateEvaluation) -> set[RejectionReason]:
    return {rejection.reason for rejection in candidate.rejections}


# ---------------------------------------------------------------------------
# Scope and domain run BEFORE criteria
# ---------------------------------------------------------------------------


def test_scope_rejection_precedes_criteria_hook() -> None:
    """A scoped-out candidate is rejected structurally; its criteria hook never fires."""
    mapping: FeatureGroupEnvironmentMapping = {ScopeOrderProbe722F: {CfwOrder722F}}

    # Premise guard: unscoped, the counting hook fires exactly once and the probe resolves.
    ScopeOrderProbe722F.criteria_calls.clear()
    unscoped = _resolve(mapping, Feature(SCOPE_ORDER_FEATURE))
    assert unscoped.status is ResolutionStatus.RESOLVED
    assert len(ScopeOrderProbe722F.criteria_calls) == 1

    ScopeOrderProbe722F.criteria_calls.clear()
    scoped_out = _resolve(mapping, Feature(SCOPE_ORDER_FEATURE, feature_group="UnrelatedScope722F"))

    assert scoped_out.status is ResolutionStatus.NOT_FOUND
    assert scoped_out.winner is None
    candidate = _candidate(scoped_out, ScopeOrderProbe722F)
    assert candidate.status is CandidateStatus.REJECTED
    assert RejectionReason.SCOPE in _reasons(candidate)
    assert ScopeOrderProbe722F.criteria_calls == [], "scope runs before criteria: the hook must never fire"


def test_domain_rejection_precedes_criteria_hook() -> None:
    """A domain-mismatched candidate is rejected structurally; its criteria hook never fires."""
    mapping: FeatureGroupEnvironmentMapping = {DomainOrderProbe722F: {CfwOrder722F}}

    # Premise guard: with the matching domain, the counting hook fires exactly once.
    DomainOrderProbe722F.criteria_calls.clear()
    matched = _resolve(mapping, Feature(DOMAIN_ORDER_FEATURE, domain=DOMAIN_ORDER_MATCH))
    assert matched.status is ResolutionStatus.RESOLVED
    assert len(DomainOrderProbe722F.criteria_calls) == 1

    DomainOrderProbe722F.criteria_calls.clear()
    mismatched = _resolve(mapping, Feature(DOMAIN_ORDER_FEATURE, domain=DOMAIN_ORDER_OTHER))

    assert mismatched.status is ResolutionStatus.NOT_FOUND
    assert mismatched.winner is None
    candidate = _candidate(mismatched, DomainOrderProbe722F)
    assert candidate.status is CandidateStatus.REJECTED
    assert RejectionReason.DOMAIN in _reasons(candidate)
    assert DomainOrderProbe722F.criteria_calls == [], "domain runs before criteria: the hook must never fire"


# ---------------------------------------------------------------------------
# The framework pin runs BEFORE capability hooks
# ---------------------------------------------------------------------------


def test_pin_excluded_framework_never_reaches_capability_hook() -> None:
    """A raising hook on a pinned-away framework is never invoked and never fatal."""
    PinCapabilityProbe722F.capability_calls.clear()
    mapping: FeatureGroupEnvironmentMapping = {PinCapabilityProbe722F: {CfwPinKeep722F, CfwPinDrop722F}}
    feature = Feature(PIN_CAPABILITY_FEATURE, compute_framework=CfwPinKeep722F.get_class_name())
    outcome = _resolve(mapping, feature)

    assert outcome.status is ResolutionStatus.RESOLVED
    assert outcome.winner is not None
    assert outcome.winner.feature_group is PinCapabilityProbe722F
    assert outcome.winner.compute_frameworks == (CfwPinKeep722F,)
    assert outcome.failures == ()
    assert PinCapabilityProbe722F.capability_calls == [CfwPinKeep722F.__name__], (
        "the pin runs before capability hooks: the pinned-away framework must never reach the hook"
    )

    candidate = _candidate(outcome, PinCapabilityProbe722F)
    assert candidate.status is CandidateStatus.WINNER
    assert _framework_evaluation(candidate, CfwPinKeep722F).status is FrameworkStatus.SUPPORTED
    dropped = _framework_evaluation(candidate, CfwPinDrop722F)
    assert dropped.status is FrameworkStatus.PIN_EXCLUDED
    assert dropped.failure is None


# ---------------------------------------------------------------------------
# Exactly-once hook discipline on the success path
# ---------------------------------------------------------------------------


def test_hooks_run_exactly_once_per_candidate_and_framework() -> None:
    """Criteria fires once per candidate; capability once per candidate/framework pair."""
    HookOnceParentProbe722F.criteria_calls.clear()
    HookOnceParentProbe722F.capability_calls.clear()
    mapping: FeatureGroupEnvironmentMapping = {
        HookOnceParentProbe722F: {CfwHookA722F, CfwHookB722F},
        HookOnceChildProbe722F: {CfwHookA722F, CfwHookB722F},
    }
    outcome = _resolve(mapping, Feature(HOOK_ONCE_FEATURE))

    assert outcome.status is ResolutionStatus.RESOLVED
    assert outcome.winner is not None
    assert outcome.winner.feature_group is HookOnceChildProbe722F

    assert HookOnceParentProbe722F.criteria_calls == {
        "HookOnceParentProbe722F": 1,
        "HookOnceChildProbe722F": 1,
    }

    expected_frameworks = sorted([CfwHookA722F.__name__, CfwHookB722F.__name__])
    capability_calls = HookOnceParentProbe722F.capability_calls
    assert set(capability_calls) == {"HookOnceParentProbe722F", "HookOnceChildProbe722F"}
    assert sorted(capability_calls["HookOnceParentProbe722F"]) == expected_frameworks
    assert sorted(capability_calls["HookOnceChildProbe722F"]) == expected_frameworks
