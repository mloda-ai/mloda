"""Target-contract tests for the authoritative FeatureGroupResolver (issue #722 Stage 3a).

These tests define ``mloda.core.resolve.outcome`` and ``mloda.core.resolve.resolver``,
which do not exist yet: every test in this module fails on import until the Green
phase implements them. The contract under test is decided in
docs/docs/in_depth/feature-group-resolution-contract.md:

- resolve() never raises for resolution failures; it returns a structured
  ResolutionOutcome. Provider exceptions become fail-closed FAILED outcomes.
- A winner is present if and only if the status is RESOLVED.
- Candidate ordering is deterministic by stable plugin identity.
- Public records serialize as plain data: no class objects, no Options values,
  no exception objects, no tracebacks.
- The engine adapter (IdentifyFeatureGroupClass) raises FeatureResolutionError,
  a ValueError subclass carrying the outcome.

Ordering and exactly-once hook discipline are covered by test_resolver_order.py.
"""

import inspect
import json
from abc import abstractmethod

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.domain import Domain
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import PropertyValueRejection
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.index.index import Index
from mloda.core.abstract_plugins.components.link import JoinSpec, Link
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.core.resolve.identity import PluginIdentity
from mloda.core.resolve.outcome import (
    CandidateEvaluation,
    CandidateStatus,
    FeatureResolutionError,
    FrameworkEvaluation,
    FrameworkStatus,
    RejectionReason,
    ResolutionOutcome,
    ResolutionStatus,
)
from mloda.core.resolve.request import ResolutionRequestSnapshot
from mloda.core.resolve.resolver import FeatureGroupResolver, snapshot_from_mapping


CLEAN_FEATURE = "probe722f_clean"
AMBIGUOUS_FEATURE = "probe722f_ambiguous"
SHADOW_FEATURE = "probe722f_shadow"
SPLIT_FEATURE = "probe722f_split"
ABSTRACT_FEATURE = "probe722f_abstract_only"
NO_FRAMEWORK_FEATURE = "probe722f_no_framework"
DOMAIN_FEATURE = "probe722f_domain"
PIN_FEATURE = "probe722f_pin"
LINKED_FEATURE = "probe722f_linked"
VALUE_REJECTION_FEATURE = "probe722f_value_rejection"
BOOM_FEATURE = "probe722f_boom"
CAPABILITY_RAISES_FEATURE = "probe722f_capability_raises"
CAPABILITY_FALSE_FEATURE = "probe722f_capability_false"
MULTI_PIN_FEATURE = "probe722f_multi_pin"
UNKNOWN_FEATURE = "probe722f_totally_unknown"

DOMAIN_MATCH = "probe722f_dom_sales"
DOMAIN_OTHER = "probe722f_dom_marketing"

VALUE_REJECTION_TEXT = "rejected value 722f: probe option out of range"
BOOM_ERROR_TEXT = "boom 722f"
CAPABILITY_ERROR_TEXT = "broken capability 722f"
MULTI_PIN_TEXT = "should only have one compute framework"


class CfwAlpha722F(ComputeFramework):
    """General-purpose framework for single-framework probe mappings."""


class CfwShadow722F(ComputeFramework):
    """Shared framework of the equal-set shadowing family."""


class CfwP722F(ComputeFramework):
    """First framework of the differing-set family; declared by parent and child."""


class CfwQ722F(ComputeFramework):
    """Second framework of the differing-set family; declared by the parent only."""


class CfwPinX722F(ComputeFramework):
    """Uniquely named framework, pinnable by name from a Feature."""


class CfwPinY722F(ComputeFramework):
    """Rival uniquely named framework for the pin sibling probe."""


class CfwCap722F(ComputeFramework):
    """Framework declared by the capability probes."""


class CfwMultiOne722F(ComputeFramework):
    """First framework of the illegal multi-pin request."""


class CfwMultiTwo722F(ComputeFramework):
    """Second framework of the illegal multi-pin request."""


class _ResolverProbeBase722F(FeatureGroup):
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


class CleanProbe722F(_ResolverProbeBase722F):
    """Single clean candidate: matches its own name on one framework."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwAlpha722F}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == CLEAN_FEATURE


class AmbiguousOneProbe722F(_ResolverProbeBase722F):
    """First of two unrelated probes matching the ambiguous feature name."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwAlpha722F}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == AMBIGUOUS_FEATURE


class AmbiguousTwoProbe722F(_ResolverProbeBase722F):
    """Second of two unrelated probes matching the ambiguous feature name."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwAlpha722F}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == AMBIGUOUS_FEATURE


class ShadowParentProbe722F(_ResolverProbeBase722F):
    """Parent of the equal-framework-set family; shadowed by its child."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwShadow722F}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == SHADOW_FEATURE


class ShadowChildProbe722F(ShadowParentProbe722F):
    """Child with the identical framework set; shadows the parent."""


class SplitParentProbe722F(_ResolverProbeBase722F):
    """Parent declaring two frameworks; the child narrows to one."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwP722F, CfwQ722F}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == SPLIT_FEATURE


class SplitChildProbe722F(SplitParentProbe722F):
    """Child narrowing the declaration to a single framework."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwP722F}


class AbstractProbe722F(_ResolverProbeBase722F):
    """Abstract probe: matches its own name but can never be instantiated."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwAlpha722F}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == ABSTRACT_FEATURE

    @classmethod
    @abstractmethod
    def _probe_hook_722f(cls) -> str:
        """Abstract hook that keeps this probe abstract."""


class NoFrameworkProbe722F(_ResolverProbeBase722F):
    """Matches its name, but the environment mapping strips all its frameworks."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwAlpha722F}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == NO_FRAMEWORK_FEATURE


class DomainProbe722F(_ResolverProbeBase722F):
    """Matches its name and lives in the matching probe domain."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwAlpha722F}

    @classmethod
    def get_domain(cls) -> Domain:
        return Domain(DOMAIN_MATCH)

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == DOMAIN_FEATURE


class PinSiblingX722F(_ResolverProbeBase722F):
    """Sibling probe on CfwPinX722F, matching the shared pin feature name."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwPinX722F}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == PIN_FEATURE


class PinSiblingY722F(_ResolverProbeBase722F):
    """Sibling probe on CfwPinY722F, matching the same pin feature name."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwPinY722F}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == PIN_FEATURE


class LinkLeftProbe722F(_ResolverProbeBase722F):
    """Anchors the left side of the foreign link; never matches a name."""


class LinkRightProbe722F(_ResolverProbeBase722F):
    """Anchors the right side of the foreign link; never matches a name."""


class LinkedProbe722F(_ResolverProbeBase722F):
    """Declares an index no link in the request carries."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwAlpha722F}

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index(("probe722f_row_id",))]

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == LINKED_FEATURE


class ValueRejectionProbe722F(_ResolverProbeBase722F):
    """Raises PropertyValueRejection from matching, gated on its shared feature name."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwAlpha722F}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        # Gated on the probe name so process-global scans in other tests never trip the raise.
        if str(feature_name) == VALUE_REJECTION_FEATURE:
            raise PropertyValueRejection(VALUE_REJECTION_TEXT)
        return False


class ValueRejectionWinnerProbe722F(_ResolverProbeBase722F):
    """Matches the value-rejection feature name cleanly."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwAlpha722F}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == VALUE_REJECTION_FEATURE


class BoomProbe722F(_ResolverProbeBase722F):
    """Raises a plain ValueError from matching, gated on its shared feature name."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwAlpha722F}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        # Gated on the probe name so process-global scans in other tests never trip the raise.
        if str(feature_name) == BOOM_FEATURE:
            raise ValueError(BOOM_ERROR_TEXT)
        return False


class BoomWinnerProbe722F(_ResolverProbeBase722F):
    """Matches the boom feature name cleanly."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwAlpha722F}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == BOOM_FEATURE


class CapabilityRaisesProbe722F(_ResolverProbeBase722F):
    """Probe whose capability hook raises on its only (remaining) framework."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwCap722F}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == CAPABILITY_RAISES_FEATURE

    @classmethod
    def supports_compute_framework(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        compute_framework: type[ComputeFramework],
    ) -> bool:
        # Gated on the probe name so process-global scans in other tests never trip the raise.
        if str(feature_name) == CAPABILITY_RAISES_FEATURE:
            raise RuntimeError(CAPABILITY_ERROR_TEXT)
        return True


class CapabilityFalseProbe722F(_ResolverProbeBase722F):
    """Probe whose capability hook rejects every framework by returning False."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwCap722F}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == CAPABILITY_FALSE_FEATURE

    @classmethod
    def supports_compute_framework(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        compute_framework: type[ComputeFramework],
    ) -> bool:
        if str(feature_name) == CAPABILITY_FALSE_FEATURE:
            return False
        return True


class MultiPinProbe722F(_ResolverProbeBase722F):
    """Matches the multi-pin feature name; the request itself is the illegal part."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwMultiOne722F, CfwMultiTwo722F}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == MULTI_PIN_FEATURE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve(
    mapping: FeatureGroupEnvironmentMapping,
    feature: Feature,
    links: set[Link] | None = None,
) -> ResolutionOutcome:
    request = ResolutionRequestSnapshot.from_feature(feature, links=links)
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
# Statuses: RESOLVED, AMBIGUOUS, NOT_FOUND
# ---------------------------------------------------------------------------


def test_single_clean_candidate_resolves() -> None:
    """A single clean candidate resolves with WINNER and SUPPORTED evaluations."""
    mapping: FeatureGroupEnvironmentMapping = {CleanProbe722F: {CfwAlpha722F}}
    outcome = _resolve(mapping, Feature(CLEAN_FEATURE))

    assert outcome.status is ResolutionStatus.RESOLVED
    assert outcome.winner is not None
    assert outcome.winner.feature_group is CleanProbe722F
    assert outcome.winner.compute_frameworks == (CfwAlpha722F,)
    assert outcome.failures == ()

    candidate = _candidate(outcome, CleanProbe722F)
    assert candidate.status is CandidateStatus.WINNER
    assert candidate.rejections == ()
    evaluation = _framework_evaluation(candidate, CfwAlpha722F)
    assert evaluation.status is FrameworkStatus.SUPPORTED
    assert evaluation.identity == PluginIdentity.from_class(CfwAlpha722F)


def test_two_unrelated_candidates_same_name_are_ambiguous() -> None:
    """Two unrelated candidates on the same name stay AMBIGUOUS; both are SURVIVOR."""
    mapping: FeatureGroupEnvironmentMapping = {
        AmbiguousOneProbe722F: {CfwAlpha722F},
        AmbiguousTwoProbe722F: {CfwAlpha722F},
    }
    outcome = _resolve(mapping, Feature(AMBIGUOUS_FEATURE))

    assert outcome.status is ResolutionStatus.AMBIGUOUS
    assert outcome.winner is None
    assert _candidate(outcome, AmbiguousOneProbe722F).status is CandidateStatus.SURVIVOR
    assert _candidate(outcome, AmbiguousTwoProbe722F).status is CandidateStatus.SURVIVOR


def test_child_shadows_parent_with_equal_framework_sets() -> None:
    """A child with the identical framework set wins; the parent is SHADOWED, visibly."""
    mapping: FeatureGroupEnvironmentMapping = {
        ShadowParentProbe722F: {CfwShadow722F},
        ShadowChildProbe722F: {CfwShadow722F},
    }
    outcome = _resolve(mapping, Feature(SHADOW_FEATURE))

    assert outcome.status is ResolutionStatus.RESOLVED
    assert outcome.winner is not None
    assert outcome.winner.feature_group is ShadowChildProbe722F

    child = _candidate(outcome, ShadowChildProbe722F)
    assert child.status is CandidateStatus.WINNER

    parent = _candidate(outcome, ShadowParentProbe722F)
    assert parent.status is CandidateStatus.SHADOWED
    assert parent.shadowed_by == PluginIdentity.from_class(ShadowChildProbe722F)
    assert RejectionReason.SUBCLASS_SHADOWED in _reasons(parent)


def test_parent_child_differing_framework_sets_stay_ambiguous_with_per_candidate_attribution() -> None:
    """Differing framework sets block shadowing; each candidate lists only ITS frameworks.

    The false-positive-3 core: the parent-only framework must never be attributed
    to the child candidate.
    """
    mapping: FeatureGroupEnvironmentMapping = {
        SplitParentProbe722F: {CfwP722F, CfwQ722F},
        SplitChildProbe722F: {CfwP722F},
    }
    outcome = _resolve(mapping, Feature(SPLIT_FEATURE))

    assert outcome.status is ResolutionStatus.AMBIGUOUS
    assert outcome.winner is None

    parent = _candidate(outcome, SplitParentProbe722F)
    assert parent.status is CandidateStatus.SURVIVOR
    assert len(parent.frameworks) == 2
    assert _framework_evaluation(parent, CfwP722F).status is FrameworkStatus.SUPPORTED
    assert _framework_evaluation(parent, CfwQ722F).status is FrameworkStatus.SUPPORTED

    child = _candidate(outcome, SplitChildProbe722F)
    assert child.status is CandidateStatus.SURVIVOR
    assert [evaluation.framework for evaluation in child.frameworks] == [CfwP722F]
    assert _framework_evaluation(child, CfwP722F).status is FrameworkStatus.SUPPORTED


def test_lone_abstract_candidate_is_not_found_with_abstract_rejection() -> None:
    """An abstract candidate can never win; the lone abstract match is NOT_FOUND."""
    assert inspect.isabstract(AbstractProbe722F), "fixture must be abstract"

    mapping: FeatureGroupEnvironmentMapping = {AbstractProbe722F: {CfwAlpha722F}}
    outcome = _resolve(mapping, Feature(ABSTRACT_FEATURE))

    assert outcome.status is ResolutionStatus.NOT_FOUND
    assert outcome.winner is None
    candidate = _candidate(outcome, AbstractProbe722F)
    assert candidate.status is CandidateStatus.REJECTED
    assert RejectionReason.ABSTRACT in _reasons(candidate)


def test_candidate_with_empty_framework_set_is_not_found_with_no_accessible_framework() -> None:
    """The false-positive-2 core: an empty framework set is a structured rejection."""
    mapping: FeatureGroupEnvironmentMapping = {NoFrameworkProbe722F: set()}
    outcome = _resolve(mapping, Feature(NO_FRAMEWORK_FEATURE))

    assert outcome.status is ResolutionStatus.NOT_FOUND
    assert outcome.winner is None
    candidate = _candidate(outcome, NoFrameworkProbe722F)
    assert candidate.status is CandidateStatus.REJECTED
    assert RejectionReason.NO_ACCESSIBLE_FRAMEWORK in _reasons(candidate)


def test_domain_mismatch_rejects_and_matching_domain_resolves() -> None:
    """A domain mismatch is a DOMAIN rejection; the matching domain resolves."""
    mapping: FeatureGroupEnvironmentMapping = {DomainProbe722F: {CfwAlpha722F}}

    mismatched = _resolve(mapping, Feature(DOMAIN_FEATURE, domain=DOMAIN_OTHER))
    assert mismatched.status is ResolutionStatus.NOT_FOUND
    assert mismatched.winner is None
    candidate = _candidate(mismatched, DomainProbe722F)
    assert candidate.status is CandidateStatus.REJECTED
    assert RejectionReason.DOMAIN in _reasons(candidate)

    matched = _resolve(mapping, Feature(DOMAIN_FEATURE, domain=DOMAIN_MATCH))
    assert matched.status is ResolutionStatus.RESOLVED
    assert matched.winner is not None
    assert matched.winner.feature_group is DomainProbe722F


# ---------------------------------------------------------------------------
# Framework pin and link/index filters
# ---------------------------------------------------------------------------


def test_framework_pin_selects_sibling_and_marks_other_pin_excluded() -> None:
    """A framework pin resolves per-framework siblings; the loser is structured.

    Pinned shape: the losing candidate is REJECTED with a FRAMEWORK_PIN rejection,
    and its pinned-away framework evaluation carries status PIN_EXCLUDED.
    """
    mapping: FeatureGroupEnvironmentMapping = {
        PinSiblingX722F: {CfwPinX722F},
        PinSiblingY722F: {CfwPinY722F},
    }
    feature = Feature(PIN_FEATURE, compute_framework=CfwPinX722F.get_class_name())
    outcome = _resolve(mapping, feature)

    assert outcome.status is ResolutionStatus.RESOLVED
    assert outcome.winner is not None
    assert outcome.winner.feature_group is PinSiblingX722F
    assert outcome.winner.compute_frameworks == (CfwPinX722F,)

    winner = _candidate(outcome, PinSiblingX722F)
    assert winner.status is CandidateStatus.WINNER
    assert _framework_evaluation(winner, CfwPinX722F).status is FrameworkStatus.SUPPORTED

    loser = _candidate(outcome, PinSiblingY722F)
    assert loser.status is CandidateStatus.REJECTED
    assert RejectionReason.FRAMEWORK_PIN in _reasons(loser)
    assert _framework_evaluation(loser, CfwPinY722F).status is FrameworkStatus.PIN_EXCLUDED


def test_link_index_mismatch_rejects_candidate() -> None:
    """A candidate whose index columns support no request link is a LINK_INDEX rejection."""
    mapping: FeatureGroupEnvironmentMapping = {LinkedProbe722F: {CfwAlpha722F}}

    # Premise guard: without links the probe resolves.
    without_links = _resolve(mapping, Feature(LINKED_FEATURE))
    assert without_links.status is ResolutionStatus.RESOLVED

    link = Link.inner(
        JoinSpec(LinkLeftProbe722F, "probe722f_left_id"),
        JoinSpec(LinkRightProbe722F, "probe722f_right_id"),
    )
    outcome = _resolve(mapping, Feature(LINKED_FEATURE), links={link})

    assert outcome.status is ResolutionStatus.NOT_FOUND
    assert outcome.winner is None
    candidate = _candidate(outcome, LinkedProbe722F)
    assert candidate.status is CandidateStatus.REJECTED
    assert RejectionReason.LINK_INDEX in _reasons(candidate)


# ---------------------------------------------------------------------------
# Provider failure semantics
# ---------------------------------------------------------------------------


def test_property_value_rejection_is_structured_and_not_fatal() -> None:
    """PropertyValueRejection is a structured VALUE_REJECTION; a clean rival still wins."""
    mapping: FeatureGroupEnvironmentMapping = {
        ValueRejectionProbe722F: {CfwAlpha722F},
        ValueRejectionWinnerProbe722F: {CfwAlpha722F},
    }
    outcome = _resolve(mapping, Feature(VALUE_REJECTION_FEATURE))

    assert outcome.status is ResolutionStatus.RESOLVED
    assert outcome.winner is not None
    assert outcome.winner.feature_group is ValueRejectionWinnerProbe722F
    assert outcome.failures == ()

    rejected = _candidate(outcome, ValueRejectionProbe722F)
    assert rejected.status is CandidateStatus.REJECTED
    rejections_by_reason = {rejection.reason: rejection for rejection in rejected.rejections}
    assert RejectionReason.VALUE_REJECTION in rejections_by_reason
    assert VALUE_REJECTION_TEXT in rejections_by_reason[RejectionReason.VALUE_REJECTION].detail


def test_plain_value_error_from_criteria_fails_closed_despite_clean_winner() -> None:
    """Any non-rejection exception from criteria is fatal; a clean winner cannot hide it."""
    mapping: FeatureGroupEnvironmentMapping = {
        BoomProbe722F: {CfwAlpha722F},
        BoomWinnerProbe722F: {CfwAlpha722F},
    }
    outcome = _resolve(mapping, Feature(BOOM_FEATURE))

    assert outcome.status is ResolutionStatus.FAILED
    assert outcome.winner is None

    failing = _candidate(outcome, BoomProbe722F)
    assert failing.status is CandidateStatus.FAILED
    assert failing.failure is not None
    assert failing.failure.category == "ValueError"
    assert failing.failure.message == BOOM_ERROR_TEXT
    assert failing.failure.plugin == PluginIdentity.from_class(BoomProbe722F)

    clean = _candidate(outcome, BoomWinnerProbe722F)
    assert clean.status is CandidateStatus.SURVIVOR

    assert len(outcome.failures) >= 1
    assert any(failure.category == "ValueError" and failure.message == BOOM_ERROR_TEXT for failure in outcome.failures)


def test_capability_hook_raise_on_remaining_framework_fails_closed() -> None:
    """A raising capability hook on a remaining framework is HOOK_FAILED, never degraded open."""
    mapping: FeatureGroupEnvironmentMapping = {CapabilityRaisesProbe722F: {CfwCap722F}}
    outcome = _resolve(mapping, Feature(CAPABILITY_RAISES_FEATURE))

    assert outcome.status is ResolutionStatus.FAILED
    assert outcome.winner is None

    candidate = _candidate(outcome, CapabilityRaisesProbe722F)
    assert candidate.status is CandidateStatus.FAILED
    assert candidate.failure is not None
    assert _framework_evaluation(candidate, CfwCap722F).status is FrameworkStatus.HOOK_FAILED

    assert len(outcome.failures) >= 1
    assert any(
        failure.category == "RuntimeError" and failure.message == CAPABILITY_ERROR_TEXT for failure in outcome.failures
    )

    # The outcome carries no exception object and no traceback: it serializes as plain data.
    dumped = json.dumps(outcome.to_payload())
    assert "Traceback" not in dumped
    assert "object at 0x" not in dumped


def test_capability_hook_false_rejects_framework_and_candidate() -> None:
    """A capability hook returning False is CAPABILITY_REJECTED, silent and non-fatal."""
    mapping: FeatureGroupEnvironmentMapping = {CapabilityFalseProbe722F: {CfwCap722F}}
    outcome = _resolve(mapping, Feature(CAPABILITY_FALSE_FEATURE))

    assert outcome.status is ResolutionStatus.NOT_FOUND
    assert outcome.winner is None
    assert outcome.failures == ()

    candidate = _candidate(outcome, CapabilityFalseProbe722F)
    assert candidate.status is CandidateStatus.REJECTED
    assert RejectionReason.CAPABILITY in _reasons(candidate)
    assert _framework_evaluation(candidate, CfwCap722F).status is FrameworkStatus.CAPABILITY_REJECTED


def test_multiple_pinned_frameworks_fail_resolution() -> None:
    """A request pinning more than one framework is FAILED with the engine's text."""
    mapping: FeatureGroupEnvironmentMapping = {MultiPinProbe722F: {CfwMultiOne722F, CfwMultiTwo722F}}
    feature = Feature(MULTI_PIN_FEATURE)
    pinned: set[type[ComputeFramework]] = {CfwMultiOne722F, CfwMultiTwo722F}
    feature.compute_frameworks = pinned
    request = ResolutionRequestSnapshot.from_feature(feature)
    assert len(request.pinned_frameworks) == 2, "premise: the request must carry two pinned frameworks"

    outcome = FeatureGroupResolver().resolve(request, snapshot_from_mapping(mapping))

    assert outcome.status is ResolutionStatus.FAILED
    assert outcome.winner is None
    assert len(outcome.failures) >= 1
    haystack = json.dumps(outcome.to_payload()) + " ".join(failure.message for failure in outcome.failures)
    assert MULTI_PIN_TEXT in haystack


# ---------------------------------------------------------------------------
# Invariants, determinism, serialization
# ---------------------------------------------------------------------------


def _invariant_scenarios() -> list[tuple[FeatureGroupEnvironmentMapping, Feature]]:
    multi_pin_feature = Feature(MULTI_PIN_FEATURE)
    multi_pinned: set[type[ComputeFramework]] = {CfwMultiOne722F, CfwMultiTwo722F}
    multi_pin_feature.compute_frameworks = multi_pinned
    return [
        ({CleanProbe722F: {CfwAlpha722F}}, Feature(CLEAN_FEATURE)),
        (
            {AmbiguousOneProbe722F: {CfwAlpha722F}, AmbiguousTwoProbe722F: {CfwAlpha722F}},
            Feature(AMBIGUOUS_FEATURE),
        ),
        (
            {ShadowParentProbe722F: {CfwShadow722F}, ShadowChildProbe722F: {CfwShadow722F}},
            Feature(SHADOW_FEATURE),
        ),
        (
            {SplitParentProbe722F: {CfwP722F, CfwQ722F}, SplitChildProbe722F: {CfwP722F}},
            Feature(SPLIT_FEATURE),
        ),
        ({AbstractProbe722F: {CfwAlpha722F}}, Feature(ABSTRACT_FEATURE)),
        ({NoFrameworkProbe722F: set()}, Feature(NO_FRAMEWORK_FEATURE)),
        (
            {ValueRejectionProbe722F: {CfwAlpha722F}, ValueRejectionWinnerProbe722F: {CfwAlpha722F}},
            Feature(VALUE_REJECTION_FEATURE),
        ),
        (
            {BoomProbe722F: {CfwAlpha722F}, BoomWinnerProbe722F: {CfwAlpha722F}},
            Feature(BOOM_FEATURE),
        ),
        ({CapabilityRaisesProbe722F: {CfwCap722F}}, Feature(CAPABILITY_RAISES_FEATURE)),
        (
            {PinSiblingX722F: {CfwPinX722F}, PinSiblingY722F: {CfwPinY722F}},
            Feature(PIN_FEATURE, compute_framework=CfwPinX722F.get_class_name()),
        ),
        ({MultiPinProbe722F: {CfwMultiOne722F, CfwMultiTwo722F}}, multi_pin_feature),
    ]


def test_outcomes_hold_invariants_and_are_deterministic() -> None:
    """Every outcome holds the contract invariants and resolves deterministically.

    Invariants: a winner exists if and only if the status is RESOLVED; candidates
    are sorted by plugin identity; two identical resolve calls agree on candidate
    ordering and payload; the payload is JSON with no class reprs.
    """
    for mapping, feature in _invariant_scenarios():
        environment = snapshot_from_mapping(mapping)
        request = ResolutionRequestSnapshot.from_feature(feature)
        first = FeatureGroupResolver().resolve(request, environment)
        second = FeatureGroupResolver().resolve(request, environment)

        assert (first.winner is not None) == (first.status is ResolutionStatus.RESOLVED)

        identities = [candidate.identity for candidate in first.candidates]
        assert identities == sorted(identities)
        assert [candidate.identity for candidate in second.candidates] == identities

        first_payload = json.dumps(first.to_payload(), sort_keys=True)
        second_payload = json.dumps(second.to_payload(), sort_keys=True)
        assert first_payload == second_payload
        assert "<class" not in first_payload
        assert "object at 0x" not in first_payload

        assert isinstance(first.environment_fingerprint, str)
        assert first.environment_fingerprint != ""


# ---------------------------------------------------------------------------
# FeatureResolutionError and the IdentifyFeatureGroupClass adapter
# ---------------------------------------------------------------------------


def test_feature_resolution_error_is_value_error_carrying_outcome() -> None:
    """FeatureResolutionError subclasses ValueError and carries the structured outcome."""
    accessible: FeatureGroupEnvironmentMapping = {
        AmbiguousOneProbe722F: {CfwAlpha722F},
        AmbiguousTwoProbe722F: {CfwAlpha722F},
    }
    with pytest.raises(FeatureResolutionError) as exc_info:
        IdentifyFeatureGroupClass(
            feature=Feature(AMBIGUOUS_FEATURE),
            accessible_plugins=accessible,
            links=None,
        )
    error = exc_info.value
    assert isinstance(error, ValueError)
    assert isinstance(error.outcome, ResolutionOutcome)


def test_identify_adapter_ambiguity_raises_feature_resolution_error() -> None:
    """The adapter's ambiguity error keeps today's text and is a FeatureResolutionError."""
    accessible: FeatureGroupEnvironmentMapping = {
        AmbiguousOneProbe722F: {CfwAlpha722F},
        AmbiguousTwoProbe722F: {CfwAlpha722F},
    }
    with pytest.raises(ValueError, match="Multiple feature groups found") as exc_info:
        IdentifyFeatureGroupClass(
            feature=Feature(AMBIGUOUS_FEATURE),
            accessible_plugins=accessible,
            links=None,
        )
    assert isinstance(exc_info.value, FeatureResolutionError)


def test_identify_adapter_no_match_raises_feature_resolution_error() -> None:
    """The adapter's no-match error keeps today's text and is a FeatureResolutionError."""
    accessible: FeatureGroupEnvironmentMapping = {CleanProbe722F: {CfwAlpha722F}}
    with pytest.raises(ValueError, match="No feature groups found") as exc_info:
        IdentifyFeatureGroupClass(
            feature=Feature(UNKNOWN_FEATURE),
            accessible_plugins=accessible,
            links=None,
        )
    assert isinstance(exc_info.value, FeatureResolutionError)


def test_identify_adapter_success_returns_mapping_and_exposes_outcome() -> None:
    """On success the adapter's .get() keeps today's shape and exposes the outcome."""
    accessible: FeatureGroupEnvironmentMapping = {CleanProbe722F: {CfwAlpha722F}}
    identifier = IdentifyFeatureGroupClass(
        feature=Feature(CLEAN_FEATURE),
        accessible_plugins=accessible,
        links=None,
        data_access_collection=None,
    )
    feature_group, compute_frameworks = identifier.get()
    assert feature_group is CleanProbe722F
    assert compute_frameworks == {CfwAlpha722F}
    assert isinstance(identifier.outcome, ResolutionOutcome)
    assert identifier.outcome.status is ResolutionStatus.RESOLVED
