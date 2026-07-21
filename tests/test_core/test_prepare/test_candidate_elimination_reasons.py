"""Red-phase pins for os-011: per-candidate elimination reasons on ``EvaluationResult``.

The design contract (``scratchpad/os-011-design.md``) adds a public elimination surface to the
resolution result:

- ``Elimination`` is a frozen dataclass with ``.stage`` and ``.reason``.
- ``EvaluationResult.eliminations`` maps every non-winning candidate that reached at least the name
  filter to the ``Elimination`` recorded at the first gate it failed.
- ``render_resolution_failure`` projects those eliminations into a shared near-miss block on the
  ``none`` and ``abstract_only`` messages.

These symbols do NOT exist yet, so importing ``Elimination`` fails collection: that is the expected
Red failure. Every feature group and compute framework double is suffixed ``011`` because test feature
groups become global ``FeatureGroup`` subclasses and must not collide with other suites.
"""

from abc import abstractmethod
from typing import ClassVar, Optional

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
from mloda.core.prepare.identify_feature_group import (
    Elimination,
    FeatureResolutionError,
    IdentifyFeatureGroupClass,
    render_resolution_failure,
)
from tests.test_core.test_prepare.identify_seam import evaluate_or_raise, identify_winner


# --- feature names, one per scenario ------------------------------------------------------------

VALUE_FEATURE = "elim_value_reject_feat_011"
DOMAIN_FEATURE = "elim_domain_feat_011"
SCOPE_FEATURE = "elim_scope_feat_011"
CAPABILITY_FEATURE = "elim_capability_feat_011"
NOT_ENABLED_FEATURE = "elim_not_enabled_feat_011"
PIN_FEATURE = "elim_pin_feat_011"
LINKS_FEATURE = "elim_links_feat_011"
ABSTRACT_FEATURE = "elim_abstract_feat_011"
SORT_FEATURE = "elim_sort_feat_011"
PRECEDENCE_CAP_FEATURE = "elim_precedence_cap_feat_011"
PRECEDENCE_NE_FEATURE = "elim_precedence_ne_feat_011"
SUCCESS_FEATURE = "elim_success_feat_011"
NONMATCH_ONLY_NAME = "elim_only_this_other_name_011"
WIN_WITH_REJECTOR_FEATURE = "elim_win_with_rejector_feat_011"
DOMAIN_AND_VALUE_REJECT_FEATURE = "elim_domain_and_value_reject_feat_011"
REPROBE_FEATURE = "elim_reprobe_feat_011"
SCOPED_ABSTRACT_SCOPE = "_ElimBaseFG"

REQUESTED_DOMAIN = "elim_requested_domain_011"
CANDIDATE_DOMAIN = "elim_candidate_domain_011"
UNRELATED_SCOPE = "ElimUnrelatedScope011"

# Value-rejection wording matches today's ``_strict_validation_rejection_reason`` output.
VALUE_REJECT_REASON = "Property value '14' failed validation for 'window_size'"
WIN_REJECT_REASON = "Property value 'bad' rejected by match_guard for 'mode'"
DOMAIN_VALUE_REASON = "Property value '7' failed validation for 'k'"
REPROBE_REASON = "a criteria-matched candidate must never be re-probed as a value rejection"

# A criteria-matched candidate's value must be inspected once (at match time). A failure-path capture that
# re-probed such a candidate via ``_strict_validation_rejection_reason`` would inspect it a second time.
REPROBE_INSPECT_CALLS: dict[str, int] = {}


# --- compute framework doubles ------------------------------------------------------------------


class ElimFwOne011(ComputeFramework):
    """First dummy compute framework for the elimination-reason tests."""


class ElimFwTwo011(ComputeFramework):
    """Second dummy compute framework for the elimination-reason tests."""


class ElimFwThree011(ComputeFramework):
    """Third dummy compute framework, used only to pin a feature away from every candidate."""


# --- feature group doubles ----------------------------------------------------------------------


class _ElimBaseFG(FeatureGroup):
    """Minimal, fully declarative feature group double driven by class variables.

    Every hook is deterministic and non-raising (except the value-rejection subclass, which raises
    only for its own matching name), so a leaked global subclass stays inert for other suites.
    """

    MATCHES: ClassVar[frozenset[str]] = frozenset()
    DOMAIN_NAME: ClassVar[Optional[str]] = None
    FRAMEWORK_RULE: ClassVar[Optional[set[type[ComputeFramework]]]] = None
    SUPPORTED_FRAMEWORKS: ClassVar[Optional[frozenset[str]]] = None
    INDEX_COLUMNS: ClassVar[Optional[list[Index]]] = None
    SUPPORTS_INDEX_RESULT: ClassVar[Optional[bool]] = None

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) in cls.MATCHES

    @classmethod
    def get_domain(cls) -> Domain:
        if cls.DOMAIN_NAME is None:
            return Domain.get_default_domain()
        return Domain(cls.DOMAIN_NAME)

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return cls.FRAMEWORK_RULE

    @classmethod
    def supports_compute_framework(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        compute_framework: type[ComputeFramework],
    ) -> bool:
        if cls.SUPPORTED_FRAMEWORKS is None:
            return True
        return compute_framework.get_class_name() in cls.SUPPORTED_FRAMEWORKS

    @classmethod
    def index_columns(cls) -> Optional[list[Index]]:
        return cls.INDEX_COLUMNS

    @classmethod
    def supports_index(cls, index: Index) -> Optional[bool]:
        return cls.SUPPORTS_INDEX_RESULT

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class ElimValueRejectFG011(_ElimBaseFG):
    """Name matches, but the criteria hook raises ``PropertyValueRejection`` on the value.

    The first match pass records the raised message as the value_rejection reason, so the rejection's own
    wording is what surfaces; no diagnostic hook is re-probed on the failure path.
    """

    MATCHES = frozenset({VALUE_FEATURE})
    FRAMEWORK_RULE = {ElimFwOne011}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        if str(feature_name) in cls.MATCHES:
            raise PropertyValueRejection(VALUE_REJECT_REASON)
        return False


class ElimDomainFG011(_ElimBaseFG):
    """Name matches, but the candidate declares a domain the run did not request."""

    MATCHES = frozenset({DOMAIN_FEATURE})
    DOMAIN_NAME = CANDIDATE_DOMAIN
    FRAMEWORK_RULE = {ElimFwOne011}


class ElimScopeFG011(_ElimBaseFG):
    """Name matches, but the candidate is outside the requested feature-group scope."""

    MATCHES = frozenset({SCOPE_FEATURE})
    FRAMEWORK_RULE = {ElimFwOne011}


class ElimCapabilityFG011(_ElimBaseFG):
    """Reaches the framework split with an empty supported set while a framework WAS enabled."""

    MATCHES = frozenset({CAPABILITY_FEATURE})
    FRAMEWORK_RULE = {ElimFwOne011, ElimFwTwo011}
    SUPPORTED_FRAMEWORKS = frozenset({"ElimFwTwo011"})


class ElimNotEnabledFG011(_ElimBaseFG):
    """Reaches the framework split with an empty supported set and NO framework enabled for the run."""

    MATCHES = frozenset({NOT_ENABLED_FEATURE})
    FRAMEWORK_RULE = {ElimFwOne011}


class ElimPinFG011(_ElimBaseFG):
    """Supports its enabled framework, but the run pins a framework it does not support."""

    MATCHES = frozenset({PIN_FEATURE})
    FRAMEWORK_RULE = {ElimFwOne011}


class ElimLinksFG011(_ElimBaseFG):
    """Supports its enabled framework and no pin, but no index column matches the run's links."""

    MATCHES = frozenset({LINKS_FEATURE})
    FRAMEWORK_RULE = {ElimFwOne011}
    INDEX_COLUMNS = [Index(("elim_index_col_011",))]
    SUPPORTS_INDEX_RESULT = False


class ElimAbstractBaseFG011(_ElimBaseFG):
    """Abstract base that matches the name but can never be instantiated (never an elimination)."""

    MATCHES = frozenset({ABSTRACT_FEATURE})
    FRAMEWORK_RULE = {ElimFwOne011}

    @classmethod
    @abstractmethod
    def _elim_abstract_hook_011(cls) -> str:
        """Abstract hook that keeps this base uninstantiable."""


class ElimAbstractNearMissFG011(_ElimBaseFG):
    """Concrete near-miss sharing the abstract base's name, eliminated at the capability gate."""

    MATCHES = frozenset({ABSTRACT_FEATURE})
    FRAMEWORK_RULE = {ElimFwOne011, ElimFwTwo011}
    SUPPORTED_FRAMEWORKS = frozenset({"ElimFwTwo011"})


class ElimSortAFG011(_ElimBaseFG):
    """First of two capability near-misses; its name sorts before ElimSortBFG011."""

    MATCHES = frozenset({SORT_FEATURE})
    FRAMEWORK_RULE = {ElimFwOne011, ElimFwTwo011}
    SUPPORTED_FRAMEWORKS = frozenset({"ElimFwTwo011"})


class ElimSortBFG011(_ElimBaseFG):
    """Second capability near-miss; its name sorts after ElimSortAFG011."""

    MATCHES = frozenset({SORT_FEATURE})
    FRAMEWORK_RULE = {ElimFwOne011, ElimFwTwo011}
    SUPPORTED_FRAMEWORKS = frozenset({"ElimFwTwo011"})


class ElimPrecedenceCapFG011(_ElimBaseFG):
    """Empty supported set with a framework enabled AND a pin set: capability must win over the pin."""

    MATCHES = frozenset({PRECEDENCE_CAP_FEATURE})
    FRAMEWORK_RULE = {ElimFwOne011, ElimFwTwo011}
    SUPPORTED_FRAMEWORKS = frozenset({"ElimFwTwo011"})


class ElimPrecedenceNotEnabledFG011(_ElimBaseFG):
    """Empty supported set, no framework enabled, AND a pin set: not-enabled must win over the pin."""

    MATCHES = frozenset({PRECEDENCE_NE_FEATURE})
    FRAMEWORK_RULE = {ElimFwOne011}


class ElimSuccessFG011(_ElimBaseFG):
    """Sole winner: matches the name and supports its enabled framework."""

    MATCHES = frozenset({SUCCESS_FEATURE})
    FRAMEWORK_RULE = {ElimFwOne011}


class ElimNonMatchingFG011(_ElimBaseFG):
    """Matches a different name; never a near-miss for the scenarios below."""

    MATCHES = frozenset({NONMATCH_ONLY_NAME})
    FRAMEWORK_RULE = {ElimFwOne011}


class ElimWinnerFG011(_ElimBaseFG):
    """Sole winner of the win-with-losing-rejector scenario; matches the shared name and supports its framework."""

    MATCHES = frozenset({WIN_WITH_REJECTOR_FEATURE})
    FRAMEWORK_RULE = {ElimFwOne011}


class ElimLosingRejectorFG011(_ElimBaseFG):
    """Matches the shared name but rejects the option VALUE, recording a reportable reason: a losing near-miss.

    The rejection is name-guarded so a leaked global subclass records nothing for any other suite's feature.
    """

    MATCHES = frozenset({WIN_WITH_REJECTOR_FEATURE})

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        # Name matches, but the value is rejected: the first pass records the reason as the match fails.
        if str(feature_name) == WIN_WITH_REJECTOR_FEATURE:
            raise PropertyValueRejection(WIN_REJECT_REASON)
        return False


class ElimDomainAndValueRejectFG011(_ElimBaseFG):
    """Rejects the option VALUE (criteria fails first) AND declares an unrequested domain.

    The value_rejection is recorded at the criteria gate, before the domain gate is ever consulted, so the
    stage is value_rejection rather than domain. The reason is name-guarded to stay inert when leaked.
    """

    MATCHES = frozenset({DOMAIN_AND_VALUE_REJECT_FEATURE})
    DOMAIN_NAME = CANDIDATE_DOMAIN

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        # The value is rejected at the criteria gate, recording the reason before the domain gate is reached.
        if str(feature_name) == DOMAIN_AND_VALUE_REJECT_FEATURE:
            raise PropertyValueRejection(DOMAIN_VALUE_REASON)
        return False


class _ElimReprobeFG011(_ElimBaseFG):
    """Criteria-matched candidate whose value is inspected once, at match time.

    Both match and the rejection hook increment the SAME per-class counter, so a re-probe of this matched
    candidate via ``_strict_validation_rejection_reason`` would push its count to two. The hook is name-guarded
    to stay inert when leaked into another suite's candidate universe.
    """

    FRAMEWORK_RULE = {ElimFwOne011}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        matched = str(feature_name) in cls.MATCHES
        if matched:
            REPROBE_INSPECT_CALLS[cls.get_class_name()] = REPROBE_INSPECT_CALLS.get(cls.get_class_name(), 0) + 1
        return matched

    @classmethod
    def _strict_validation_rejection_reason(cls, feature_name: str | FeatureName, options: Options) -> str | None:
        if str(feature_name) != REPROBE_FEATURE:
            return None
        REPROBE_INSPECT_CALLS[cls.get_class_name()] = REPROBE_INSPECT_CALLS.get(cls.get_class_name(), 0) + 1
        return REPROBE_REASON


class ElimReprobeAFG011(_ElimReprobeFG011):
    """First of two matched winners sharing the re-probe name; the pair triggers a 'multiple' failure."""

    MATCHES = frozenset({REPROBE_FEATURE})


class ElimReprobeBFG011(_ElimReprobeFG011):
    """Second matched winner sharing the re-probe name."""

    MATCHES = frozenset({REPROBE_FEATURE})


# A link whose indexes ElimLinksFG011 does not support, driving the links gate to reject.
ELIM_LINK = Link.inner(
    JoinSpec(ElimLinksFG011, "elim_left_index_011"),
    JoinSpec(ElimLinksFG011, "elim_right_index_011"),
)


# --- helpers ------------------------------------------------------------------------------------


def _fail(
    feature: Feature,
    accessible_plugins: FeatureGroupEnvironmentMapping,
    links: Optional[set[Link]] = None,
) -> FeatureResolutionError:
    """Drive the engine seam and return the raised typed error (carrying ``.result`` and message)."""
    with pytest.raises(FeatureResolutionError) as excinfo:
        evaluate_or_raise(feature, accessible_plugins, links=links)
    return excinfo.value


class TestEliminationStages:
    """One test per stage: the eliminated candidate maps to the expected (stage, reason), and the
    near-miss message line carries the contract's stage label."""

    def test_value_rejection_stage(self) -> None:
        feature = Feature(VALUE_FEATURE)
        plugins: FeatureGroupEnvironmentMapping = {ElimValueRejectFG011: {ElimFwOne011}}

        err = _fail(feature, plugins)

        assert err.result.eliminations[ElimValueRejectFG011] == Elimination(
            stage="value_rejection", reason=VALUE_REJECT_REASON
        )
        assert f"  - ElimValueRejectFG011 (option value): {VALUE_REJECT_REASON}" in str(err)

    def test_domain_stage(self) -> None:
        feature = Feature(DOMAIN_FEATURE, domain=REQUESTED_DOMAIN)
        plugins: FeatureGroupEnvironmentMapping = {ElimDomainFG011: {ElimFwOne011}}

        err = _fail(feature, plugins)

        reason = f"declares domain '{CANDIDATE_DOMAIN}', but the run requested '{REQUESTED_DOMAIN}'"
        assert err.result.eliminations[ElimDomainFG011] == Elimination(stage="domain", reason=reason)
        assert f"  - ElimDomainFG011 (domain): {reason}" in str(err)

    def test_scope_stage(self) -> None:
        feature = Feature(SCOPE_FEATURE, feature_group=UNRELATED_SCOPE)
        plugins: FeatureGroupEnvironmentMapping = {ElimScopeFG011: {ElimFwOne011}}

        err = _fail(feature, plugins)

        reason = "outside the requested feature group scope"
        assert err.result.eliminations[ElimScopeFG011] == Elimination(stage="scope", reason=reason)
        assert f"  - ElimScopeFG011 (scope): {reason}" in str(err)

    def test_capability_stage(self) -> None:
        feature = Feature(CAPABILITY_FEATURE)
        plugins: FeatureGroupEnvironmentMapping = {ElimCapabilityFG011: {ElimFwOne011}}

        err = _fail(feature, plugins)

        reason = "supports_compute_framework rejected ['ElimFwOne011']"
        assert err.result.eliminations[ElimCapabilityFG011] == Elimination(stage="capability", reason=reason)
        assert f"  - ElimCapabilityFG011 (compute framework): {reason}" in str(err)

    def test_frameworks_not_enabled_stage(self) -> None:
        feature = Feature(NOT_ENABLED_FEATURE)
        plugins: FeatureGroupEnvironmentMapping = {ElimNotEnabledFG011: set()}

        err = _fail(feature, plugins)

        reason = "none of its compute frameworks are enabled for this run"
        assert err.result.eliminations[ElimNotEnabledFG011] == Elimination(
            stage="frameworks_not_enabled", reason=reason
        )
        assert f"  - ElimNotEnabledFG011 (compute framework): {reason}" in str(err)

    def test_framework_pin_stage(self) -> None:
        feature = Feature(PIN_FEATURE, compute_framework="ElimFwThree011")
        plugins: FeatureGroupEnvironmentMapping = {ElimPinFG011: {ElimFwOne011}}

        err = _fail(feature, plugins)

        reason = "pinned compute framework 'ElimFwThree011' is not among its supported ['ElimFwOne011']"
        assert err.result.eliminations[ElimPinFG011] == Elimination(stage="framework_pin", reason=reason)
        assert f"  - ElimPinFG011 (compute framework pin): {reason}" in str(err)

    def test_links_stage(self) -> None:
        feature = Feature(LINKS_FEATURE)
        plugins: FeatureGroupEnvironmentMapping = {ElimLinksFG011: {ElimFwOne011}}

        err = _fail(feature, plugins, links={ELIM_LINK})

        reason = "no index column matches the run's links"
        assert err.result.eliminations[ElimLinksFG011] == Elimination(stage="links", reason=reason)
        assert f"  - ElimLinksFG011 (links): {reason}" in str(err)


class TestNearMissMessageBlock:
    """The shared near-miss block projects the eliminations onto the none / abstract_only messages."""

    def test_none_message_renders_sorted_near_miss_block(self) -> None:
        feature = Feature(SORT_FEATURE)
        # B inserted before A so a sorted rendering (by class name, then module) is observable.
        plugins: FeatureGroupEnvironmentMapping = {
            ElimSortBFG011: {ElimFwOne011},
            ElimSortAFG011: {ElimFwOne011},
        }

        err = _fail(feature, plugins)

        assert err.result.failure_kind == "none"
        message = str(err)
        assert message.startswith(f"No feature groups found for feature name: '{SORT_FEATURE}'.")
        assert (
            f"Feature group(s) eliminated while matching '{SORT_FEATURE}':\n"
            "  - ElimSortAFG011 (compute framework): supports_compute_framework rejected ['ElimFwOne011']\n"
            "  - ElimSortBFG011 (compute framework): supports_compute_framework rejected ['ElimFwOne011']"
        ) in message

    def test_abstract_only_message_appends_near_miss_block(self) -> None:
        feature = Feature(ABSTRACT_FEATURE)
        plugins: FeatureGroupEnvironmentMapping = {
            ElimAbstractBaseFG011: {ElimFwOne011},
            ElimAbstractNearMissFG011: {ElimFwOne011},
        }

        err = _fail(feature, plugins)

        assert err.result.failure_kind == "abstract_only"
        # The abstract base is handled by abstract_matched, never recorded as an elimination.
        assert ElimAbstractBaseFG011 not in err.result.eliminations
        assert err.result.eliminations[ElimAbstractNearMissFG011] == Elimination(
            stage="capability", reason="supports_compute_framework rejected ['ElimFwOne011']"
        )

        message = str(err)
        # The existing abstract sentence is retained ...
        assert f"No feature groups found for feature name: '{ABSTRACT_FEATURE}'." in message
        # ... and the shared near-miss block is appended.
        assert (
            f"Feature group(s) eliminated while matching '{ABSTRACT_FEATURE}':\n"
            "  - ElimAbstractNearMissFG011 (compute framework): supports_compute_framework rejected ['ElimFwOne011']"
        ) in message


class TestEmptySupportedPrecedence:
    """An empty supported set is decided before the pin filter: capability / not-enabled win over pin."""

    def test_pin_over_empty_supported_reports_capability_not_pin(self) -> None:
        feature = Feature(PRECEDENCE_CAP_FEATURE, compute_framework="ElimFwThree011")
        plugins: FeatureGroupEnvironmentMapping = {ElimPrecedenceCapFG011: {ElimFwOne011}}

        err = _fail(feature, plugins)

        elimination = err.result.eliminations[ElimPrecedenceCapFG011]
        assert elimination.stage == "capability"
        assert elimination.reason == "supports_compute_framework rejected ['ElimFwOne011']"

    def test_pin_over_empty_supported_none_enabled_reports_frameworks_not_enabled(self) -> None:
        feature = Feature(PRECEDENCE_NE_FEATURE, compute_framework="ElimFwThree011")
        plugins: FeatureGroupEnvironmentMapping = {ElimPrecedenceNotEnabledFG011: set()}

        err = _fail(feature, plugins)

        elimination = err.result.eliminations[ElimPrecedenceNotEnabledFG011]
        assert elimination.stage == "frameworks_not_enabled"
        assert elimination.reason == "none of its compute frameworks are enabled for this run"


class TestNoEliminationRecorded:
    """Winners, abstract bases, and name-mismatched candidates are never near-misses."""

    def test_winner_has_no_elimination_entry(self) -> None:
        feature = Feature(SUCCESS_FEATURE)
        plugins: FeatureGroupEnvironmentMapping = {ElimSuccessFG011: {ElimFwOne011}}

        winner_fg, _ = identify_winner(feature, plugins)
        assert winner_fg is ElimSuccessFG011

        result = IdentifyFeatureGroupClass.evaluate(feature, plugins, None)
        assert result.failure_kind is None
        assert result.eliminations == {}

    def test_abstract_base_has_no_elimination_entry(self) -> None:
        feature = Feature(ABSTRACT_FEATURE)
        plugins: FeatureGroupEnvironmentMapping = {ElimAbstractBaseFG011: {ElimFwOne011}}

        err = _fail(feature, plugins)

        assert err.result.failure_kind == "abstract_only"
        assert ElimAbstractBaseFG011 not in err.result.eliminations
        assert err.result.eliminations == {}

    def test_name_mismatch_has_no_elimination_entry(self) -> None:
        feature = Feature(VALUE_FEATURE)
        plugins: FeatureGroupEnvironmentMapping = {
            ElimValueRejectFG011: {ElimFwOne011},
            ElimNonMatchingFG011: {ElimFwOne011},
        }

        err = _fail(feature, plugins)

        # The value-rejecting candidate matched the name and is recorded ...
        assert ElimValueRejectFG011 in err.result.eliminations
        # ... but the candidate whose name never matched is not a near-miss.
        assert ElimNonMatchingFG011 not in err.result.eliminations


class TestValueRejectionCaptureCorrectness:
    """value_rejection is recorded for a criteria-FAILING candidate at that first gate, regardless of domain
    or scope and regardless of the overall outcome; a criteria-matched candidate is never re-probed."""

    def test_losing_value_rejector_recorded_even_when_a_sibling_wins(self) -> None:
        """A losing value-rejecter is a near-miss even when a sibling WINS and the resolution succeeds."""
        feature = Feature(WIN_WITH_REJECTOR_FEATURE)
        plugins: FeatureGroupEnvironmentMapping = {
            ElimWinnerFG011: {ElimFwOne011},
            ElimLosingRejectorFG011: {ElimFwOne011},
        }

        result = IdentifyFeatureGroupClass.evaluate(feature, plugins, None)

        # A sibling won, so the resolution as a whole is a success ...
        assert result.failure_kind is None
        winner_fg, _ = next(iter(result.identified.items()))
        assert winner_fg is ElimWinnerFG011
        # ... yet the losing value-rejecter is still recorded, which the old failure-only capture never did.
        assert result.eliminations[ElimLosingRejectorFG011] == Elimination(
            stage="value_rejection", reason=WIN_REJECT_REASON
        )
        assert ElimWinnerFG011 not in result.eliminations

    def test_value_rejection_recorded_even_with_a_domain_mismatch(self) -> None:
        """A candidate that value-rejects AND declares an unrequested domain is staged value_rejection.

        The criteria gate is first, so the reason is captured there; the old domain/scope-gated capture
        dropped such a candidate entirely.
        """
        feature = Feature(DOMAIN_AND_VALUE_REJECT_FEATURE, domain=REQUESTED_DOMAIN)
        plugins: FeatureGroupEnvironmentMapping = {ElimDomainAndValueRejectFG011: {ElimFwOne011}}

        err = _fail(feature, plugins)

        assert err.result.eliminations[ElimDomainAndValueRejectFG011] == Elimination(
            stage="value_rejection", reason=DOMAIN_VALUE_REASON
        )

    def test_criteria_matched_candidates_are_not_reprobed_as_value_rejections(self) -> None:
        """Two matched winners (a 'multiple' failure) are each inspected once, never re-probed as rejections."""
        REPROBE_INSPECT_CALLS.clear()
        feature = Feature(REPROBE_FEATURE)
        plugins: FeatureGroupEnvironmentMapping = {
            ElimReprobeAFG011: {ElimFwOne011},
            ElimReprobeBFG011: {ElimFwOne011},
        }

        err = _fail(feature, plugins)

        assert err.result.failure_kind == "multiple"
        # Each matched candidate's value was inspected exactly once, at match time: the rejection hook, which
        # would inspect it a second time, is never asked of a criteria-matched candidate.
        assert REPROBE_INSPECT_CALLS == {"ElimReprobeAFG011": 1, "ElimReprobeBFG011": 1}
        # And no matched winner is recorded as a value_rejection near-miss.
        assert err.result.eliminations == {}


class TestScopedAbstractOnlyCallout:
    """FIX C: the scope callout in the abstract_only branch sits on the sentence line, never glued to a bullet."""

    def test_scoped_abstract_only_callout_is_not_glued_to_a_near_miss_bullet(self) -> None:
        feature = Feature(ABSTRACT_FEATURE, feature_group=SCOPED_ABSTRACT_SCOPE)
        plugins: FeatureGroupEnvironmentMapping = {
            ElimAbstractBaseFG011: {ElimFwOne011},
            ElimAbstractNearMissFG011: {ElimFwOne011},
        }

        err = _fail(feature, plugins)
        message = str(err)

        assert err.result.failure_kind == "abstract_only"
        # A near-miss block is present, so a glued callout would land on the last bullet line.
        assert f"Feature group(s) eliminated while matching '{ABSTRACT_FEATURE}':" in message
        assert message.count("Scoped to feature group:") == 1
        callout_line = next(line for line in message.split("\n") if "Scoped to feature group:" in line)
        assert not callout_line.startswith("  - ")
        assert callout_line.startswith(f"No feature groups found for feature name: '{ABSTRACT_FEATURE}'.")


def test_render_resolution_failure_is_importable() -> None:
    """Guard that the renderer entry point the near-miss block lives in stays importable."""
    assert callable(render_resolution_failure)
