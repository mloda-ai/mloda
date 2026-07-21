"""Contract tests for the resolution test-seam helpers in identify_seam.py (board issue os-014).

``evaluate_or_raise`` and ``identify_winner`` are the canonical raise-on-failure entry points for
resolution tests, mirroring the removed raising IdentifyFeatureGroupClass wrapper. Until now they were
only exercised indirectly by consumers; this suite pins their own promised contract: the success path
returns the same structured outcome as ``IdentifyFeatureGroupClass.evaluate(...)``, and the failure path
raises a typed ``FeatureResolutionError`` whose message is exactly the pure-renderer projection.
"""

from abc import abstractmethod
from typing import Any, Optional

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.domain import Domain
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import (
    EvaluationResult,
    FeatureResolutionError,
    IdentifyFeatureGroupClass,
    render_resolution_failure,
)
from tests.test_core.test_prepare.identify_seam import evaluate_or_raise, identify_winner


# Unique names/tokens so nothing collides with the parallel suite (issue os-014).
SEAM_CONTRACT_MATCH_FEATURE = "seam_contract_014_single_match"
SEAM_CONTRACT_NO_MATCH_FEATURE = "seam_contract_014_no_match_at_all"
SEAM_CONTRACT_MULTIPLE_FEATURE = "seam_contract_014_multiple_match"
SEAM_CONTRACT_ABSTRACT_FEATURE = "seam_contract_014_abstract_match"


class SeamContract014Fw(ComputeFramework):
    """Dummy compute framework for the seam-contract tests."""


class SeamContract014FwBeta(ComputeFramework):
    """Second dummy compute framework, distinct from SeamContract014Fw."""


class SeamContract014FG(FeatureGroup):
    """Concrete feature group matching exactly one unique seam-contract feature name."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == SEAM_CONTRACT_MATCH_FEATURE

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class SeamContract014SiblingAFG(FeatureGroup):
    """First of two unrelated siblings matching the same name (distinct domain 'seam_contract_014_a')."""

    @classmethod
    def get_domain(cls) -> Domain:
        return Domain("seam_contract_014_a")

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == SEAM_CONTRACT_MULTIPLE_FEATURE

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class SeamContract014SiblingBFG(FeatureGroup):
    """Second unrelated sibling matching the same name (distinct domain 'seam_contract_014_b')."""

    @classmethod
    def get_domain(cls) -> Domain:
        return Domain("seam_contract_014_b")

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == SEAM_CONTRACT_MULTIPLE_FEATURE

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class SeamContract014AbstractFG(FeatureGroup):
    """Abstract base matching a unique name; uninstantiable via an unimplemented abstract hook.

    No concrete subclass is registered, so this base is the only name-match and can never win.
    """

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == SEAM_CONTRACT_ABSTRACT_FEATURE

    @classmethod
    @abstractmethod
    def _seam_contract_014_abstract_hook(cls, data: Any) -> Any:
        """Abstract hook that makes this base uninstantiable."""
        ...

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class TestEvaluateOrRaiseSuccess:
    """evaluate_or_raise returns the same structured outcome as evaluate() when a single winner exists."""

    def test_returns_success_result_matching_evaluate(self) -> None:
        """One concrete match: no raise, failure_kind is None, and identified equals evaluate()'s."""
        feature = Feature(SEAM_CONTRACT_MATCH_FEATURE)
        accessible_plugins: FeatureGroupEnvironmentMapping = {SeamContract014FG: {SeamContract014Fw}}

        result = evaluate_or_raise(feature, accessible_plugins, links=None)
        direct = IdentifyFeatureGroupClass.evaluate(feature, accessible_plugins, links=None)

        assert isinstance(result, EvaluationResult)
        assert result.failure_kind is None
        assert result.identified == direct.identified
        assert result.identified == {SeamContract014FG: {SeamContract014Fw}}


class TestEvaluateOrRaiseFailure:
    """evaluate_or_raise raises the typed error whose message is exactly the renderer projection."""

    def test_no_match_raises_feature_resolution_error(self) -> None:
        """A name no group matches raises FeatureResolutionError projecting the pure renderer text."""
        feature = Feature(SEAM_CONTRACT_NO_MATCH_FEATURE)
        accessible_plugins: FeatureGroupEnvironmentMapping = {SeamContract014FG: {SeamContract014Fw}}

        with pytest.raises(FeatureResolutionError) as exc_info:
            evaluate_or_raise(feature, accessible_plugins, links=None)
        err = exc_info.value
        direct = IdentifyFeatureGroupClass.evaluate(feature, accessible_plugins, links=None)

        assert isinstance(err, ValueError)
        assert err.feature_name == str(feature.name)
        assert err.result.failure_kind == "none"
        assert err.result.identified == direct.identified
        assert str(err) == render_resolution_failure(direct, feature)


class TestIdentifyWinnerSuccess:
    """identify_winner returns the single winning (feature_group, frameworks) pair."""

    def test_returns_winning_pair(self) -> None:
        """The 2-tuple equals the single entry of the winning identified mapping."""
        feature = Feature(SEAM_CONTRACT_MATCH_FEATURE)
        accessible_plugins: FeatureGroupEnvironmentMapping = {SeamContract014FG: {SeamContract014Fw}}

        winner = identify_winner(feature, accessible_plugins, links=None)
        result = IdentifyFeatureGroupClass.evaluate(feature, accessible_plugins, links=None)

        assert winner == next(iter(result.identified.items()))
        feature_group, frameworks = winner
        assert feature_group is SeamContract014FG
        assert frameworks == {SeamContract014Fw}


class TestIdentifyWinnerFailure:
    """identify_winner raises the typed error when nothing matches."""

    def test_no_match_raises_feature_resolution_error(self) -> None:
        """A name no group matches raises FeatureResolutionError before any winner is returned."""
        feature = Feature(SEAM_CONTRACT_NO_MATCH_FEATURE)
        accessible_plugins: FeatureGroupEnvironmentMapping = {SeamContract014FG: {SeamContract014Fw}}

        with pytest.raises(FeatureResolutionError) as exc_info:
            identify_winner(feature, accessible_plugins, links=None)
        err = exc_info.value

        assert isinstance(err, ValueError)
        assert err.feature_name == str(feature.name)


class TestEvaluateOrRaiseMultiple:
    """evaluate_or_raise raises when more than one concrete candidate survives."""

    def test_multiple_matches_raise(self) -> None:
        """Two unrelated siblings match the same name on distinct frameworks: failure_kind 'multiple'."""
        feature = Feature(SEAM_CONTRACT_MULTIPLE_FEATURE)
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            SeamContract014SiblingAFG: {SeamContract014Fw},
            SeamContract014SiblingBFG: {SeamContract014FwBeta},
        }

        with pytest.raises(FeatureResolutionError) as exc_info:
            evaluate_or_raise(feature, accessible_plugins, links=None)

        assert exc_info.value.result.failure_kind == "multiple"


class TestIdentifyWinnerMultiple:
    """identify_winner must raise on ambiguity rather than return an arbitrary winner."""

    def test_multiple_matches_raise(self) -> None:
        """Two surviving candidates: identify_winner raises instead of picking next(iter(...))."""
        feature = Feature(SEAM_CONTRACT_MULTIPLE_FEATURE)
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            SeamContract014SiblingAFG: {SeamContract014Fw},
            SeamContract014SiblingBFG: {SeamContract014FwBeta},
        }

        with pytest.raises(FeatureResolutionError):
            identify_winner(feature, accessible_plugins, links=None)


class TestEvaluateOrRaiseAbstractOnly:
    """evaluate_or_raise raises when only an uninstantiable abstract base matches."""

    def test_abstract_only_match_raises(self) -> None:
        """Only an abstract base matches name+domain+scope: failure_kind 'abstract_only'."""
        feature = Feature(SEAM_CONTRACT_ABSTRACT_FEATURE)
        accessible_plugins: FeatureGroupEnvironmentMapping = {SeamContract014AbstractFG: {SeamContract014Fw}}

        with pytest.raises(FeatureResolutionError) as exc_info:
            evaluate_or_raise(feature, accessible_plugins, links=None)

        assert exc_info.value.result.failure_kind == "abstract_only"
        assert SeamContract014AbstractFG in exc_info.value.result.abstract_matched
