"""Pinning tests for the non-raising evaluation seam on IdentifyFeatureGroupClass (issue #754).

The engine wrapper ``IdentifyFeatureGroupClass(...)`` validates and RAISES ``ValueError`` on
any unresolved / ambiguous match. This suite pins a parallel, non-raising classmethod
``IdentifyFeatureGroupClass.evaluate(feature, accessible_plugins, links, data_access_collection=None)``
that runs the same matching/filter logic but returns a structured ``EvaluationResult`` instead of
raising. Both symbols are imported from ``mloda.core.prepare.identify_feature_group``; the seam is
implemented, and these tests pin its non-raising contract.
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
from mloda.core.api.plugin_docs import resolve_feature
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import (
    ComputeFrameworkPinError,
    EvaluationResult,
    IdentifyFeatureGroupClass,
)
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


# Unique feature names so no other double in the parallel suite collides on them.
SEAM_SINGLE_FEATURE = "seam_single_match_test_754"
SEAM_MULTIPLE_FEATURE = "seam_multiple_match_test_754"
SEAM_ABSTRACT_FEATURE = "seam_abstract_only_test_754"
SEAM_NO_MATCH_FEATURE = "seam_no_match_at_all_test_754"

# A name no group matches, used to prove the pin-cardinality check fires BEFORE matching (issue #851).
PIN_NO_MATCH_FEATURE = "pin_no_match_test_851"


class SeamFwAlpha(ComputeFramework):
    """First dummy compute framework for the evaluation-seam tests."""


class SeamFwBeta(ComputeFramework):
    """Second dummy compute framework, distinct from SeamFwAlpha."""


class SeamSingleFG(FeatureGroup):
    """Concrete feature group matching exactly one unique feature name."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == SEAM_SINGLE_FEATURE

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class SeamSiblingAFG(FeatureGroup):
    """First of two unrelated siblings matching the same name (distinct domain 'seam_a')."""

    @classmethod
    def get_domain(cls) -> Domain:
        return Domain("seam_a")

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == SEAM_MULTIPLE_FEATURE

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class SeamSiblingBFG(FeatureGroup):
    """Second unrelated sibling matching the same name (distinct domain 'seam_b')."""

    @classmethod
    def get_domain(cls) -> Domain:
        return Domain("seam_b")

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == SEAM_MULTIPLE_FEATURE

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class SeamAbstractOnlyFG(FeatureGroup):
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
        return str(feature_name) == SEAM_ABSTRACT_FEATURE

    @classmethod
    @abstractmethod
    def _seam_abstract_hook(cls, data: Any) -> Any:
        """Abstract hook that makes this base uninstantiable."""
        ...

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class TestEvaluationSeamNeverRaises:
    """evaluate(...) returns a structured EvaluationResult for every outcome, without raising."""

    def test_single_match_reports_success(self) -> None:
        """Exactly one concrete winner: failure_kind is None and identified has one entry."""
        feature = Feature(SEAM_SINGLE_FEATURE)
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            SeamSingleFG: {SeamFwAlpha},
        }

        result = IdentifyFeatureGroupClass.evaluate(
            feature=feature,
            accessible_plugins=accessible_plugins,
            links=None,
        )

        assert isinstance(result, EvaluationResult)
        assert result.failure_kind is None
        assert len(result.identified) == 1
        assert SeamSingleFG in result.identified
        assert result.criteria_matched == {SeamSingleFG}
        assert result.abstract_matched == set()

    def test_no_match_at_all_reports_none(self) -> None:
        """A name no group matches: empty identified, failure_kind 'none', nothing matched."""
        feature = Feature(SEAM_NO_MATCH_FEATURE)
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            SeamSingleFG: {SeamFwAlpha},
        }

        result = IdentifyFeatureGroupClass.evaluate(
            feature=feature,
            accessible_plugins=accessible_plugins,
            links=None,
        )

        assert result.identified == {}
        assert result.failure_kind == "none"
        assert result.criteria_matched == set()
        assert result.abstract_matched == set()

    def test_abstract_only_match_reports_abstract_only(self) -> None:
        """Only an abstract base matched name+domain+scope: empty identified, 'abstract_only'."""
        feature = Feature(SEAM_ABSTRACT_FEATURE)
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            SeamAbstractOnlyFG: {SeamFwAlpha},
        }

        result = IdentifyFeatureGroupClass.evaluate(
            feature=feature,
            accessible_plugins=accessible_plugins,
            links=None,
        )

        assert result.identified == {}
        assert result.failure_kind == "abstract_only"
        assert result.abstract_matched
        assert SeamAbstractOnlyFG in result.abstract_matched

    def test_multiple_concrete_matches_reports_multiple(self) -> None:
        """Two unrelated siblings match the same name on distinct frameworks; both survive."""
        feature = Feature(SEAM_MULTIPLE_FEATURE)
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            SeamSiblingAFG: {SeamFwAlpha},
            SeamSiblingBFG: {SeamFwBeta},
        }

        result = IdentifyFeatureGroupClass.evaluate(
            feature=feature,
            accessible_plugins=accessible_plugins,
            links=None,
        )

        assert len(result.identified) > 1
        assert result.failure_kind == "multiple"
        assert result.criteria_matched == {SeamSiblingAFG, SeamSiblingBFG}


class TestEvaluationSeamParityWithEngineWrapper:
    """The engine wrapper still raises on the same inputs the seam evaluates without raising."""

    def test_no_match_wrapper_raises_but_evaluate_does_not(self) -> None:
        """Same no-match inputs: the constructor raises ValueError, evaluate() returns a result."""
        feature = Feature(SEAM_NO_MATCH_FEATURE)
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            SeamSingleFG: {SeamFwAlpha},
        }

        with pytest.raises(ValueError):
            IdentifyFeatureGroupClass(
                feature=feature,
                accessible_plugins=accessible_plugins,
                links=None,
                data_access_collection=None,
            )

        result = IdentifyFeatureGroupClass.evaluate(
            feature=feature,
            accessible_plugins=accessible_plugins,
            links=None,
            data_access_collection=None,
        )

        assert result.identified == {}
        assert result.failure_kind == "none"


class TestSingleFrameworkPinValidatedBeforeMatching:
    """A >1 compute-framework pin is a misuse that must be reported even when the name matches nothing (issue #851)."""

    def test_evaluate_raises_pin_error_when_no_candidate_matches(self) -> None:
        """evaluate() validates pin cardinality at entry, so a >1 pin raises even with zero name matches (#851)."""
        feature = Feature(SEAM_NO_MATCH_FEATURE)
        feature.compute_frameworks = {SeamFwAlpha, SeamFwBeta}
        accessible_plugins: FeatureGroupEnvironmentMapping = {SeamSingleFG: {SeamFwAlpha}}

        with pytest.raises(ComputeFrameworkPinError):
            IdentifyFeatureGroupClass.evaluate(
                feature=feature,
                accessible_plugins=accessible_plugins,
                links=None,
            )

    def test_engine_wrapper_raises_pin_error_when_no_candidate_matches(self) -> None:
        """The raising engine wrapper surfaces the same pin error, still an isinstance of ValueError (#851)."""
        feature = Feature(SEAM_NO_MATCH_FEATURE)
        feature.compute_frameworks = {SeamFwAlpha, SeamFwBeta}
        accessible_plugins: FeatureGroupEnvironmentMapping = {SeamSingleFG: {SeamFwAlpha}}

        with pytest.raises(ComputeFrameworkPinError) as exc_info:
            IdentifyFeatureGroupClass(
                feature=feature,
                accessible_plugins=accessible_plugins,
                links=None,
                data_access_collection=None,
            )

        assert isinstance(exc_info.value, ValueError)

    def test_resolve_feature_projects_the_pin_error(self) -> None:
        """resolve_feature never raises: it projects the pin error into ResolvedFeature.error with no candidates (#851)."""
        feature = Feature(PIN_NO_MATCH_FEATURE)
        feature.compute_frameworks = {PandasDataFrame, PythonDictFramework}

        result = resolve_feature(feature)

        assert result.feature_group is None
        assert result.candidates == []
        assert result.error is not None
        assert str(feature.name) in result.error
        assert PandasDataFrame.get_class_name() in result.error
        assert PythonDictFramework.get_class_name() in result.error

    def test_engine_and_resolve_feature_report_the_same_pin_error(self) -> None:
        """Engine and resolve_feature report identical text, mirroring the #850 parity model (unscoped: empty scope suffix)."""
        feature = Feature(PIN_NO_MATCH_FEATURE)
        feature.compute_frameworks = {PandasDataFrame, PythonDictFramework}
        accessible_plugins: FeatureGroupEnvironmentMapping = {SeamSingleFG: {SeamFwAlpha}}

        with pytest.raises(ComputeFrameworkPinError) as exc_info:
            IdentifyFeatureGroupClass(
                feature=feature,
                accessible_plugins=accessible_plugins,
                links=None,
                data_access_collection=None,
            )
        engine_message = str(exc_info.value)

        result = resolve_feature(feature)

        assert result.error == engine_message
        assert result.feature_group is None
        assert result.candidates == []
