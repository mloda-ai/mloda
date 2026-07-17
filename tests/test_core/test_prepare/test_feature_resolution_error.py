"""Failing tests for the typed resolution exception ``FeatureResolutionError`` (issue #809).

The raising seam in ``IdentifyFeatureGroupClass.__init__`` raises a bare ``ValueError`` when a feature
cannot be resolved. This suite pins its replacement: ``FeatureResolutionError(ValueError)``, defined in
``mloda.core.prepare.identify_feature_group`` beside the evaluation seam. It carries ``feature_name: str``
and ``result: EvaluationResult``, keeps the rendered message text unchanged (the message still comes from
``render_resolution_failure``), and is re-exported from ``mloda.user`` and ``mloda.steward``.

All fixture names carry an ``_809`` suffix: test feature groups become global subclasses and the suite
runs in parallel, so a shared name would leak into another module's candidate universe.
"""

from typing import Optional

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import (
    FeatureResolutionError,
    IdentifyFeatureGroupClass,
    render_resolution_failure,
)
from mloda.user import mlodaAPI


# Unique feature names so no other double in the parallel suite collides on them.
KNOWN_FEATURE_809 = "resolution_error_known_809"
NO_MATCH_FEATURE_809 = "resolution_error_no_match_809"
MULTIPLE_FEATURE_809 = "resolution_error_multiple_809"


class ResolutionErrorFw_809(ComputeFramework):
    """Dummy compute framework for the typed-resolution-error tests."""


class ResolutionErrorKnownFG_809(FeatureGroup):
    """Concrete feature group matching exactly one unique feature name."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == KNOWN_FEATURE_809

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class ResolutionErrorSiblingAFG_809(FeatureGroup):
    """First of two unrelated siblings matching the same name, forcing an ambiguous match."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == MULTIPLE_FEATURE_809

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class ResolutionErrorSiblingBFG_809(FeatureGroup):
    """Second unrelated sibling matching the same name, forcing an ambiguous match."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == MULTIPLE_FEATURE_809

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


def _no_match_accessible_plugins() -> FeatureGroupEnvironmentMapping:
    """One candidate that matches only its own known name: any other name yields failure_kind 'none'."""
    return {ResolutionErrorKnownFG_809: {ResolutionErrorFw_809}}


class TestFeatureResolutionErrorAtTheSeam:
    """The raising seam raises the typed error, carrying the evaluation it just ran."""

    def test_unresolvable_feature_raises_the_typed_error(self) -> None:
        """An unresolvable feature raises FeatureResolutionError, which is a ValueError."""
        feature = Feature(NO_MATCH_FEATURE_809)

        with pytest.raises(FeatureResolutionError) as exc_info:
            IdentifyFeatureGroupClass(
                feature=feature,
                accessible_plugins=_no_match_accessible_plugins(),
                links=None,
                data_access_collection=None,
            )

        assert isinstance(exc_info.value, ValueError)
        assert exc_info.value.feature_name == NO_MATCH_FEATURE_809

    def test_the_error_carries_the_evaluation_result(self) -> None:
        """The raised error exposes the structured EvaluationResult of its single pass."""
        feature = Feature(NO_MATCH_FEATURE_809)

        with pytest.raises(FeatureResolutionError) as exc_info:
            IdentifyFeatureGroupClass(
                feature=feature,
                accessible_plugins=_no_match_accessible_plugins(),
                links=None,
                data_access_collection=None,
            )

        result = exc_info.value.result
        assert result.failure_kind == "none"
        assert isinstance(result.criteria_matched, set)
        assert isinstance(result.candidate_frameworks, dict)

    def test_the_message_text_is_unchanged(self) -> None:
        """str(error) is exactly render_resolution_failure over the same evaluation."""
        feature = Feature(NO_MATCH_FEATURE_809)

        with pytest.raises(FeatureResolutionError) as exc_info:
            IdentifyFeatureGroupClass(
                feature=feature,
                accessible_plugins=_no_match_accessible_plugins(),
                links=None,
                data_access_collection=None,
            )

        result = IdentifyFeatureGroupClass.evaluate(
            feature=feature,
            accessible_plugins=_no_match_accessible_plugins(),
            links=None,
        )
        expected = render_resolution_failure(result, feature)
        assert expected is not None
        assert str(exc_info.value) == expected

    def test_ambiguous_match_reports_multiple_with_its_candidates(self) -> None:
        """An ambiguous match carries failure_kind 'multiple' and a non-empty criteria_matched."""
        feature = Feature(MULTIPLE_FEATURE_809)
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            ResolutionErrorSiblingAFG_809: {ResolutionErrorFw_809},
            ResolutionErrorSiblingBFG_809: {ResolutionErrorFw_809},
        }

        with pytest.raises(FeatureResolutionError) as exc_info:
            IdentifyFeatureGroupClass(
                feature=feature,
                accessible_plugins=accessible_plugins,
                links=None,
                data_access_collection=None,
            )

        assert exc_info.value.feature_name == MULTIPLE_FEATURE_809
        result = exc_info.value.result
        assert result.failure_kind == "multiple"
        assert result.criteria_matched


class TestFeatureResolutionErrorEndToEnd:
    """The engine path surfaces the typed error unchanged."""

    def test_run_all_raises_the_typed_error_for_an_unresolvable_name(self) -> None:
        """mlodaAPI.run_all with an unresolvable feature name raises FeatureResolutionError."""
        with pytest.raises(FeatureResolutionError):
            mlodaAPI.run_all(
                [NO_MATCH_FEATURE_809],
                compute_frameworks={ResolutionErrorFw_809},
                plugin_collector=PluginCollector.enabled_feature_groups({ResolutionErrorKnownFG_809}),
            )


class TestFeatureResolutionErrorImportSurfaces:
    """The typed error is re-exported from the user and steward surfaces."""

    def test_user_surface_reexports_the_same_class(self) -> None:
        """mloda.user exposes the exact class the core module defines."""
        from mloda.user import FeatureResolutionError as user_error

        assert user_error is FeatureResolutionError

    def test_steward_surface_reexports_the_same_class(self) -> None:
        """mloda.steward exposes the exact class the core module defines."""
        from mloda.steward import FeatureResolutionError as steward_error

        assert steward_error is FeatureResolutionError
