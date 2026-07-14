"""A pipeline_name outside PIPELINE_TYPES is a non-match on both paths, not an exception.

``SklearnPipelineFeatureGroup`` overrides the match hook and calls the parser directly, so it is the case where
a rejection could escape the engine. The user gets "No feature groups found" with the reason instead.
"""

from __future__ import annotations

import pytest

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.provider import DefaultOptionKeys
from mloda.user import Options
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.experimental.sklearn.pipeline.base import SklearnPipelineFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.pipeline.pandas import PandasSklearnPipelineFeatureGroup


# "custom" is deliberately NOT a member of PIPELINE_TYPES.
BOGUS_PIPELINE_NAME = "custom"
BOGUS_FEATURE = "income__sklearn_pipeline_custom"
VALID_FEATURE = "income__sklearn_pipeline_scaling"


class TestBogusPipelineNameIsANonMatch:
    def test_pipeline_types_does_not_contain_the_bogus_value(self) -> None:
        """Precondition: the rejected value really is outside the declared value space."""
        assert BOGUS_PIPELINE_NAME not in SklearnPipelineFeatureGroup.PIPELINE_TYPES

    def test_name_path_bogus_pipeline_name_returns_false(self) -> None:
        """A string-named pipeline feature carrying a bogus pipeline_name does not match."""
        options = Options(context={SklearnPipelineFeatureGroup.PIPELINE_NAME: BOGUS_PIPELINE_NAME})

        result = SklearnPipelineFeatureGroup.match_feature_group_criteria(BOGUS_FEATURE, options)

        assert result is False

    def test_config_path_bogus_pipeline_name_returns_false(self) -> None:
        """The config-based path reaches the same verdict, and does not raise either."""
        options = Options(
            context={
                SklearnPipelineFeatureGroup.PIPELINE_NAME: BOGUS_PIPELINE_NAME,
                DefaultOptionKeys.in_features: "income",
            }
        )

        result = SklearnPipelineFeatureGroup.match_feature_group_criteria("placeholder", options)

        assert result is False

    def test_rejection_reason_names_key_and_value(self) -> None:
        """The discarded message stays actionable."""
        options = Options(context={SklearnPipelineFeatureGroup.PIPELINE_NAME: BOGUS_PIPELINE_NAME})

        reason = SklearnPipelineFeatureGroup._strict_validation_rejection_reason(BOGUS_FEATURE, options)

        assert reason is not None
        assert SklearnPipelineFeatureGroup.PIPELINE_NAME in reason
        assert BOGUS_PIPELINE_NAME in reason


class TestValidPipelineNameStillMatches:
    """Guard against over-rejecting: only bogus values are rejected."""

    def test_valid_pipeline_name_on_name_path_matches(self) -> None:
        options = Options(context={SklearnPipelineFeatureGroup.PIPELINE_NAME: "scaling"})

        assert SklearnPipelineFeatureGroup.match_feature_group_criteria(VALID_FEATURE, options) is True

    def test_name_path_without_options_matches(self) -> None:
        """The name carries the pipeline; no options at all is still a match."""
        assert SklearnPipelineFeatureGroup.match_feature_group_criteria(VALID_FEATURE, Options()) is True

    def test_config_path_with_valid_pipeline_name_matches(self) -> None:
        options = Options(
            context={
                SklearnPipelineFeatureGroup.PIPELINE_NAME: "scaling",
                DefaultOptionKeys.in_features: "income",
            }
        )

        assert SklearnPipelineFeatureGroup.match_feature_group_criteria("placeholder", options) is True


class TestBogusPipelineNameAtEngineLevel:
    """The rejection must not travel out of feature identification."""

    def test_engine_reports_no_feature_groups_found_with_the_reason(self) -> None:
        feature = Feature(BOGUS_FEATURE, Options(context={SklearnPipelineFeatureGroup.PIPELINE_NAME: "custom"}))
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            PandasSklearnPipelineFeatureGroup: {PandasDataFrame},
        }

        with pytest.raises(ValueError) as exc_info:
            IdentifyFeatureGroupClass(
                feature=feature,
                accessible_plugins=accessible_plugins,
                links=None,
                data_access_collection=None,
            )

        message = str(exc_info.value)
        assert "No feature groups found" in message, (
            f"A bogus pipeline_name must be a non-match with mloda's standard error, "
            f"not a bare ValueError out of the engine, but got: {message}"
        )
        assert SklearnPipelineFeatureGroup.PIPELINE_NAME in message
        assert BOGUS_PIPELINE_NAME in message
