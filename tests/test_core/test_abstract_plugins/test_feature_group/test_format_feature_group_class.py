"""Tests for the format_feature_group_class (singular) helper, which stays live in
production (engine.py, execution_plan.py, feature_group_step.py) after the plural
format_feature_group_classes was removed in os-015."""

from typing import Optional

from mloda.core.abstract_plugins.components.domain import Domain
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.feature_group import FeatureGroup, format_feature_group_class
from mloda.user import Feature, Options


class SampleFeatureGroupAlpha(FeatureGroup):
    """A test feature group for formatting tests."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class SampleFeatureGroupWithDomain(FeatureGroup):
    """A test feature group with a custom domain."""

    @classmethod
    def get_domain(cls) -> Domain:
        return Domain("sales")

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class TestFormatFeatureGroupClass:
    """Direct unit tests for the format_feature_group_class (singular) helper."""

    def test_returns_class_name_and_module(self) -> None:
        """format_feature_group_class returns the ClassName (module.path) format."""
        result = format_feature_group_class(SampleFeatureGroupAlpha)

        assert "SampleFeatureGroupAlpha" in result
        assert SampleFeatureGroupAlpha.__module__ in result

    def test_output_format_structure(self) -> None:
        """The exact structure is ClassName (module.path)."""
        result = format_feature_group_class(SampleFeatureGroupAlpha)

        assert result.count("(") == 1
        assert result.count(")") == 1
        assert result.startswith("SampleFeatureGroupAlpha")
        assert result.endswith(")")

    def test_with_feature_group_with_domain(self) -> None:
        """Formatting works for a FeatureGroup that declares a custom domain."""
        result = format_feature_group_class(SampleFeatureGroupWithDomain)

        assert "SampleFeatureGroupWithDomain" in result
        assert SampleFeatureGroupWithDomain.__module__ in result
