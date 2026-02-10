"""Tests for format_feature_group_classes and format_feature_group_class utility functions."""

from typing import Generator, Optional, Set, Type

import pytest

from mloda.core.abstract_plugins.components.domain import Domain
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.feature_group import (
    FeatureGroup,
    format_feature_group_class,
    format_feature_group_classes,
)
from mloda.user import Feature, Options


class SampleFeatureGroupAlpha(FeatureGroup):
    """A test feature group for formatting tests."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return None


class SampleFeatureGroupBeta(FeatureGroup):
    """Another test feature group for formatting tests."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return None


class SampleFeatureGroupWithDomain(FeatureGroup):
    """A test feature group with a custom domain."""

    @classmethod
    def get_domain(cls) -> Domain:
        return Domain("sales")

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return None


class SampleFeatureGroupFinance(FeatureGroup):
    """A test feature group with finance domain."""

    @classmethod
    def get_domain(cls) -> Domain:
        return Domain("finance")

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return None


class TestFormatFeatureGroupClassesBasic:
    """Tests for basic formatting of FeatureGroup classes."""

    def test_formats_single_class_with_name_and_module(self) -> None:
        """Test that a single FeatureGroup class is formatted with name and module.

        Expected output format:
          - SampleFeatureGroupAlpha (tests.test_core.test_abstract_plugins.test_feature_group.test_format_feature_group_classes)
        """
        result = format_feature_group_classes([SampleFeatureGroupAlpha])

        assert "SampleFeatureGroupAlpha" in result
        assert "test_format_feature_group_classes" in result
        assert result.startswith("  - ")

    def test_formats_multiple_classes_newline_separated(self) -> None:
        """Test that multiple FeatureGroup classes are formatted as newline-separated list.

        Expected output format:
          - SampleFeatureGroupAlpha (module.path)
          - SampleFeatureGroupBeta (module.path)
        """
        result = format_feature_group_classes([SampleFeatureGroupAlpha, SampleFeatureGroupBeta])

        lines = result.split("\n")
        assert len(lines) == 2

        assert "SampleFeatureGroupAlpha" in lines[0]
        assert "SampleFeatureGroupBeta" in lines[1]

        for line in lines:
            assert line.startswith("  - ")

    def test_empty_iterable_returns_empty_string(self) -> None:
        """Test that an empty iterable returns an empty string."""
        result = format_feature_group_classes([])

        assert result == ""

    def test_single_class_format_structure(self) -> None:
        """Test the exact structure of the output for a single class.

        The output should follow the pattern:
          - ClassName (module.path)
        """
        result = format_feature_group_classes([SampleFeatureGroupAlpha])

        assert result.count("(") == 1
        assert result.count(")") == 1
        assert " - " in result


class TestFormatFeatureGroupClassesWithDomain:
    """Tests for formatting FeatureGroup classes with domain information."""

    def test_include_domain_shows_domain_info(self) -> None:
        """Test that include_domain=True shows domain information.

        Expected output format:
          - SampleFeatureGroupWithDomain (module.path) [domain: sales]
        """
        result = format_feature_group_classes([SampleFeatureGroupWithDomain], include_domain=True)

        assert "SampleFeatureGroupWithDomain" in result
        assert "[domain: sales]" in result

    def test_include_domain_false_hides_domain_info(self) -> None:
        """Test that include_domain=False (default) does not show domain information."""
        result = format_feature_group_classes([SampleFeatureGroupWithDomain], include_domain=False)

        assert "SampleFeatureGroupWithDomain" in result
        assert "[domain:" not in result
        assert "sales" not in result

    def test_default_domain_shown_when_include_domain_true(self) -> None:
        """Test that default domain is shown when include_domain=True."""
        result = format_feature_group_classes([SampleFeatureGroupAlpha], include_domain=True)

        assert "SampleFeatureGroupAlpha" in result
        assert "[domain: default_domain]" in result

    def test_multiple_classes_with_different_domains(self) -> None:
        """Test formatting multiple classes with different domains.

        Expected output:
          - SampleFeatureGroupWithDomain (module.path) [domain: sales]
          - SampleFeatureGroupFinance (module.path) [domain: finance]
        """
        result = format_feature_group_classes(
            [SampleFeatureGroupWithDomain, SampleFeatureGroupFinance], include_domain=True
        )

        assert "[domain: sales]" in result
        assert "[domain: finance]" in result

        lines = result.strip().split("\n")
        assert len(lines) == 2


class TestFormatFeatureGroupClassesEdgeCases:
    """Tests for edge cases in formatting FeatureGroup classes."""

    def test_accepts_generator(self) -> None:
        """Test that the function accepts a generator (Iterable, not just list)."""

        def class_generator() -> Generator[Type[FeatureGroup], None, None]:
            yield SampleFeatureGroupAlpha
            yield SampleFeatureGroupBeta

        result = format_feature_group_classes(class_generator())

        assert "SampleFeatureGroupAlpha" in result
        assert "SampleFeatureGroupBeta" in result

    def test_accepts_set(self) -> None:
        """Test that the function accepts a set of classes."""
        classes = {SampleFeatureGroupAlpha, SampleFeatureGroupBeta}
        result = format_feature_group_classes(classes)

        assert "SampleFeatureGroupAlpha" in result
        assert "SampleFeatureGroupBeta" in result

    def test_accepts_tuple(self) -> None:
        """Test that the function accepts a tuple of classes."""
        classes = (SampleFeatureGroupAlpha,)
        result = format_feature_group_classes(classes)

        assert "SampleFeatureGroupAlpha" in result


class TestFormatFeatureGroupClass:
    """Tests for format_feature_group_class (singular) helper function."""

    def test_returns_class_name_and_module(self) -> None:
        """Test that format_feature_group_class returns ClassName (module.path) format."""
        result = format_feature_group_class(SampleFeatureGroupAlpha)

        assert "SampleFeatureGroupAlpha" in result
        assert "test_format_feature_group_classes" in result

    def test_output_format_structure(self) -> None:
        """Test the exact format structure: ClassName (module.path)."""
        result = format_feature_group_class(SampleFeatureGroupAlpha)

        assert result.count("(") == 1
        assert result.count(")") == 1
        assert result.startswith("SampleFeatureGroupAlpha")
        assert result.endswith(")")

    def test_with_feature_group_with_domain(self) -> None:
        """Test formatting works with a FeatureGroup that has a custom domain."""
        result = format_feature_group_class(SampleFeatureGroupWithDomain)

        assert "SampleFeatureGroupWithDomain" in result
        assert "test_format_feature_group_classes" in result
