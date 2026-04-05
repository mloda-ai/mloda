"""Tests for __eq__ methods returning False on type mismatch instead of raising exceptions.

Issue #313: FeatureGroup, FeatureName, and Domain __eq__ methods raise exceptions
when compared to incompatible types. Per Python data model conventions, __eq__
should return NotImplemented (which Python translates to False) rather than raising.
"""

from typing import Any, Optional, Set

import pytest

from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.domain import Domain
from mloda.core.abstract_plugins.components.options import Options


def create_feature_group_class(name: str, domain: Domain) -> type:
    """Factory function to create FeatureGroup subclasses with specific name and domain."""

    class DynamicFeatureGroup(FeatureGroup):
        @classmethod
        def get_domain(cls) -> Domain:
            return domain

        def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Any]]:
            return None

    DynamicFeatureGroup.__name__ = name
    DynamicFeatureGroup.__qualname__ = name
    return DynamicFeatureGroup


class TestFeatureGroupEqTypeMismatch:
    """FeatureGroup.__eq__ should return False when compared to non-FeatureGroup types."""

    @pytest.mark.parametrize(
        "other",
        [
            pytest.param(None, id="None"),
            pytest.param(42, id="int"),
            pytest.param("string", id="str"),
            pytest.param(object(), id="object"),
        ],
    )
    def test_feature_group_eq_non_feature_group_returns_false(self, other: Any) -> None:
        """Comparing a FeatureGroup to a non-FeatureGroup type should return False, not raise."""
        FGClass = create_feature_group_class("TestFG", Domain("test"))
        instance = FGClass()
        assert (instance == other) is False


class TestFeatureNameEqTypeMismatch:
    """FeatureName.__eq__ should return False when compared to non-FeatureName/non-str types."""

    @pytest.mark.parametrize(
        "other",
        [
            pytest.param(None, id="None"),
            pytest.param(42, id="int"),
            pytest.param([], id="list"),
            pytest.param(object(), id="object"),
        ],
    )
    def test_feature_name_eq_incompatible_type_returns_false(self, other: Any) -> None:
        """Comparing a FeatureName to an incompatible type should return False, not raise."""
        fn = FeatureName("test")
        assert (fn == other) is False


class TestDomainEqTypeMismatch:
    """Domain.__eq__ should return False when compared to non-Domain types."""

    @pytest.mark.parametrize(
        "other",
        [
            pytest.param(None, id="None"),
            pytest.param(42, id="int"),
            pytest.param("string", id="str"),
            pytest.param(object(), id="object"),
        ],
    )
    def test_domain_eq_non_domain_returns_false(self, other: Any) -> None:
        """Comparing a Domain to a non-Domain type should return False, not raise."""
        d = Domain("test")
        assert (d == other) is False


class TestNormalEqualityStillWorks:
    """Ensure normal same-type equality comparisons continue to work correctly."""

    def test_feature_name_equal_same_name(self) -> None:
        assert FeatureName("a") == FeatureName("a")

    def test_feature_name_not_equal_different_name(self) -> None:
        assert FeatureName("a") != FeatureName("b")

    def test_feature_name_equal_str(self) -> None:
        assert FeatureName("a") == "a"

    def test_domain_equal_same_name(self) -> None:
        assert Domain("x") == Domain("x")

    def test_domain_not_equal_different_name(self) -> None:
        assert Domain("x") != Domain("y")
