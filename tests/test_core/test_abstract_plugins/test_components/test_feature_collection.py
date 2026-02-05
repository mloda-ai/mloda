"""Tests for domain propagation in Features class."""

import pytest

from mloda.core.abstract_plugins.components.feature_collection import Features
from mloda.core.abstract_plugins.components.feature import Feature


class TestFeaturesDomainPropagation:
    """Test suite for parent_domain propagation in Features class."""

    def test_features_propagates_domain_to_string_features(self) -> None:
        """When Features is created with string features and parent_domain,
        each feature should have the parent domain applied."""
        collection = Features(["feature1", "feature2"], parent_domain="TestDomain")

        for feature in collection:
            assert feature.domain is not None
            assert feature.domain.name == "TestDomain"

    def test_features_propagates_domain_to_feature_without_domain(self) -> None:
        """When a Feature without domain is passed to Features with parent_domain,
        the feature should inherit the parent domain."""
        input_feature = Feature("feature1")
        assert input_feature.domain is None

        collection = Features([input_feature], parent_domain="TestDomain")

        result_feature = collection.collection[0]
        assert result_feature.domain is not None
        assert result_feature.domain.name == "TestDomain"

    def test_features_preserves_explicit_domain(self) -> None:
        """When a Feature has an explicit domain, it should NOT be overwritten
        by the parent_domain."""
        input_feature = Feature("feature1", domain="ExplicitDomain")
        assert input_feature.domain is not None
        assert input_feature.domain.name == "ExplicitDomain"

        collection = Features([input_feature], parent_domain="ParentDomain")

        result_feature = collection.collection[0]
        assert result_feature.domain is not None
        assert result_feature.domain.name == "ExplicitDomain"

    def test_features_no_domain_when_parent_none(self) -> None:
        """When Features is created without parent_domain,
        string features should have no domain."""
        collection = Features(["feature1"])

        result_feature = collection.collection[0]
        assert result_feature.domain is None
