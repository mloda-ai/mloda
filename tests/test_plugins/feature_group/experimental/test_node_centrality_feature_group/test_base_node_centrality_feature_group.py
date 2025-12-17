"""
Tests for the base NodeCentralityFeatureGroup class.
"""

import pytest

from mloda import Feature
from mloda.user import FeatureName
from mloda import Options
from mloda_plugins.feature_group.experimental.node_centrality.base import NodeCentralityFeatureGroup


class TestNodeCentralityFeatureGroup:
    """Tests for the NodeCentralityFeatureGroup class."""

    def test_match_feature_group_criteria(self) -> None:
        """Test the match_feature_group_criteria method."""
        # Valid feature names
        assert NodeCentralityFeatureGroup.match_feature_group_criteria("user__degree_centrality", Options())
        assert NodeCentralityFeatureGroup.match_feature_group_criteria("product__betweenness_centrality", Options())
        assert NodeCentralityFeatureGroup.match_feature_group_criteria("website__closeness_centrality", Options())
        assert NodeCentralityFeatureGroup.match_feature_group_criteria("node__eigenvector_centrality", Options())
        assert NodeCentralityFeatureGroup.match_feature_group_criteria("page__pagerank_centrality", Options())

        # Invalid feature names
        assert not NodeCentralityFeatureGroup.match_feature_group_criteria("centrality_degree__user", Options())

    def test_parse_centrality_prefix(self) -> None:
        """Test the parse_centrality_prefix method."""
        # Valid feature names
        centrality_type = NodeCentralityFeatureGroup.parse_centrality_prefix("user__degree_centrality")
        assert centrality_type == "degree"

        centrality_type = NodeCentralityFeatureGroup.parse_centrality_prefix("product__betweenness_centrality")
        assert centrality_type == "betweenness"

        # Invalid feature names
        with pytest.raises(ValueError):
            NodeCentralityFeatureGroup.parse_centrality_prefix("centrality_degree__user")

        with pytest.raises(ValueError):
            NodeCentralityFeatureGroup.parse_centrality_prefix("invalid_centrality__product")

        with pytest.raises(ValueError):
            NodeCentralityFeatureGroup.parse_centrality_prefix("degree_invalid__website")

        with pytest.raises(ValueError):
            NodeCentralityFeatureGroup.parse_centrality_prefix("degree_centrality_website")

    def test_get_centrality_type(self) -> None:
        """Test the get_centrality_type method."""
        assert NodeCentralityFeatureGroup.get_centrality_type("user__degree_centrality") == "degree"
        assert NodeCentralityFeatureGroup.get_centrality_type("product__betweenness_centrality") == "betweenness"
        assert NodeCentralityFeatureGroup.get_centrality_type("website__closeness_centrality") == "closeness"
        assert NodeCentralityFeatureGroup.get_centrality_type("node__eigenvector_centrality") == "eigenvector"
        assert NodeCentralityFeatureGroup.get_centrality_type("page__pagerank_centrality") == "pagerank"

    def test_input_features(self) -> None:
        """Test the input_features method."""
        feature_group = NodeCentralityFeatureGroup()

        # Test with different centrality types
        input_features = feature_group.input_features(Options(), FeatureName("user__degree_centrality"))
        assert input_features is not None
        assert len(input_features) == 1
        assert Feature("user") in input_features

        input_features = feature_group.input_features(Options(), FeatureName("product__betweenness_centrality"))
        assert input_features is not None
        assert len(input_features) == 1
        assert Feature("product") in input_features
