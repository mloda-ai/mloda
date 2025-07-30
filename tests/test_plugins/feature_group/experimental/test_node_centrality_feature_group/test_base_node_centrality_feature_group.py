"""
Tests for the base NodeCentralityFeatureGroup class.
"""

import pytest

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.options import Options
from mloda_plugins.feature_group.experimental.node_centrality.base import NodeCentralityFeatureGroup
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


class TestNodeCentralityFeatureGroup:
    """Tests for the NodeCentralityFeatureGroup class."""

    def test_match_feature_group_criteria(self) -> None:
        """Test the match_feature_group_criteria method."""
        # Valid feature names
        assert NodeCentralityFeatureGroup.match_feature_group_criteria("degree_centrality__user", Options())
        assert NodeCentralityFeatureGroup.match_feature_group_criteria("betweenness_centrality__product", Options())
        assert NodeCentralityFeatureGroup.match_feature_group_criteria("closeness_centrality__website", Options())
        assert NodeCentralityFeatureGroup.match_feature_group_criteria("eigenvector_centrality__node", Options())
        assert NodeCentralityFeatureGroup.match_feature_group_criteria("pagerank_centrality__page", Options())

        # Invalid feature names
        assert not NodeCentralityFeatureGroup.match_feature_group_criteria("centrality_degree__user", Options())

    def test_parse_centrality_prefix(self) -> None:
        """Test the parse_centrality_prefix method."""
        # Valid feature names
        centrality_type = NodeCentralityFeatureGroup.parse_centrality_prefix("degree_centrality__user")
        assert centrality_type == "degree"

        centrality_type = NodeCentralityFeatureGroup.parse_centrality_prefix("betweenness_centrality__product")
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
        assert NodeCentralityFeatureGroup.get_centrality_type("degree_centrality__user") == "degree"
        assert NodeCentralityFeatureGroup.get_centrality_type("betweenness_centrality__product") == "betweenness"
        assert NodeCentralityFeatureGroup.get_centrality_type("closeness_centrality__website") == "closeness"
        assert NodeCentralityFeatureGroup.get_centrality_type("eigenvector_centrality__node") == "eigenvector"
        assert NodeCentralityFeatureGroup.get_centrality_type("pagerank_centrality__page") == "pagerank"

    def test_input_features(self) -> None:
        """Test the input_features method."""
        feature_group = NodeCentralityFeatureGroup()

        # Test with different centrality types
        input_features = feature_group.input_features(Options(), FeatureName("degree_centrality__user"))
        assert input_features is not None
        assert len(input_features) == 1
        assert Feature("user") in input_features

        input_features = feature_group.input_features(Options(), FeatureName("betweenness_centrality__product"))
        assert input_features is not None
        assert len(input_features) == 1
        assert Feature("product") in input_features
