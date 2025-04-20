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
        assert not NodeCentralityFeatureGroup.match_feature_group_criteria("invalid_centrality__product", Options())
        assert not NodeCentralityFeatureGroup.match_feature_group_criteria("degree_invalid__website", Options())
        assert not NodeCentralityFeatureGroup.match_feature_group_criteria("degree_centrality_website", Options())

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

    def test_feature_chain_parser_configuration(self) -> None:
        """Test the configurable_feature_chain_parser method."""
        parser_config = NodeCentralityFeatureGroup.configurable_feature_chain_parser()
        assert parser_config is not None

    def test_feature_chain_parser_integration(self) -> None:
        """Test integration with FeatureChainParser."""
        # Create a feature with options
        feature = Feature(
            "placeholder",
            Options(
                {
                    NodeCentralityFeatureGroup.CENTRALITY_TYPE: "degree",
                    DefaultOptionKeys.mloda_source_feature: "user",
                }
            ),
        )

        # Parse the feature using the parser configuration
        parser_config = NodeCentralityFeatureGroup.configurable_feature_chain_parser()
        assert parser_config is not None

        # Create a feature without options
        parsed_feature = parser_config.create_feature_without_options(feature)
        assert parsed_feature is not None
        assert parsed_feature.name.name == "degree_centrality__user"

        # Check that the options were removed
        assert NodeCentralityFeatureGroup.CENTRALITY_TYPE not in parsed_feature.options.data
        assert DefaultOptionKeys.mloda_source_feature not in parsed_feature.options.data

    def test_parse_from_options(self) -> None:
        """Test the parse_from_options method of the configurable feature chain parser."""
        parser_config = NodeCentralityFeatureGroup.configurable_feature_chain_parser()
        assert parser_config is not None

        # Valid options
        options = Options(
            {
                NodeCentralityFeatureGroup.CENTRALITY_TYPE: "degree",
                DefaultOptionKeys.mloda_source_feature: "user",
            }
        )
        feature_name = parser_config.parse_from_options(options)
        assert feature_name == "degree_centrality__user"

        # Different centrality type
        options = Options(
            {
                NodeCentralityFeatureGroup.CENTRALITY_TYPE: "betweenness",
                DefaultOptionKeys.mloda_source_feature: "product",
            }
        )
        feature_name = parser_config.parse_from_options(options)
        assert feature_name == "betweenness_centrality__product"

        # Missing options
        options = Options(
            {
                # Missing CENTRALITY_TYPE
                DefaultOptionKeys.mloda_source_feature: "user",
            }
        )
        feature_name = parser_config.parse_from_options(options)
        assert feature_name is None

        # Invalid centrality type
        options = Options(
            {
                NodeCentralityFeatureGroup.CENTRALITY_TYPE: "invalid",
                DefaultOptionKeys.mloda_source_feature: "user",
            }
        )
        with pytest.raises(ValueError):
            parser_config.parse_from_options(options)
