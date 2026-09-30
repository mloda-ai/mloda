"""
Tests for the base NodeCentralityFeatureGroup class.
"""

import pytest

from mloda.core.abstract_plugins.components.match_rejection import MatchRejection
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Options
from mloda.provider import DefaultOptionKeys
from mloda_plugins.feature_group.experimental.node_centrality.base import NodeCentralityFeatureGroup
from mloda_plugins.feature_group.experimental.node_centrality.pandas import PandasNodeCentralityFeatureGroup


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

    def test_match_feature_group_criteria_configuration_without_weight_column(self) -> None:
        """Configuration-based matching must succeed when the optional weight_column is omitted."""
        options = Options(
            context={
                NodeCentralityFeatureGroup.CENTRALITY_TYPE: "degree",
                NodeCentralityFeatureGroup.GRAPH_TYPE: "undirected",
                DefaultOptionKeys.in_features: "source",
            }
        )

        assert NodeCentralityFeatureGroup.match_feature_group_criteria("placeholder", options)

    def test_graph_type_value_in_the_centrality_slot_is_rejected(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        options = Options(context={NodeCentralityFeatureGroup.CENTRALITY_TYPE: "degree"})
        assert NodeCentralityFeatureGroup.match_feature_group_criteria("x__directed_centrality", options) is False
        assert len(rejection_window) == 1
        assert "directed" in next(iter(rejection_window.values())).reason

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
        feature_group = PandasNodeCentralityFeatureGroup()

        # Test with different centrality types
        input_features = feature_group.input_features(Options(), FeatureName("user__degree_centrality"))
        assert input_features is not None
        assert len(input_features) == 1
        assert Feature("user") in input_features

        input_features = feature_group.input_features(Options(), FeatureName("product__betweenness_centrality"))
        assert input_features is not None
        assert len(input_features) == 1
        assert Feature("product") in input_features

    @pytest.mark.parametrize(
        "name",
        ["x__mean_imputed__degree_centrality", "a__b__c__degree_centrality"],
    )
    def test_chained_source_name_parses_from_the_last_suffix(self, name: str) -> None:
        assert NodeCentralityFeatureGroup.parse_centrality_prefix(name) == "degree"

    @pytest.mark.parametrize(
        "name",
        [
            "a__foo_bar_centrality",
            "a__degree_extra_centrality",
            "a__degree_centrality_extra",
            "x__mean_imputed__foo_bar_centrality",
            "__degree_centrality",
        ],
    )
    def test_parse_centrality_prefix_rejects_underscores_inside_the_type(self, name: str) -> None:
        with pytest.raises(ValueError):
            NodeCentralityFeatureGroup.parse_centrality_prefix(name)

    @pytest.mark.parametrize(
        ("name", "expected"),
        [("x__degree_centr", "degree"), ("x__mean_imputed__pagerank_centr", "pagerank")],
    )
    def test_parse_centrality_prefix_follows_an_overridden_prefix_pattern(self, name: str, expected: str) -> None:
        """The parts come from PREFIX_PATTERN, not from a hand-written copy of the grammar."""

        class CentrPatternGroup(NodeCentralityFeatureGroup):
            PREFIX_PATTERN = r".*__(?P<centrality_type>[\w]+)_centr$"

        assert CentrPatternGroup.parse_centrality_prefix(name) == expected
        with pytest.raises(ValueError):
            CentrPatternGroup.parse_centrality_prefix("x__degree_centrality")
