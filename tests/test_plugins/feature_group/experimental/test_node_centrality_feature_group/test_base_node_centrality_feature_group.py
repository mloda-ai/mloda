"""
Tests for the base NodeCentralityFeatureGroup class.
"""

from typing import Any

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
        assert {str(f.name) for f in input_features} == {"user", "source", "target"}
        assert Feature("user") in input_features

        input_features = feature_group.input_features(Options(), FeatureName("product__betweenness_centrality"))
        assert input_features is not None
        assert {str(f.name) for f in input_features} == {"product", "source", "target"}
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
        """Parts are read from PREFIX_PATTERN."""

        class CentrPatternGroup(NodeCentralityFeatureGroup):
            PREFIX_PATTERN = r".*__(?P<centrality_type>[\w]+)_centr$"

        assert CentrPatternGroup.parse_centrality_prefix(name) == expected
        with pytest.raises(ValueError):
            CentrPatternGroup.parse_centrality_prefix("x__degree_centrality")

    @pytest.mark.parametrize(
        ("context", "expected"),
        [
            ({}, {"user", "source", "target"}),
            ({"weight_column": "w"}, {"user", "source", "target", "w"}),
            ({"weight_column": None}, {"user", "source", "target"}),
            ({"weight_column": "source"}, {"user", "source", "target"}),
        ],
    )
    def test_input_features_declare_edge_columns(self, context: dict[str, Any], expected: set[str]) -> None:
        result = PandasNodeCentralityFeatureGroup().input_features(
            Options(context=context), FeatureName("user__degree_centrality")
        )
        assert result is not None
        assert {str(f.name) for f in result} == expected
        assert len(result) == len(expected)

    @pytest.mark.parametrize(
        ("source", "expected_group"),
        [
            pytest.param(Feature("user", Options(group={"scope": "a"})), {}, id="scoped_source"),
            pytest.param(
                Feature("user", Options(group={"aggregation_type": "sum", DefaultOptionKeys.in_features: "raw_user"})),
                {"aggregation_type": "sum"},
                id="chained_source",
            ),
        ],
    )
    def test_edge_columns_carry_only_forwarded_group_options_of_source(
        self, source: Feature, expected_group: dict[str, Any]
    ) -> None:
        options = Options(context={DefaultOptionKeys.in_features: frozenset([source]), "weight_column": "w"})
        result = PandasNodeCentralityFeatureGroup().input_features(options, FeatureName("user__degree_centrality"))
        assert result is not None
        by_name = {str(f.name): f for f in result}
        assert set(by_name) == {"user", "source", "target", "w"}
        for name in ("source", "target", "w"):
            assert dict(by_name[name].options.group) == expected_group
            assert DefaultOptionKeys.in_features not in by_name[name].options.group
            if not expected_group:
                assert by_name[name] == Feature(name)
