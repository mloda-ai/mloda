"""
Tests for the PandasNodeCentrality implementation.
"""

from unittest.mock import Mock

import pytest
import pandas as pd

from tests.test_plugins.feature_group.experimental.core_input_check import assert_core_rejects_missing_input
from mloda.user import Feature
from mloda.provider import FeatureSet
from mloda_plugins.feature_group.experimental.node_centrality.base import NodeCentralityFeatureGroup
from mloda_plugins.feature_group.experimental.node_centrality.pandas import PandasNodeCentrality


class TestPandasNodeCentrality:
    """Tests for the PandasNodeCentrality implementation."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create a sample DataFrame for testing."""
        # Create a DataFrame with edge data
        edges = [
            {"source": "A", "target": "B", "weight": 1.0},
            {"source": "A", "target": "C", "weight": 2.0},
            {"source": "B", "target": "C", "weight": 3.0},
            {"source": "B", "target": "D", "weight": 4.0},
            {"source": "C", "target": "D", "weight": 5.0},
            {"source": "D", "target": "E", "weight": 6.0},
        ]
        return pd.DataFrame(edges)

    def _run_centrality(self, data: pd.DataFrame, feature_name: str) -> pd.DataFrame:
        """Run the group centrality loop with a stub perform and the real pandas add/columns callables."""
        feature_set = FeatureSet()
        feature_set.add(Feature(feature_name))
        feature_set.materialize_option_defaults(NodeCentralityFeatureGroup)
        scores = pd.Series(0.5, index=["A", "B", "C", "D", "E"])
        result: pd.DataFrame = NodeCentralityFeatureGroup.centrality(
            data,
            feature_set,
            perform=Mock(return_value=scores),
            add_column=PandasNodeCentrality._add_result_to_data,
        )
        return result

    def test_centrality_source_present(self, sample_data: pd.DataFrame) -> None:
        """A present source writes the result column."""
        result = self._run_centrality(sample_data, "source__degree_centrality")
        assert "source__degree_centrality" in result.columns

    def test_centrality_missing_source_raises(self, sample_data: pd.DataFrame) -> None:
        """A missing source raises while the other edge columns are present (partial presence)."""
        error = assert_core_rejects_missing_input(
            PandasNodeCentrality, sample_data, "invalid_feature__degree_centrality", ("invalid_feature",)
        )
        assert "target" in error.available_columns
        assert "weight" in error.available_columns

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("a__degree_centrality", "a"),
            ("a__b", None),
            ("centrality_result", None),
            ("x__a__degree_centrality", "x__a"),
        ],
    )
    def test_node_column_only_for_prefix_pattern(self, name: str, expected: str | None) -> None:
        """Only names matching PREFIX_PATTERN yield a node column."""
        assert NodeCentralityFeatureGroup._node_column(name) == expected

    @pytest.mark.parametrize(
        "name, mapped_column",
        [
            ("a__degree_centrality", "a"),
            ("a__b", "source"),
            ("centrality_result", "source"),
            ("x__a__degree_centrality", "x__a"),
        ],
    )
    def test_add_result_to_data_maps_through_column_only_for_prefix_pattern(
        self, name: str, mapped_column: str
    ) -> None:
        """Only a non-None node column maps through that column, else source is used."""
        data = pd.DataFrame(
            {
                "source": ["A", "B", "C", "D"],
                "target": ["B", "C", "D", "E"],
                "a": ["B", "C", "D", "E"],
                "x__a": ["C", "D", "E", "A"],
            }
        )
        scores = pd.Series([0.5, 0.3, 0.2, 0.1, 0.0], index=["A", "B", "C", "D", "E"])

        updated = PandasNodeCentrality._add_result_to_data(
            data.copy(), name, scores, NodeCentralityFeatureGroup._node_column(name)
        )

        assert updated[name].tolist() == data[mapped_column].map(scores).tolist()

    def test_create_adjacency_matrix(self, sample_data: pd.DataFrame) -> None:
        """Test the _create_adjacency_matrix method."""
        # Get unique nodes
        nodes = pd.concat([sample_data["source"], sample_data["target"]]).unique()

        # Create adjacency matrix
        adj_matrix = PandasNodeCentrality._create_adjacency_matrix(
            sample_data, nodes, "source", "target", "weight", "undirected"
        )

        # Check that the adjacency matrix has the correct shape
        assert adj_matrix.shape == (len(nodes), len(nodes))

        # Check that the adjacency matrix has the correct values
        assert adj_matrix.loc["A", "B"] == 1.0
        assert adj_matrix.loc["B", "A"] == 1.0  # Undirected
        assert adj_matrix.loc["A", "C"] == 2.0
        assert adj_matrix.loc["C", "A"] == 2.0  # Undirected

        # Create directed adjacency matrix
        adj_matrix = PandasNodeCentrality._create_adjacency_matrix(
            sample_data, nodes, "source", "target", "weight", "directed"
        )

        # Check that the adjacency matrix has the correct values for directed graph
        assert adj_matrix.loc["A", "B"] == 1.0
        assert adj_matrix.loc["B", "A"] == 0.0  # Directed

    def test_calculate_degree_centrality(self, sample_data: pd.DataFrame) -> None:
        """Test the _calculate_degree_centrality method."""
        # Get unique nodes
        nodes = pd.concat([sample_data["source"], sample_data["target"]]).unique()

        # Create adjacency matrix
        adj_matrix = PandasNodeCentrality._create_adjacency_matrix(
            sample_data, nodes, "source", "target", "weight", "undirected"
        )

        # Calculate degree centrality
        centrality = PandasNodeCentrality._calculate_degree_centrality(adj_matrix, nodes, "undirected")

        # Check that the centrality has the correct shape
        assert len(centrality) == len(nodes)

        # Check that the centrality values are non-negative
        assert (centrality >= 0).all()
        # Note: Centrality values may exceed 1 depending on the normalization method

        # Check that the node with the most connections has the highest centrality
        assert centrality["C"] > 0
        assert centrality["D"] > 0

    @pytest.mark.parametrize("algorithm", ["closeness", "betweenness", "eigenvector", "pagerank"])
    def test_calculate_bounded_centrality(self, sample_data: pd.DataFrame, algorithm: str) -> None:
        """These take (adj_matrix, nodes) and score every node within [0, 1]; degree is separate, it may exceed 1."""
        # Get unique nodes
        nodes = pd.concat([sample_data["source"], sample_data["target"]]).unique()

        # Create adjacency matrix
        adj_matrix = PandasNodeCentrality._create_adjacency_matrix(
            sample_data, nodes, "source", "target", "weight", "undirected"
        )

        calculate = getattr(PandasNodeCentrality, f"_calculate_{algorithm}_centrality")
        centrality = calculate(adj_matrix, nodes)

        # Check that the centrality has the correct shape
        assert len(centrality) == len(nodes)

        # Check that the centrality values are between 0 and 1
        assert (centrality >= 0).all()
        assert (centrality <= 1).all()

    def test_calculate_betweenness_centrality_path_graph_undirected(self) -> None:
        """Undirected path a-b-c: only unordered pair (a,c) runs through b, so C_B(b) must be 1.0, not 2.0."""
        edges = pd.DataFrame(
            [
                {"source": "a", "target": "b"},
                {"source": "b", "target": "c"},
            ]
        )
        nodes = pd.concat([edges["source"], edges["target"]]).unique()

        adj_matrix = PandasNodeCentrality._create_adjacency_matrix(edges, nodes, "source", "target", None, "undirected")

        centrality = PandasNodeCentrality._calculate_betweenness_centrality(adj_matrix, nodes)

        # b sits on the only shortest path between a and c: hand-computed C_B(b) == 1.0
        assert centrality["b"] == 1.0
        # a and c are endpoints, on no shortest path between any other pair: C_B == 0.0
        assert centrality["a"] == 0.0
        assert centrality["c"] == 0.0

    def test_calculate_centrality(self, sample_data: pd.DataFrame) -> None:
        """Test the _calculate_centrality method."""
        # Test degree centrality
        centrality = PandasNodeCentrality._calculate_centrality(sample_data, "degree", "source", "undirected", "weight")

        # Check that the centrality has the correct shape
        assert len(centrality) == len(pd.concat([sample_data["source"], sample_data["target"]]).unique())

        # Check that the centrality values are non-negative
        assert (centrality >= 0).all()
        # Note: Centrality values may exceed 1 depending on the normalization method

        # Test betweenness centrality
        centrality = PandasNodeCentrality._calculate_centrality(
            sample_data, "betweenness", "source", "undirected", "weight"
        )

        # Check that the centrality has the correct shape
        assert len(centrality) == len(pd.concat([sample_data["source"], sample_data["target"]]).unique())

        # Check that the centrality values are non-negative
        assert (centrality >= 0).all()
        # Note: Centrality values may exceed 1 depending on the normalization method

    def test_calculate_feature(self, sample_data: pd.DataFrame) -> None:
        """Test the calculate_feature method."""
        # Create a feature set
        feature_set = FeatureSet()
        feature_set.add(Feature("source__degree_centrality"))

        # Direct call bypasses run_calculate_feature, so mirror its default materialization (#796)
        feature_set.materialize_option_defaults(NodeCentralityFeatureGroup)

        # Calculate the feature
        result = PandasNodeCentrality.calculate(NodeCentralityFeatureGroup, sample_data, feature_set, None)

        # Check that the result has the expected column
        assert "source__degree_centrality" in result.columns

        # Check that the centrality values are non-negative
        assert (result["source__degree_centrality"] >= 0).all()
        # Note: Centrality values may exceed 1 depending on the normalization method

    def test_calculate_feature_multiple(self, sample_data: pd.DataFrame) -> None:
        """Test the calculate_feature method with multiple centrality features."""
        # Create a feature set
        feature_set = FeatureSet()
        features = [
            Feature("source__degree_centrality"),
            Feature("source__betweenness_centrality"),
            Feature("source__closeness_centrality"),
        ]
        for feature in features:
            feature_set.add(feature)

        # Direct call bypasses run_calculate_feature, so mirror its default materialization (#796)
        feature_set.materialize_option_defaults(NodeCentralityFeatureGroup)

        # Calculate the features
        result = PandasNodeCentrality.calculate(NodeCentralityFeatureGroup, sample_data, feature_set, None)

        # Check that the result has the expected columns
        assert "source__degree_centrality" in result.columns
        assert "source__betweenness_centrality" in result.columns
        assert "source__closeness_centrality" in result.columns

        # Check that the centrality values are non-negative
        for column in ["source__degree_centrality", "source__betweenness_centrality", "source__closeness_centrality"]:
            assert (result[column] >= 0).all()
            # Note: Centrality values may exceed 1 depending on the normalization method
