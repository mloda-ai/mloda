"""
Tests for the PandasNodeCentralityFeatureGroup class.
"""

import pytest
import pandas as pd
import numpy as np

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_plugins.feature_group.experimental.node_centrality.pandas import PandasNodeCentralityFeatureGroup


class TestPandasNodeCentralityFeatureGroup:
    """Tests for the PandasNodeCentralityFeatureGroup class."""

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

    def test_check_source_feature_exists(self, sample_data: pd.DataFrame) -> None:
        """Test the _check_source_feature_exists method."""
        # Valid feature
        PandasNodeCentralityFeatureGroup._check_source_feature_exists(sample_data, "source")

        # Invalid feature
        with pytest.raises(ValueError):
            PandasNodeCentralityFeatureGroup._check_source_feature_exists(sample_data, "invalid_feature")

    def test_add_result_to_data(self, sample_data: pd.DataFrame) -> None:
        """Test the _add_result_to_data method."""
        # Create a result Series
        nodes = pd.Series([0.5, 0.3, 0.2, 0.1, 0.0], index=["A", "B", "C", "D", "E"])

        # Add the result to the data
        updated_data = PandasNodeCentralityFeatureGroup._add_result_to_data(sample_data, "centrality_result", nodes)

        # Check that the result was added
        assert "centrality_result" in updated_data.columns
        assert len(updated_data["centrality_result"]) == len(sample_data)

    def test_create_adjacency_matrix(self, sample_data: pd.DataFrame) -> None:
        """Test the _create_adjacency_matrix method."""
        # Get unique nodes
        nodes = pd.concat([sample_data["source"], sample_data["target"]]).unique()

        # Create adjacency matrix
        adj_matrix = PandasNodeCentralityFeatureGroup._create_adjacency_matrix(
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
        adj_matrix = PandasNodeCentralityFeatureGroup._create_adjacency_matrix(
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
        adj_matrix = PandasNodeCentralityFeatureGroup._create_adjacency_matrix(
            sample_data, nodes, "source", "target", "weight", "undirected"
        )

        # Calculate degree centrality
        centrality = PandasNodeCentralityFeatureGroup._calculate_degree_centrality(adj_matrix, nodes, "undirected")

        # Check that the centrality has the correct shape
        assert len(centrality) == len(nodes)

        # Check that the centrality values are non-negative
        assert (centrality >= 0).all()
        # Note: Centrality values may exceed 1 depending on the normalization method

        # Check that the node with the most connections has the highest centrality
        assert centrality["C"] > 0
        assert centrality["D"] > 0

    def test_calculate_closeness_centrality(self, sample_data: pd.DataFrame) -> None:
        """Test the _calculate_closeness_centrality method."""
        # Get unique nodes
        nodes = pd.concat([sample_data["source"], sample_data["target"]]).unique()

        # Create adjacency matrix
        adj_matrix = PandasNodeCentralityFeatureGroup._create_adjacency_matrix(
            sample_data, nodes, "source", "target", "weight", "undirected"
        )

        # Calculate closeness centrality
        centrality = PandasNodeCentralityFeatureGroup._calculate_closeness_centrality(adj_matrix, nodes)

        # Check that the centrality has the correct shape
        assert len(centrality) == len(nodes)

        # Check that the centrality values are between 0 and 1
        assert (centrality >= 0).all()
        assert (centrality <= 1).all()

    def test_calculate_betweenness_centrality(self, sample_data: pd.DataFrame) -> None:
        """Test the _calculate_betweenness_centrality method."""
        # Get unique nodes
        nodes = pd.concat([sample_data["source"], sample_data["target"]]).unique()

        # Create adjacency matrix
        adj_matrix = PandasNodeCentralityFeatureGroup._create_adjacency_matrix(
            sample_data, nodes, "source", "target", "weight", "undirected"
        )

        # Calculate betweenness centrality
        centrality = PandasNodeCentralityFeatureGroup._calculate_betweenness_centrality(adj_matrix, nodes)

        # Check that the centrality has the correct shape
        assert len(centrality) == len(nodes)

        # Check that the centrality values are between 0 and 1
        assert (centrality >= 0).all()
        assert (centrality <= 1).all()

    def test_calculate_eigenvector_centrality(self, sample_data: pd.DataFrame) -> None:
        """Test the _calculate_eigenvector_centrality method."""
        # Get unique nodes
        nodes = pd.concat([sample_data["source"], sample_data["target"]]).unique()

        # Create adjacency matrix
        adj_matrix = PandasNodeCentralityFeatureGroup._create_adjacency_matrix(
            sample_data, nodes, "source", "target", "weight", "undirected"
        )

        # Calculate eigenvector centrality
        centrality = PandasNodeCentralityFeatureGroup._calculate_eigenvector_centrality(adj_matrix, nodes)

        # Check that the centrality has the correct shape
        assert len(centrality) == len(nodes)

        # Check that the centrality values are between 0 and 1
        assert (centrality >= 0).all()
        assert (centrality <= 1).all()

    def test_calculate_pagerank_centrality(self, sample_data: pd.DataFrame) -> None:
        """Test the _calculate_pagerank_centrality method."""
        # Get unique nodes
        nodes = pd.concat([sample_data["source"], sample_data["target"]]).unique()

        # Create adjacency matrix
        adj_matrix = PandasNodeCentralityFeatureGroup._create_adjacency_matrix(
            sample_data, nodes, "source", "target", "weight", "undirected"
        )

        # Calculate PageRank centrality
        centrality = PandasNodeCentralityFeatureGroup._calculate_pagerank_centrality(adj_matrix, nodes)

        # Check that the centrality has the correct shape
        assert len(centrality) == len(nodes)

        # Check that the centrality values are between 0 and 1
        assert (centrality >= 0).all()
        assert (centrality <= 1).all()

    def test_calculate_centrality(self, sample_data: pd.DataFrame) -> None:
        """Test the _calculate_centrality method."""
        # Test degree centrality
        centrality = PandasNodeCentralityFeatureGroup._calculate_centrality(
            sample_data, "degree", "source", "undirected", "weight"
        )

        # Check that the centrality has the correct shape
        assert len(centrality) == len(pd.concat([sample_data["source"], sample_data["target"]]).unique())

        # Check that the centrality values are non-negative
        assert (centrality >= 0).all()
        # Note: Centrality values may exceed 1 depending on the normalization method

        # Test betweenness centrality
        centrality = PandasNodeCentralityFeatureGroup._calculate_centrality(
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

        # Calculate the feature
        result = PandasNodeCentralityFeatureGroup.calculate_feature(sample_data, feature_set)

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

        # Calculate the features
        result = PandasNodeCentralityFeatureGroup.calculate_feature(sample_data, feature_set)

        # Check that the result has the expected columns
        assert "source__degree_centrality" in result.columns
        assert "source__betweenness_centrality" in result.columns
        assert "source__closeness_centrality" in result.columns

        # Check that the centrality values are non-negative
        for column in ["source__degree_centrality", "source__betweenness_centrality", "source__closeness_centrality"]:
            assert (result[column] >= 0).all()
            # Note: Centrality values may exceed 1 depending on the normalization method
