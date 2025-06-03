"""
Pandas implementation for node centrality feature groups.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, cast


try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None  # type: ignore

from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.experimental.node_centrality.base import NodeCentralityFeatureGroup


class PandasNodeCentralityFeatureGroup(NodeCentralityFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFrameWork]]:
        """Define the compute framework for this feature group."""
        return {PandasDataframe}

    @classmethod
    def _check_source_feature_exists(cls, data: pd.DataFrame, feature_name: str) -> None:
        """
        Check if the source feature exists in the DataFrame.

        Args:
            data: The pandas DataFrame
            feature_name: The name of the feature to check

        Raises:
            ValueError: If the feature does not exist in the DataFrame
        """
        if feature_name not in data.columns:
            raise ValueError(f"Feature '{feature_name}' not found in the data")

    @classmethod
    def _add_result_to_data(cls, data: pd.DataFrame, feature_name: str, result: pd.Series) -> pd.DataFrame:
        """
        Add the centrality result to the DataFrame.

        Args:
            data: The pandas DataFrame
            feature_name: The name of the feature to add
            result: The centrality result (node scores)

        Returns:
            The updated DataFrame with the centrality result added
        """
        # Create a mapping from node to centrality score
        node_to_score = result.to_dict()

        # Check if the feature name follows the expected format with a double underscore
        if "__" in feature_name:
            # Extract the node feature name from the feature name
            node_feature = feature_name.split("__")[1]

            # If the node feature is in the DataFrame, use it to map nodes to scores
            if node_feature in data.columns:
                # Map each row's node value to its centrality score
                data[feature_name] = data[node_feature].map(node_to_score)
                return data

        # If the feature name doesn't have a double underscore or the node feature is not in the DataFrame,
        # add the result as a new column using source and target columns
        source_col = "source"
        target_col = "target"

        # Create a new column with NaN values
        data[feature_name] = float("nan")

        # Fill in the centrality scores for source and target nodes
        if source_col in data.columns:
            for i, row in data.iterrows():
                source = row[source_col]
                if source in node_to_score:
                    data.at[i, feature_name] = node_to_score[source]

        # If we didn't fill all values from source column, try target column
        if target_col in data.columns and data[feature_name].isna().any():
            for i, row in data.iterrows():
                if pd.isna(data.at[i, feature_name]):  # Only fill if still NaN
                    target = row[target_col]
                    if target in node_to_score:
                        data.at[i, feature_name] = node_to_score[target]

        return data

    @classmethod
    def _calculate_centrality(
        cls,
        data: Any,
        centrality_type: str,
        node_feature: str,
        graph_type: str = "undirected",
        weight_column: Optional[str] = None,
    ) -> pd.Series:
        """
        Calculate centrality metrics for nodes in a graph.

        Args:
            data: The pandas DataFrame
            centrality_type: The type of centrality to calculate
            node_feature: The feature representing the nodes
            graph_type: The type of graph (directed or undirected)
            weight_column: The column to use for edge weights (optional)

        Returns:
            A pandas Series containing the centrality scores for each node
        """
        # Cast data to pandas DataFrame
        df = cast(pd.DataFrame, data)

        # Identify source and target columns for edge data
        # We assume the DataFrame contains edge data with source and target columns
        # The node_feature is used to identify the node column
        source_col = "source"
        target_col = "target"

        # Check if the required columns exist
        required_cols = [source_col, target_col]
        if weight_column is not None:
            required_cols.append(weight_column)

        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in the data")

        # Get unique nodes
        nodes = pd.concat([df[source_col], df[target_col]]).unique()

        # Create adjacency matrix
        adj_matrix = cls._create_adjacency_matrix(df, nodes, source_col, target_col, weight_column, graph_type)

        # Calculate centrality based on the specified type
        if centrality_type == "degree":
            centrality_scores = cls._calculate_degree_centrality(adj_matrix, nodes, graph_type)
        elif centrality_type == "betweenness":
            centrality_scores = cls._calculate_betweenness_centrality(adj_matrix, nodes)
        elif centrality_type == "closeness":
            centrality_scores = cls._calculate_closeness_centrality(adj_matrix, nodes)
        elif centrality_type == "eigenvector":
            centrality_scores = cls._calculate_eigenvector_centrality(adj_matrix, nodes)
        elif centrality_type == "pagerank":
            centrality_scores = cls._calculate_pagerank_centrality(adj_matrix, nodes)
        else:
            raise ValueError(f"Unsupported centrality type: {centrality_type}")

        return centrality_scores

    @classmethod
    def _create_adjacency_matrix(
        cls,
        df: pd.DataFrame,
        nodes: np.ndarray,  # type: ignore
        source_col: str,
        target_col: str,
        weight_column: Optional[str] = None,
        graph_type: str = "undirected",
    ) -> pd.DataFrame:
        """
        Create an adjacency matrix from edge data.

        Args:
            df: The pandas DataFrame containing edge data
            nodes: Array of unique node identifiers
            source_col: Column name for source nodes
            target_col: Column name for target nodes
            weight_column: Column name for edge weights (optional)
            graph_type: Type of graph (directed or undirected)

        Returns:
            A pandas DataFrame representing the adjacency matrix
        """
        # Create empty adjacency matrix
        adj_matrix = pd.DataFrame(0, index=nodes, columns=nodes)

        # Fill adjacency matrix with edge weights
        for _, row in df.iterrows():
            source = row[source_col]
            target = row[target_col]
            weight = 1.0
            if weight_column is not None:
                weight = row[weight_column]

            adj_matrix.at[source, target] = weight

            # For undirected graphs, make the matrix symmetric
            if graph_type == "undirected":
                adj_matrix.at[target, source] = weight

        return adj_matrix

    @classmethod
    def _calculate_degree_centrality(cls, adj_matrix: pd.DataFrame, nodes: np.ndarray, graph_type: str) -> pd.Series:  # type: ignore
        """
        Calculate degree centrality for each node.

        Args:
            adj_matrix: Adjacency matrix
            nodes: Array of unique node identifiers
            graph_type: Type of graph (directed or undirected)

        Returns:
            A pandas Series with degree centrality scores
        """
        n = len(nodes)

        if graph_type == "directed":
            # For directed graphs, calculate in-degree and out-degree
            in_degree = adj_matrix.sum(axis=0)
            out_degree = adj_matrix.sum(axis=1)

            # Total degree is the sum of in-degree and out-degree
            degree = in_degree + out_degree
        else:
            # For undirected graphs, just sum the rows (or columns, they're the same)
            degree = adj_matrix.sum(axis=1)

        # Normalize by the maximum possible degree (n-1)
        if n > 1:
            degree_centrality = degree / (n - 1)
        else:
            degree_centrality = degree

        return degree_centrality

    @classmethod
    def _calculate_closeness_centrality(cls, adj_matrix: pd.DataFrame, nodes: np.ndarray) -> pd.Series:  # type: ignore
        """
        Calculate closeness centrality for each node.

        Args:
            adj_matrix: Adjacency matrix
            nodes: Array of unique node identifiers

        Returns:
            A pandas Series with closeness centrality scores normalized between 0 and 1
        """
        n = len(nodes)

        # Convert adjacency matrix to distance matrix
        # Create a new distance matrix with float64 data type to avoid dtype warnings
        dist_matrix = pd.DataFrame(0.0, index=nodes, columns=nodes, dtype=np.float64)

        # Fill the distance matrix based on the adjacency matrix
        for i in range(n):
            for j in range(n):
                if i == j:
                    # Diagonal elements are 0
                    dist_matrix.iloc[i, j] = 0.0
                elif adj_matrix.iloc[i, j] == 0:
                    # No edge means infinite distance
                    dist_matrix.iloc[i, j] = np.inf
                else:
                    # If there's a weight, use its inverse as the distance
                    dist_matrix.iloc[i, j] = 1.0 / adj_matrix.iloc[i, j]

        # Floyd-Warshall algorithm for all-pairs shortest paths
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist_matrix.iloc[i, j] > dist_matrix.iloc[i, k] + dist_matrix.iloc[k, j]:
                        dist_matrix.iloc[i, j] = dist_matrix.iloc[i, k] + dist_matrix.iloc[k, j]

        # Calculate closeness centrality
        closeness = pd.Series(0.0, index=nodes)
        for i, node in enumerate(nodes):
            # Sum of distances to all other nodes
            sum_distances = sum(dist_matrix.iloc[i, :])

            # Handle disconnected nodes
            if sum_distances == 0 or np.isinf(sum_distances):
                closeness[node] = 0.0
            else:
                # Closeness is the inverse of the average distance
                closeness[node] = (n - 1) / sum_distances

        # Normalize to [0, 1] by dividing by the maximum value
        if closeness.max() > 0:
            closeness = closeness / closeness.max()

        return closeness

    @classmethod
    def _calculate_betweenness_centrality(cls, adj_matrix: pd.DataFrame, nodes: np.ndarray) -> pd.Series:  # type: ignore
        """
        Calculate betweenness centrality for each node.

        Args:
            adj_matrix: Adjacency matrix
            nodes: Array of unique node identifiers

        Returns:
            A pandas Series with betweenness centrality scores
        """
        n = len(nodes)

        # Initialize betweenness centrality
        betweenness = pd.Series(0.0, index=nodes)

        # Convert adjacency matrix to distance matrix
        # Create a new distance matrix with float64 data type to avoid dtype warnings
        dist_matrix = pd.DataFrame(0.0, index=nodes, columns=nodes, dtype=np.float64)

        # Fill the distance matrix based on the adjacency matrix
        for i in range(n):
            for j in range(n):
                if i == j:
                    # Diagonal elements are 0
                    dist_matrix.iloc[i, j] = 0.0
                elif adj_matrix.iloc[i, j] == 0:
                    # No edge means infinite distance
                    dist_matrix.iloc[i, j] = np.inf
                else:
                    # If there's a weight, use its inverse as the distance
                    dist_matrix.iloc[i, j] = 1.0 / adj_matrix.iloc[i, j]

        # For each pair of nodes (s,t), find all shortest paths and count how many pass through each node
        for s in range(n):
            # Run Dijkstra's algorithm from node s
            distances, predecessors = cls._dijkstra(dist_matrix, s)

            # For each target node t
            for t in range(n):
                if s == t:
                    continue

                # Count the number of shortest paths from s to t
                num_paths = cls._count_shortest_paths(predecessors, s, t)

                # For each node v that is not s or t
                for v in range(n):
                    if v == s or v == t or num_paths == 0:
                        continue

                    # Count the number of shortest paths from s to t that pass through v
                    num_paths_through_v = cls._count_shortest_paths_through(predecessors, s, t, v)

                    # Add the fraction to the betweenness centrality
                    betweenness.iloc[v] += num_paths_through_v / num_paths

        # Normalize by the maximum possible betweenness
        if n > 2:
            betweenness = betweenness / ((n - 1) * (n - 2) / 2)

        return betweenness

    @classmethod
    def _dijkstra(cls, dist_matrix: pd.DataFrame, source: int) -> tuple[np.ndarray, Dict[int, list[int]]]:  # type: ignore
        """
        Dijkstra's algorithm for single-source shortest paths.

        Args:
            dist_matrix: Distance matrix
            source: Source node index

        Returns:
            Tuple of (distances, predecessors)
        """
        n = len(dist_matrix)

        # Initialize distances and predecessors
        distances = np.full(n, np.inf)
        distances[source] = 0
        predecessors = {i: [] for i in range(n)}  # type: ignore

        # Initialize visited set
        visited = set()  # type: ignore

        # Main loop
        while len(visited) < n:
            # Find the unvisited node with the smallest distance
            min_dist = np.inf
            min_node = -1
            for i in range(n):
                if i not in visited and distances[i] < min_dist:
                    min_dist = distances[i]
                    min_node = i

            # If no unvisited node with finite distance is found, break
            if min_node == -1:
                break

            # Mark the node as visited
            visited.add(min_node)

            # Update distances to neighbors
            for neighbor in range(n):
                if neighbor not in visited:
                    weight = dist_matrix.iloc[min_node, neighbor]
                    if weight < np.inf:
                        distance = distances[min_node] + weight
                        if distance < distances[neighbor]:
                            distances[neighbor] = distance
                            predecessors[neighbor] = [min_node]
                        elif distance == distances[neighbor]:
                            predecessors[neighbor].append(min_node)

        return distances, predecessors

    @classmethod
    def _count_shortest_paths(cls, predecessors: Dict[int, list[int]], source: int, target: int) -> int:
        """
        Count the number of shortest paths from source to target.

        Args:
            predecessors: Dictionary mapping each node to its predecessors
            source: Source node index
            target: Target node index

        Returns:
            Number of shortest paths
        """
        # Base case: source is the target
        if source == target:
            return 1

        # Base case: target has no predecessors
        if not predecessors[target]:
            return 0

        # Recursive case: count paths through each predecessor
        count = 0
        for pred in predecessors[target]:
            count += cls._count_shortest_paths(predecessors, source, pred)

        return count

    @classmethod
    def _count_shortest_paths_through(
        cls, predecessors: Dict[int, list[int]], source: int, target: int, through: int
    ) -> int:
        """
        Count the number of shortest paths from source to target that pass through a specific node.

        Args:
            predecessors: Dictionary mapping each node to its predecessors
            source: Source node index
            target: Target node index
            through: Node that paths must pass through

        Returns:
            Number of shortest paths that pass through the specified node
        """
        # Count paths from source to through
        paths_to_through = cls._count_shortest_paths(predecessors, source, through)

        # Count paths from through to target
        paths_from_through = cls._count_shortest_paths(predecessors, through, target)

        # Total paths through the node is the product
        return paths_to_through * paths_from_through

    @classmethod
    def _calculate_eigenvector_centrality(
        cls,
        adj_matrix: pd.DataFrame,
        nodes: np.ndarray,  # type: ignore
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> pd.Series:
        """
        Calculate eigenvector centrality for each node using power iteration.

        Args:
            adj_matrix: Adjacency matrix
            nodes: Array of unique node identifiers
            max_iter: Maximum number of iterations
            tol: Convergence tolerance

        Returns:
            A pandas Series with eigenvector centrality scores
        """
        n = len(nodes)

        # Initialize eigenvector with equal values
        eigenvector = np.ones(n) / n

        # Convert adjacency matrix to numpy array for faster computation
        A = adj_matrix.values

        # Power iteration
        for _ in range(max_iter):
            # Compute new eigenvector
            new_eigenvector = A.dot(eigenvector)

            # Normalize
            norm = np.linalg.norm(new_eigenvector)
            if norm > 0:
                new_eigenvector = new_eigenvector / norm

            # Check convergence
            if np.linalg.norm(new_eigenvector - eigenvector) < tol:
                break

            eigenvector = new_eigenvector

        # Create Series with node labels
        return pd.Series(eigenvector, index=nodes)

    @classmethod
    def _calculate_pagerank_centrality(
        cls,
        adj_matrix: pd.DataFrame,
        nodes: np.ndarray,  # type: ignore
        alpha: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> pd.Series:
        """
        Calculate PageRank centrality for each node.

        Args:
            adj_matrix: Adjacency matrix
            nodes: Array of unique node identifiers
            alpha: Damping factor (typically 0.85)
            max_iter: Maximum number of iterations
            tol: Convergence tolerance

        Returns:
            A pandas Series with PageRank centrality scores
        """
        n = len(nodes)

        # Initialize PageRank with equal values
        pagerank = np.ones(n) / n

        # Convert adjacency matrix to numpy array for faster computation
        A = adj_matrix.values

        # Normalize the adjacency matrix by out-degree
        out_degrees = A.sum(axis=1)

        # Handle nodes with no outgoing edges
        out_degrees[out_degrees == 0] = 1.0

        # Create the transition matrix
        M = A / out_degrees[:, np.newaxis]

        # Power iteration
        for _ in range(max_iter):
            # Compute new PageRank
            new_pagerank = alpha * M.T.dot(pagerank) + (1 - alpha) / n

            # Normalize
            new_pagerank = new_pagerank / new_pagerank.sum()

            # Check convergence
            if np.linalg.norm(new_pagerank - pagerank) < tol:
                break

            pagerank = new_pagerank

        # Create Series with node labels
        return pd.Series(pagerank, index=nodes)
