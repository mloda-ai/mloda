"""
Integration tests for the DimensionalityReductionFeatureGroup.
"""

from typing import Any, Dict, List

import numpy as np

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI

from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.experimental.dimensionality_reduction.base import DimensionalityReductionFeatureGroup
from mloda_plugins.feature_group.experimental.dimensionality_reduction.pandas import (
    PandasDimensionalityReductionFeatureGroup,
)
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator


class DimensionalityReductionFeatureTestDataCreator(ATestDataCreator):
    """Base class for dimensionality reduction feature test data creators."""

    compute_framework = PandasDataFrame

    @classmethod
    def get_raw_data(cls) -> Dict[str, Any]:
        """Return the raw data as a dictionary with features for dimensionality reduction."""
        # Create data with some structure for dimensionality reduction
        np.random.seed(42)
        # Use minimal samples for unit tests - only need to verify functionality works
        n_samples = 10
        n_features = 10

        # Create data with some structure
        data = np.random.randn(n_samples, n_features)

        # Add some structure (correlations between features)
        data[:, 1] = data[:, 0] * 0.8 + np.random.randn(n_samples) * 0.2
        data[:, 2] = data[:, 0] * 0.6 + np.random.randn(n_samples) * 0.4

        # Create column names
        columns = [f"feature{i}" for i in range(n_features)]

        # Add a categorical column for LDA
        categories = ["A", "B", "C"]
        category_column = np.random.choice(categories, size=n_samples)

        # Create a dictionary using the column names
        result = {column: data[:, i] for i, column in enumerate(columns)}
        result["category"] = category_column

        return result


def validate_dimensionality_reduction_results(result: List) -> None:  # type: ignore
    """
    Validate the results of the dimensionality reduction feature test.

    Args:
        result: List of DataFrames from the mlodaAPI.run_all call

    Raises:
        AssertionError: If validation fails
    """
    # Verify we have at least one result
    assert len(result) >= 1, "Expected at least one result"

    # Convert all results to pandas DataFrames for consistent validation
    dfs = []
    for res in result:
        if hasattr(res, "to_pandas"):
            dfs.append(res.to_pandas())
        else:
            dfs.append(res)

    # Find the DataFrame with the dimensionality reduction features
    pca_feature = "feature0,feature1,feature2__pca_2d"
    tsne_feature = "feature0,feature1,feature2__tsne_2d"

    # Check that all features exist in the results
    result_df = None

    for df in dfs:
        # If the DataFrame contains any of our dimensionality reduction features, use it
        if any(col.startswith(pca_feature) or col.startswith(tsne_feature) for col in df.columns):
            result_df = df
            break

    # Verify that the result DataFrame was found
    assert result_df is not None, "DataFrame with dimensionality reduction features not found"

    # Verify the dimensionality reduction results
    # With the multiple result columns pattern, we check for the dimension columns
    assert f"{pca_feature}~dim1" in result_df.columns, f"{pca_feature}~dim1 not found"
    assert f"{pca_feature}~dim2" in result_df.columns, f"{pca_feature}~dim2 not found"
    assert f"{tsne_feature}~dim1" in result_df.columns, f"{tsne_feature}~dim1 not found"
    assert f"{tsne_feature}~dim2" in result_df.columns, f"{tsne_feature}~dim2 not found"


class TestDimensionalityReductionFeatureGroupIntegration:
    """Integration tests for the DimensionalityReductionFeatureGroup."""

    def test_integration_with_feature_names_dimension(self) -> None:
        """Test integration with mlodaAPI using explicit feature names."""
        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {DimensionalityReductionFeatureTestDataCreator, PandasDimensionalityReductionFeatureGroup}
        )

        # Define the features
        features: List[Feature | str] = [
            "feature0",
            "feature1",
            "feature2",
            "feature0,feature1,feature2__pca_2d",
            "feature0,feature1,feature2__tsne_2d",
        ]

        # Run the API
        result = mlodaAPI.run_all(features, compute_frameworks={PandasDataFrame}, plugin_collector=plugin_collector)

        # Validate the results
        validate_dimensionality_reduction_results(result)

    def atest_integration_with_feature_parser(self) -> None:
        """Test integration with mlodaAPI using the parser."""
        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {DimensionalityReductionFeatureTestDataCreator, PandasDimensionalityReductionFeatureGroup}
        )

        # Create features using the parser configuration
        pca_feature = Feature(
            "placeholder",
            Options(
                {
                    DimensionalityReductionFeatureGroup.ALGORITHM: "pca",
                    DimensionalityReductionFeatureGroup.DIMENSION: 2,
                    DefaultOptionKeys.in_features: "feature0,feature1,feature2",
                }
            ),
        )

        tsne_feature = Feature(
            "placeholder",
            Options(
                {
                    DimensionalityReductionFeatureGroup.ALGORITHM: "tsne",
                    DimensionalityReductionFeatureGroup.DIMENSION: 2,
                    DefaultOptionKeys.in_features: "feature0,feature1,feature2",
                }
            ),
        )

        # Define the features
        features: List[str | Feature] = ["feature0", "feature1", "feature2", pca_feature, tsne_feature]

        # Run the API
        result = mlodaAPI.run_all(features, compute_frameworks={PandasDataFrame}, plugin_collector=plugin_collector)

        # Validate the results
        validate_dimensionality_reduction_results(result)

    def test_integration_with_different_algorithms(self) -> None:
        """Test integration with mlodaAPI using different dimensionality reduction algorithms."""
        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {DimensionalityReductionFeatureTestDataCreator, PandasDimensionalityReductionFeatureGroup}
        )

        # Define the features
        features: List[str | Feature] = [
            "feature0",
            "feature1",
            "feature2",
            "feature0,feature1,feature2__pca_2d",
            "feature0,feature1,feature2__ica_2d",
            "feature0,feature1,feature2__isomap_2d",
        ]

        # Run the API
        result = mlodaAPI.run_all(features, compute_frameworks={PandasDataFrame}, plugin_collector=plugin_collector)

        # Verify we have at least one result
        assert len(result) >= 1, "Expected at least one result"

        # Convert all results to pandas DataFrames for consistent validation
        dfs = []
        for res in result:
            if hasattr(res, "to_pandas"):
                dfs.append(res.to_pandas())
            else:
                dfs.append(res)

        # Find the DataFrame with the dimensionality reduction features
        pca_feature = "feature0,feature1,feature2__pca_2d"
        ica_feature = "feature0,feature1,feature2__ica_2d"
        isomap_feature = "feature0,feature1,feature2__isomap_2d"

        # Check that all features exist in the results
        result_df = None

        for df in dfs:
            # If the DataFrame contains any of our dimensionality reduction features, use it
            if any(
                col.startswith(pca_feature) or col.startswith(ica_feature) or col.startswith(isomap_feature)
                for col in df.columns
            ):
                result_df = df
                break

        # Verify that the result DataFrame was found
        assert result_df is not None, "DataFrame with dimensionality reduction features not found"

        # Verify the dimensionality reduction results
        # With the multiple result columns pattern, we check for the dimension columns
        assert f"{pca_feature}~dim1" in result_df.columns, f"{pca_feature}~dim1 not found"
        assert f"{pca_feature}~dim2" in result_df.columns, f"{pca_feature}~dim2 not found"
        assert f"{ica_feature}~dim1" in result_df.columns, f"{ica_feature}~dim1 not found"
        assert f"{ica_feature}~dim2" in result_df.columns, f"{ica_feature}~dim2 not found"
        assert f"{isomap_feature}~dim1" in result_df.columns, f"{isomap_feature}~dim1 not found"
        assert f"{isomap_feature}~dim2" in result_df.columns, f"{isomap_feature}~dim2 not found"
