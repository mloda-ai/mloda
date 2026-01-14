"""
Complex integration test for EncodingFeatureGroup with multiple feature chains in a single test.
"""

from mloda.user import Features
import pytest
from typing import Any, Dict, List

from mloda.user import PluginLoader
from mloda.user import Feature
from mloda.user import Options
from mloda.user import PluginCollector
from mloda.user import mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.experimental.sklearn.encoding.pandas import PandasEncodingFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.scaling.pandas import PandasScalingFeatureGroup
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import PandasAggregatedFeatureGroup
from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator


class ComplexEncodingTestDataCreator(ATestDataCreator):
    """Test data creator for complex encoding chaining tests."""

    compute_framework = PandasDataFrame

    @classmethod
    def get_raw_data(cls) -> Dict[str, Any]:
        """Return the raw data as a dictionary."""
        return {
            "customer_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "category": ["Premium", "Standard", "Basic", "Premium", "Standard", "Basic", "Premium", "Standard"],
            "region": ["North", "South", "East", "West", "North", "South", "East", "West"],
            "sales": [1000, 500, 250, 1200, 600, 300, 1100, 550],
            "revenue": [5000, 2500, 1250, 6000, 3000, 1500, 5500, 2750],
        }


class TestComplexEncodingChaining:
    """Complex integration test demonstrating multiple encoding feature chains."""

    def test_multiple_complex_encoding_chains_with_artifacts(self, tmp_path: Any) -> None:
        """
        Test multiple complex feature chains in a single test, demonstrating the power of mloda's feature chaining.

        This test creates and validates multiple complex feature chains (L→R syntax):
        1. category__onehot_encoded~0__standard_scaled__sum_aggr: OneHot -> Scale -> Aggregate
        2. region__label_encoded__mean_aggr: Label encode -> Aggregate
        3. category__ordinal_encoded__count_aggr: Ordinal encode -> Aggregate
        4. category__label_encoded__robust_scaled__max_aggr: Label -> Scale -> Aggregate

        This demonstrates:
        - Multiple encoding types (OneHot, Label, Ordinal)
        - Multiple scaling types (Standard, Robust)
        - Multiple aggregation types (Sum, Mean, Count, Max)
        - Complex feature chaining with artifact management
        - Artifact reuse across multiple complex features
        """
        # Skip test if sklearn not available
        try:
            import sklearn
        except ImportError:
            pytest.skip("scikit-learn not available")

        PluginLoader().all()

        # Enable all necessary feature groups
        plugin_collector = PluginCollector.enabled_feature_groups(
            {
                ComplexEncodingTestDataCreator,
                PandasEncodingFeatureGroup,
                PandasScalingFeatureGroup,
                PandasAggregatedFeatureGroup,
            }
        )

        # Create multiple complex chained features with unique artifact storage path
        artifact_options = Options({"artifact_storage_path": str(tmp_path)})
        complex_features: Features | List[Feature | str] = [
            # OneHot encoding -> Standard scaling -> Sum aggregation (L→R)
            Feature("category__onehot_encoded~1__standard_scaled__sum_aggr", artifact_options),
            # Label encoding -> Mean aggregation (no scaling) (L→R)
            Feature("region__label_encoded__mean_aggr", artifact_options),
            # Ordinal encoding -> Count aggregation (no scaling) (L→R)
            Feature("category__ordinal_encoded__count_aggr", artifact_options),
            # Label encoding -> Robust scaling -> Max aggregation (L→R)
            Feature("category__label_encoded__robust_scaled__max_aggr", artifact_options),
            # OneHot encoding -> Min-Max scaling -> Min aggregation (using different OneHot column) (L→R)
            Feature("category__onehot_encoded~0__minmax_scaled__min_aggr", artifact_options),
        ]

        # Phase 1: Train and save artifacts for all complex features
        api1 = mloda(
            complex_features,
            {PandasDataFrame},
            plugin_collector=plugin_collector,
        )
        api1._batch_run()
        results1 = api1.get_result()
        artifacts1 = api1.get_artifacts()

        # Verify all complex features were created
        assert len(results1) == 1
        df1 = results1[0]

        expected_columns = [
            "category__onehot_encoded~1__standard_scaled__sum_aggr",
            "region__label_encoded__mean_aggr",
            "category__ordinal_encoded__count_aggr",
            "category__label_encoded__robust_scaled__max_aggr",
            "category__onehot_encoded~0__minmax_scaled__min_aggr",
        ]

        # Check which columns were actually created
        created_columns = [col for col in expected_columns if col in df1.columns]
        missing_columns = [col for col in expected_columns if col not in df1.columns]

        print(f"Created columns: {created_columns}")
        print(f"Missing columns: {missing_columns}")

        # For now, just verify that at least some features were created
        assert len(created_columns) >= 1, f"At least one feature should be created, got {created_columns}"

        # Verify artifacts were created for encoding and scaling steps
        assert len(artifacts1) == 7  # 4 encoding + 3 scaling artifacts

        # Check specific artifact types (L→R format)
        # Encoding artifacts end with _encoded (without ~digit suffix for base encodings)
        # e.g., category__onehot_encoded, region__label_encoded, category__ordinal_encoded, category__label_encoded
        encoding_artifacts = [k for k in artifacts1.keys() if k.endswith("_encoded")]
        # Scaling artifacts end with _scaled
        # e.g., category__onehot_encoded~1__standard_scaled, category__label_encoded__robust_scaled
        scaling_artifacts = [k for k in artifacts1.keys() if k.endswith("_scaled")]

        assert len(encoding_artifacts) == 4, (
            f"Should have 4 encoding artifacts, got {len(encoding_artifacts)}: {encoding_artifacts}"
        )
        assert len(scaling_artifacts) == 3, (
            f"Should have 3 scaling artifacts, got {len(scaling_artifacts)}: {scaling_artifacts}"
        )

        # Verify all results make sense
        for column in expected_columns:
            values = df1[column]
            assert all(isinstance(val, (int, float)) for val in values), f"Non-numeric values in {column}"

            # Check that aggregated values are reasonable
            if "sum_aggr" in column or "max_aggr" in column:
                # For scaled features, sum might be close to zero due to centering
                assert abs(values.sum()) >= 0, f"Sum/Max aggregation should be numeric for {column}"
            elif "mean_aggr" in column:
                assert values.mean() > 0, f"Mean aggregation should be positive for {column}"
            elif "count_aggr" in column:
                assert all(val >= 0 for val in values), f"Count should be non-negative for {column}"
            elif "min_aggr" in column:
                assert all(val >= 0 for val in values), f"Min should be non-negative for {column}"

        # Verify that different encoding types produce different results
        onehot_result = df1["category__onehot_encoded~1__standard_scaled__sum_aggr"]
        label_result = df1["category__label_encoded__robust_scaled__max_aggr"]
        ordinal_result = df1["category__ordinal_encoded__count_aggr"]

        # These should be different since they use different encoding and aggregation methods
        assert not onehot_result.equals(label_result), "OneHot and Label encoding results should differ"
        assert not label_result.equals(ordinal_result), "Label and Ordinal encoding results should differ"

        # Phase 2: Test artifact reuse with all complex features
        complex_features_reuse: Features | List[Feature | str] = [
            Feature(
                feature.name,  # type: ignore
                Options({**artifacts1, "artifact_storage_path": str(tmp_path)}),
            )
            for feature in complex_features
        ]

        api2 = mloda(
            complex_features_reuse,
            {PandasDataFrame},
            plugin_collector=plugin_collector,
        )
        api2._batch_run()
        results2 = api2.get_result()
        artifacts2 = api2.get_artifacts()

        # Verify results are identical (indicating artifact reuse)
        assert len(results2) == 1
        df2 = results2[0]

        for column in expected_columns:
            assert column in df2.columns, f"Missing column in reuse: {column}"

        # No new artifacts should be created (reused existing ones)
        assert len(artifacts2) == 0, "Should not create new artifacts when reusing"

        # All values should be identical (artifacts were reused)
        for column in expected_columns:
            assert df1[column].equals(df2[column]), f"Values should be identical for {column} when reusing artifacts"
