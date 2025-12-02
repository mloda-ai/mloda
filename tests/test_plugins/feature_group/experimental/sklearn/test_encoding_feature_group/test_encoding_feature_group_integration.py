"""
Integration tests for the EncodingFeatureGroup classes.
"""

import pytest
from typing import Any, Dict

from mloda_core.abstract_plugins.plugin_loader.plugin_loader import PluginLoader
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.experimental.sklearn.encoding.pandas import PandasEncodingFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.encoding.base import EncodingFeatureGroup
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys
from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator


class EncodingIntegrationTestDataCreator(ATestDataCreator):
    """Test data creator for encoding integration tests."""

    compute_framework = PandasDataframe

    @classmethod
    def get_raw_data(cls) -> Dict[str, Any]:
        """Return the raw data as a dictionary."""
        return {
            "customer_id": [1, 2, 3, 4, 5],
            "category": ["Premium", "Standard", "Basic", "Premium", "Standard"],
            "status": ["A", "B", "C", "A", "B"],
            "value": [100, 50, 25, 120, 60],
        }


class TestEncodingFeatureGroupIntegration:
    """Integration tests for encoding feature groups."""

    def test_label_encoding_with_artifacts(self, tmp_path: Any) -> None:
        """Test label encoding feature group with artifact save/load."""
        # Skip test if sklearn not available
        try:
            import sklearn
        except ImportError:
            pytest.skip("scikit-learn not available")

        PluginLoader().all()

        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {EncodingIntegrationTestDataCreator, PandasEncodingFeatureGroup}
        )

        # Create label encoding feature with unique artifact storage path
        label_feature = Feature("category__label_encoded", Options({"artifact_storage_path": str(tmp_path)}))

        # Phase 1: Train and save artifacts
        api1 = mlodaAPI(
            [label_feature],
            {PandasDataframe},
            plugin_collector=plugin_collector,
        )
        api1._batch_run()
        results1 = api1.get_result()
        artifacts1 = api1.get_artifacts()

        # Verify encoding feature was created
        assert len(results1) == 1
        df1 = results1[0]
        assert "category__label_encoded" in df1.columns

        # Verify artifacts were created for the encoding feature
        assert len(artifacts1) >= 1
        assert "category__label_encoded" in artifacts1

        # Verify that encoding was applied (should have numeric values)
        encoded_values = df1["category__label_encoded"]
        assert all(isinstance(val, (int, float)) for val in encoded_values)
        # Should have 3 unique categories: Premium, Standard, Basic
        assert len(set(encoded_values)) == 3

        # Phase 2: Load artifacts and apply to same data (simulating reuse)
        label_feature_reuse = Feature(
            "category__label_encoded",
            Options({**artifacts1, "artifact_storage_path": str(tmp_path)}),
        )

        api2 = mlodaAPI(
            [label_feature_reuse],
            {PandasDataframe},
            plugin_collector=plugin_collector,
        )
        api2._batch_run()
        results2 = api2.get_result()
        artifacts2 = api2.get_artifacts()

        # Verify results are identical (indicating artifact reuse)
        assert len(results2) == 1
        df2 = results2[0]
        assert "category__label_encoded" in df2.columns

        # No new artifacts should be created (reused existing ones)
        assert len(artifacts2) == 0

        # Values should be identical (artifact was reused)
        assert df1["category__label_encoded"].equals(df2["category__label_encoded"])

    def test_onehot_encoding_with_artifacts(self, tmp_path: Any) -> None:
        """Test one-hot encoding feature group with artifact save/load."""
        # Skip test if sklearn not available
        try:
            import sklearn
        except ImportError:
            pytest.skip("scikit-learn not available")

        PluginLoader().all()

        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {EncodingIntegrationTestDataCreator, PandasEncodingFeatureGroup}
        )

        # Create one-hot encoding feature with unique artifact storage path
        onehot_feature = Feature("category__onehot_encoded", Options({"artifact_storage_path": str(tmp_path)}))

        # Phase 1: Train and save artifacts
        api1 = mlodaAPI(
            [onehot_feature],
            {PandasDataframe},
            plugin_collector=plugin_collector,
        )
        api1._batch_run()
        results1 = api1.get_result()
        artifacts1 = api1.get_artifacts()

        # Verify encoding feature was created
        assert len(results1) == 1
        df1 = results1[0]

        # Check that multiple columns were created with ~ separator
        onehot_columns = [col for col in df1.columns if col.startswith("category__onehot_encoded~")]
        assert len(onehot_columns) >= 2  # Should have multiple categories

        # Verify artifacts were created for the encoding feature
        assert len(artifacts1) >= 1
        assert "category__onehot_encoded" in artifacts1

        # Each row should have exactly one 1 and rest 0s
        for i in range(len(df1)):
            row_values = [df1[col].iloc[i] for col in onehot_columns]
            assert sum(row_values) == 1  # Exactly one 1
            assert all(val in [0, 1] for val in row_values)  # Only 0s and 1s

    def test_onehot_encoding_specific_column_access(self) -> None:
        """Test one-hot encoding with specific column access using ~0, ~1, etc."""
        # Skip test if sklearn not available
        try:
            import sklearn
        except ImportError:
            pytest.skip("scikit-learn not available")

        PluginLoader().all()

        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {EncodingIntegrationTestDataCreator, PandasEncodingFeatureGroup}
        )

        # Create specific one-hot encoding features for individual columns
        onehot_feature_0 = Feature("category__onehot_encoded~0")
        onehot_feature_1 = Feature("category__onehot_encoded~1")

        # Phase 1: Test individual column access
        api1 = mlodaAPI(
            [onehot_feature_0, onehot_feature_1],
            {PandasDataframe},
            plugin_collector=plugin_collector,
        )
        api1._batch_run()
        results1 = api1.get_result()
        artifacts1 = api1.get_artifacts()

        # Verify specific columns were created
        assert len(results1) == 1
        df1 = results1[0]

        # Should have the specific columns we requested
        assert "category__onehot_encoded~0" in df1.columns
        assert "category__onehot_encoded~1" in df1.columns

        # Verify artifacts were created for the encoding feature
        assert len(artifacts1) >= 1
        assert "category__onehot_encoded" in artifacts1

        # Verify the columns contain only 0s and 1s
        assert all(val in [0, 1] for val in df1["category__onehot_encoded~0"])
        assert all(val in [0, 1] for val in df1["category__onehot_encoded~1"])

        # Test that the columns are complementary for binary case or part of multi-class
        col_0_values = df1["category__onehot_encoded~0"].tolist()
        col_1_values = df1["category__onehot_encoded~1"].tolist()

        # At least one column should have some 1s (not all zeros)
        assert sum(col_0_values) > 0 or sum(col_1_values) > 0

        # Phase 2: Test that we can also get the full onehot encoding
        onehot_full_feature = Feature("category__onehot_encoded")

        api2 = mlodaAPI(
            [onehot_full_feature],
            {PandasDataframe},
            plugin_collector=plugin_collector,
        )
        api2._batch_run()
        results2 = api2.get_result()

        # Verify full encoding creates all columns
        assert len(results2) == 1
        df2 = results2[0]

        # Should have multiple columns with ~ separator
        full_onehot_columns = [col for col in df2.columns if col.startswith("category__onehot_encoded~")]
        assert len(full_onehot_columns) >= 2  # Should have at least the columns we know exist

        # The individual columns we requested should match the corresponding columns in full encoding
        if "category__onehot_encoded~0" in df2.columns:
            assert df1["category__onehot_encoded~0"].equals(df2["category__onehot_encoded~0"])
        if "category__onehot_encoded~1" in df2.columns:
            assert df1["category__onehot_encoded~1"].equals(df2["category__onehot_encoded~1"])

    def test_configuration_based_onehot_with_column_suffix(self) -> None:
        """Test configuration-based OneHot encoding with specific column access (~0, ~1)."""
        # Skip test if sklearn not available
        try:
            import sklearn
        except ImportError:
            pytest.skip("scikit-learn not available")

        PluginLoader().all()

        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {EncodingIntegrationTestDataCreator, PandasEncodingFeatureGroup}
        )

        # Create configuration-based OneHot encoding features with specific column access
        onehot_config_feature_0 = Feature(
            "onehot1",  # Specific column with ~0 suffix
            Options(
                context={
                    EncodingFeatureGroup.ENCODER_TYPE: "onehot",
                    DefaultOptionKeys.in_features: "category",
                }
            ),
        )

        onehot_config_feature_1 = Feature(
            "onehot2",  # Specific column with ~1 suffix
            Options(
                context={
                    EncodingFeatureGroup.ENCODER_TYPE: "onehot",
                    DefaultOptionKeys.in_features: "category",
                }
            ),
        )

        # Test configuration-based features with column suffixes
        api = mlodaAPI(
            [onehot_config_feature_0, onehot_config_feature_1],
            {PandasDataframe},
            plugin_collector=plugin_collector,
        )
        api._batch_run()
        results = api.get_result()
        artifacts = api.get_artifacts()

        # Verify results
        assert len(results) == 1
        df = results[0]

        # Should have the specific columns we requested
        assert "onehot1~1" in df.columns
        assert "onehot2~2" in df.columns

        # Verify artifacts were created for the encoding feature
        assert len(artifacts) >= 1
        assert "category__onehot_encoded" in artifacts

        # Verify the columns contain only 0s and 1s
        assert all(val in [0, 1] for val in df["onehot1~1"])
        assert all(val in [0, 1] for val in df["onehot2~2"])

        # Test that configuration-based and string-based approaches produce identical results
        # Create equivalent string-based features
        onehot_string_feature_0 = Feature("category__onehot_encoded~0")
        onehot_string_feature_1 = Feature("category__onehot_encoded~1")

        api_string = mlodaAPI(
            [onehot_string_feature_0, onehot_string_feature_1],
            {PandasDataframe},
            plugin_collector=plugin_collector,
        )
        api_string._batch_run()
        results_string = api_string.get_result()

        # Verify string-based results match configuration-based results
        assert len(results_string) == 1
        df_string = results_string[0]

        # Compare specific column results
        assert df["onehot1~0"].equals(df_string["category__onehot_encoded~0"])
        assert df["onehot2~1"].equals(df_string["category__onehot_encoded~1"])

        # At least one column should have some 1s (not all zeros)
        col_0_values = df["onehot1~1"].tolist()
        col_1_values = df["onehot2~2"].tolist()
        assert sum(col_0_values) > 0 or sum(col_1_values) > 0
