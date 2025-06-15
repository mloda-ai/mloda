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

    def test_label_encoding_with_artifacts(self) -> None:
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

        # Create label encoding feature
        label_feature = Feature("label_encoded__category")

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
        assert "label_encoded__category" in df1.columns

        # Verify artifacts were created for the encoding feature
        assert len(artifacts1) >= 1
        assert "label_encoded__category" in artifacts1

        # Verify that encoding was applied (should have numeric values)
        encoded_values = df1["label_encoded__category"]
        assert all(isinstance(val, (int, float)) for val in encoded_values)
        # Should have 3 unique categories: Premium, Standard, Basic
        assert len(set(encoded_values)) == 3

        # Phase 2: Load artifacts and apply to same data (simulating reuse)
        label_feature_reuse = Feature(
            "label_encoded__category",
            Options(artifacts1),
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
        assert "label_encoded__category" in df2.columns

        # No new artifacts should be created (reused existing ones)
        assert len(artifacts2) == 0

        # Values should be identical (artifact was reused)
        assert df1["label_encoded__category"].equals(df2["label_encoded__category"])

    def test_onehot_encoding_with_artifacts(self) -> None:
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

        # Create one-hot encoding feature
        onehot_feature = Feature("onehot_encoded__category")

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
        onehot_columns = [col for col in df1.columns if col.startswith("onehot_encoded__category~")]
        assert len(onehot_columns) >= 2  # Should have multiple categories

        # Verify artifacts were created for the encoding feature
        assert len(artifacts1) >= 1
        assert "onehot_encoded__category" in artifacts1

        # Each row should have exactly one 1 and rest 0s
        for i in range(len(df1)):
            row_values = [df1[col].iloc[i] for col in onehot_columns]
            assert sum(row_values) == 1  # Exactly one 1
            assert all(val in [0, 1] for val in row_values)  # Only 0s and 1s

    def test_configuration_based_feature_creation(self) -> None:
        """Test creating encoding features using configuration options."""
        # Create feature with configuration options
        feature = Feature(
            "placeholder",
            Options(
                {PandasEncodingFeatureGroup.ENCODER_TYPE: "label", DefaultOptionKeys.mloda_source_feature: "status"}
            ),
        )

        # Get parser configuration
        parser_config = PandasEncodingFeatureGroup.configurable_feature_chain_parser()
        assert parser_config is not None

        # Parse feature name from options
        parsed_feature = parser_config.create_feature_without_options(feature)
        assert parsed_feature is not None
        assert parsed_feature.name.name == "label_encoded__status"

        # Verify options were removed
        assert PandasEncodingFeatureGroup.ENCODER_TYPE not in parsed_feature.options.data
        assert DefaultOptionKeys.mloda_source_feature not in parsed_feature.options.data
