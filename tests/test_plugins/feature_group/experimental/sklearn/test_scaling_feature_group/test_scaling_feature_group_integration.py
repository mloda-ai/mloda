"""
Integration test for ScalingFeatureGroup with mlodaAPI and artifact management.
"""

from mloda_core.abstract_plugins.plugin_loader.plugin_loader import PluginLoader
import pytest
from typing import Any, Dict

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.experimental.sklearn.scaling.pandas import PandasScalingFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.scaling.base import ScalingFeatureGroup
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import PandasAggregatedFeatureGroup

from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator


class ScalingIntegrationTestDataCreator(ATestDataCreator):
    """Test data creator for scaling integration tests."""

    compute_framework = PandasDataFrame

    @classmethod
    def get_raw_data(cls) -> Dict[str, Any]:
        """Return the raw data as a dictionary."""
        return {
            "Sales": [100, 200, 300, 400, 500],
            "Revenue": [1000, 2000, 3000, 4000, 5000],
        }


class TestScalingFeatureGroupIntegration:
    """Integration test for ScalingFeatureGroup."""

    def test_scaling_with_aggregated_source_and_artifacts(self) -> None:
        """Test scaling feature group using aggregated feature as source with artifact save/load."""
        # Skip test if sklearn not available
        try:
            import sklearn
        except ImportError:
            pytest.skip("scikit-learn not available")

        PluginLoader().all()

        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {ScalingIntegrationTestDataCreator, PandasScalingFeatureGroup, PandasAggregatedFeatureGroup}
        )

        # Create features: aggregated feature and scaling of that aggregated feature
        scaling_feature = Feature("Sales__sum_aggr__standard_scaled")

        # Phase 1: Train and save artifacts
        api1 = mlodaAPI(
            [scaling_feature],
            {PandasDataFrame},
            plugin_collector=plugin_collector,
        )
        api1._batch_run()
        results1 = api1.get_result()
        artifacts1 = api1.get_artifacts()

        # Verify scaling feature was created
        assert len(results1) == 1
        df1 = results1[0]
        assert "Sales__sum_aggr__standard_scaled" in df1.columns

        # Verify artifacts were created for the scaling feature
        assert len(artifacts1) >= 1
        assert "Sales__sum_aggr__standard_scaled" in artifacts1

        # Verify that scaling was applied (all values are 0 since source is constant)
        scaled_values = df1["Sales__sum_aggr__standard_scaled"]
        assert abs(scaled_values.mean()) < 0.1
        # Since the aggregated feature is constant (all 1500), scaling results in all 0s
        assert scaled_values.std() == 0.0

        # Phase 2: Load artifacts and apply to same data (simulating reuse)
        from mloda_core.abstract_plugins.components.options import Options

        # Create features with artifact options for reuse
        scaling_feature_reuse = Feature(
            "Sales__sum_aggr__standard_scaled",
            Options(artifacts1),
        )

        api2 = mlodaAPI(
            [scaling_feature_reuse],
            {PandasDataFrame},
            plugin_collector=plugin_collector,
        )
        api2._batch_run()
        results2 = api2.get_result()
        artifacts2 = api2.get_artifacts()

        # Verify results are identical (indicating artifact reuse)
        assert len(results2) == 1
        df2 = results2[0]
        assert "Sales__sum_aggr__standard_scaled" in df2.columns

        # No new artifacts should be created (reused existing ones)
        assert len(artifacts2) == 0

        # Values should be identical (artifact was reused)
        assert df1["Sales__sum_aggr__standard_scaled"].equals(df2["Sales__sum_aggr__standard_scaled"])
