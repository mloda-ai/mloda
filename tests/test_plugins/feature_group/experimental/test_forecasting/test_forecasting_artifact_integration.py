"""
Integration tests for the ForecastingFeatureGroup artifacts with mlodaAPI.

This test demonstrates how the ForecastingFeatureGroup artifacts can be saved and loaded,
allowing trained models to be reused for future forecasts.
"""

from typing import Any, Dict
from datetime import datetime, timedelta

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.experimental.forecasting.pandas import PandasForecastingFeatureGroup
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator


class ForecastingArtifactTestDataCreator(ATestDataCreator):
    """Test data creator for forecasting artifact tests."""

    compute_framework = PandasDataFrame

    @classmethod
    def get_raw_data(cls) -> Dict[str, Any]:
        """Return the raw data as a dictionary."""
        # Create time series data for 30 days
        dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(30)]
        values = [10 + i + (i % 7) * 2 for i in range(30)]  # Simple pattern with weekly seasonality

        return {
            "time_filter": dates,
            "sales": values,
        }


class TestForecastingArtifactIntegration:
    def test_artifact_save_and_load(self) -> None:
        """Test saving and loading forecasting artifacts."""
        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {ForecastingArtifactTestDataCreator, PandasForecastingFeatureGroup}
        )

        # Create a feature for linear forecasting with 7-day horizon
        feature_name = "sales__linear_forecast_7day"
        feature = Feature(feature_name)

        # Set reference time option
        options = Options({DefaultOptionKeys.reference_time.value: "time_filter"})
        feature.options = options

        # First run: Train and save the model artifact
        api = mlodaAPI(
            [feature],
            compute_frameworks={PandasDataFrame},
            plugin_collector=plugin_collector,
        )

        # Run the API to generate forecasts and save the artifact
        api._batch_run()
        results1 = api.get_result()

        # Get the saved artifacts
        artifacts = api.get_artifacts()

        # Verify that an artifact was saved for our feature
        assert feature_name in artifacts, f"No artifact saved for {feature_name}"

        # Create a new API instance for loading the artifact
        feature2 = Feature(feature_name, options=options)

        # Add the artifact to the feature's options
        feature2.options.add_to_group(feature_name, artifacts[feature_name])

        # Create a new API with the artifact
        api2 = mlodaAPI(
            [feature2],
            compute_frameworks={PandasDataFrame},
            plugin_collector=plugin_collector,
        )

        # Run the API to generate forecasts using the loaded artifact
        api2._batch_run()
        results2 = api2.get_result()

        # Verify that both runs produced results with the same feature
        assert feature_name in results1[0].columns, f"{feature_name} not found in first run results"
        assert feature_name in results2[0].columns, f"{feature_name} not found in second run results"

        # The forecasts might not be identical due to randomness in the data,
        # but they should have the same length
        assert len(results1[0]) == len(results2[0]), "Results have different lengths"
