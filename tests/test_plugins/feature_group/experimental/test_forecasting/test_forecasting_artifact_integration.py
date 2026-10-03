"""
Integration tests for the ForecastingFeatureGroup artifacts with mloda.

This test demonstrates how the ForecastingFeatureGroup artifacts can be saved and loaded,
allowing trained models to be reused for future forecasts.
"""

from typing import Any
from datetime import datetime, timedelta

from mloda.user import Feature
from mloda.user import Options
from mloda.user import PluginCollector
from mloda.user import mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.experimental.sklearn.encoding.pandas import PandasEncodingFeatureGroup
from mloda_plugins.feature_group.experimental.forecasting.pandas import PandasForecastingFeatureGroup
from mloda.provider import DefaultOptionKeys

from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator


class ForecastingArtifactTestDataCreator(ATestDataCreator):
    """Test data creator for forecasting artifact tests."""

    compute_framework = PandasDataFrame

    @classmethod
    def get_raw_data(cls) -> dict[str, Any]:
        """Return the raw data as a dictionary."""
        # Create time series data for 30 days
        dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(30)]
        values = [10 + i + (i % 7) * 2 for i in range(30)]  # Simple pattern with weekly seasonality

        return {
            "time_filter": dates,
            "sales": values,
        }


class ForecastingCategoryTestDataCreator(ATestDataCreator):
    """Test data creator with a time column and a string category."""

    compute_framework = PandasDataFrame

    @classmethod
    def get_raw_data(cls) -> dict[str, Any]:
        """Return the raw data as a dictionary."""
        return {
            "time_filter": [datetime(2025, 1, 1) + timedelta(days=i) for i in range(30)],
            "category": [["a", "b", "c"][i % 3] for i in range(30)],
        }


class TestForecastingArtifactIntegration:
    def test_forecast_chained_onehot_source_per_category(self) -> None:
        """A chained one-hot source yields one forecast column per category."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {ForecastingCategoryTestDataCreator, PandasEncodingFeatureGroup, PandasForecastingFeatureGroup}
        )
        feature_name = "category__onehot_encoded__linear_forecast_7day"
        options = Options({DefaultOptionKeys.reference_time: "time_filter"})
        feature = Feature(feature_name, options)

        api = mloda([feature], compute_frameworks=[PandasDataFrame], plugin_collector=plugin_collector)
        results = api.run()

        columns = [c for r in results for c in r.columns]
        assert feature_name not in columns
        for suffix in ("0", "1", "2"):
            assert f"{feature_name}~{suffix}" in columns

        artifacts = api.get_artifacts()
        assert feature_name in artifacts

        feature2 = Feature(feature_name, options=options)
        feature2.options.add_to_group(feature_name, artifacts[feature_name])
        api2 = mloda([feature2], compute_frameworks=[PandasDataFrame], plugin_collector=plugin_collector)
        results2 = api2.run()

        columns2 = [c for r in results2 for c in r.columns]
        assert feature_name not in columns2
        for suffix in ("0", "1", "2"):
            assert f"{feature_name}~{suffix}" in columns2

    def test_forecast_chained_onehot_source_selects_one_category(self) -> None:
        """Requesting `name~1` on a chained one-hot source returns only the `~1` forecast."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {ForecastingCategoryTestDataCreator, PandasEncodingFeatureGroup, PandasForecastingFeatureGroup}
        )
        base_name = "category__onehot_encoded__linear_forecast_7day"
        options = Options({DefaultOptionKeys.reference_time: "time_filter"})
        feature = Feature(f"{base_name}~1", options)

        results = mloda([feature], compute_frameworks=[PandasDataFrame], plugin_collector=plugin_collector).run()

        columns = [c for r in results for c in r.columns]
        assert f"{base_name}~1" in columns
        assert f"{base_name}~0" not in columns
        assert f"{base_name}~2" not in columns

    def test_forecast_multi_and_selected_together_artifact_reload(self) -> None:
        """Requesting `name` and `name~1` together saves the nested artifact under `name` and reloads."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {ForecastingCategoryTestDataCreator, PandasEncodingFeatureGroup, PandasForecastingFeatureGroup}
        )
        base_name = "category__onehot_encoded__linear_forecast_7day"
        options = Options({DefaultOptionKeys.reference_time: "time_filter"})
        features: list[Feature | str] = [Feature(base_name, options), Feature(f"{base_name}~1", options)]

        api = mloda(features, compute_frameworks=[PandasDataFrame], plugin_collector=plugin_collector)
        api.run()
        artifact = api.get_artifacts()[base_name]

        options2 = Options({DefaultOptionKeys.reference_time: "time_filter"})
        options2.add_to_group(base_name, artifact)
        features2: list[Feature | str] = [Feature(base_name, options2), Feature(f"{base_name}~1", options2)]
        api2 = mloda(features2, compute_frameworks=[PandasDataFrame], plugin_collector=plugin_collector)
        results2 = api2.run()

        columns2 = [c for r in results2 for c in r.columns]
        for suffix in ("0", "1", "2"):
            assert f"{base_name}~{suffix}" in columns2
        assert f"{base_name}~1" in columns2

    def test_artifact_save_and_load(self) -> None:
        """Test saving and loading forecasting artifacts."""
        # Enable the necessary feature groups
        plugin_collector = PluginCollector.enabled_feature_groups(
            {ForecastingArtifactTestDataCreator, PandasForecastingFeatureGroup}
        )

        # Create a feature for linear forecasting with 7-day horizon
        feature_name = "sales__linear_forecast_7day"
        feature = Feature(feature_name)

        # Set reference time option
        options = Options({DefaultOptionKeys.reference_time: "time_filter"})
        feature.options = options

        # First run: Train and save the model artifact
        api = mloda(
            [feature],
            compute_frameworks=[PandasDataFrame],
            plugin_collector=plugin_collector,
        )

        # Run the mloda to generate forecasts and save the artifact
        results1 = api.run()

        # Get the saved artifacts
        artifacts = api.get_artifacts()

        # Verify that an artifact was saved for our feature
        assert feature_name in artifacts, f"No artifact saved for {feature_name}"

        # Create a new mloda instance for loading the artifact
        feature2 = Feature(feature_name, options=options)

        # Add the artifact to the feature's options
        feature2.options.add_to_group(feature_name, artifacts[feature_name])

        # Create a new mloda with the artifact
        api2 = mloda(
            [feature2],
            compute_frameworks=[PandasDataFrame],
            plugin_collector=plugin_collector,
        )

        # Run the mloda to generate forecasts using the loaded artifact
        results2 = api2.run()

        # Verify that both runs produced results with the same feature
        assert feature_name in results1[0].columns, f"{feature_name} not found in first run results"
        assert feature_name in results2[0].columns, f"{feature_name} not found in second run results"

        # The forecasts might not be identical due to randomness in the data,
        # but they should have the same length
        assert len(results1[0]) == len(results2[0]), "Results have different lengths"
