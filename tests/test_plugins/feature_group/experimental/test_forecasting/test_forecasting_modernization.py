"""
Tests for the modernized ForecastingFeatureGroup with both string-based and configuration-based features.

This test verifies that the modernization successfully supports both approaches:
1. String-based feature creation (legacy)
2. Configuration-based feature creation (modern)
"""

from mloda_core.abstract_plugins.components.feature_collection import Features
import pytest
from typing import Any, Dict, List
from datetime import datetime, timedelta

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.experimental.forecasting.base import ForecastingFeatureGroup
from mloda_plugins.feature_group.experimental.forecasting.pandas import PandasForecastingFeatureGroup
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator


class ForecastingModernizationTestDataCreator(ATestDataCreator):
    """Test data creator for forecasting modernization tests."""

    compute_framework = PandasDataframe

    @classmethod
    def get_raw_data(cls) -> Dict[str, Any]:
        """Return the raw data as a dictionary."""
        # Create time series data for 1000 days to ensure robust testing with lag features
        dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(1000)]
        values = [10 + i + (i % 7) * 2 for i in range(1000)]  # Simple pattern with weekly seasonality

        return {
            "time_filter": dates,
            "sales": values,
        }


class TestForecastingModernization:
    """Test the modernized ForecastingFeatureGroup with both string-based and configuration-based features."""

    def test_string_based_feature_creation(self) -> None:
        """Test that string-based feature creation still works (legacy approach)."""
        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {ForecastingModernizationTestDataCreator, PandasForecastingFeatureGroup}
        )

        # Create a string-based feature
        feature_name = "linear_forecast_7day__sales"
        feature = Feature(feature_name)

        # Set reference time option
        options = Options({DefaultOptionKeys.reference_time.value: "time_filter"})
        feature.options = options

        # Run the API
        api = mlodaAPI(
            [feature],
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        api._batch_run()
        results = api.get_result()

        # Verify that the feature was created successfully
        assert len(results) == 1
        assert feature_name in results[0].columns
        assert len(results[0]) > 0  # Should have data

    def test_configuration_based_feature_creation(self) -> None:
        """Test that configuration-based feature creation works (modern approach)."""
        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {ForecastingModernizationTestDataCreator, PandasForecastingFeatureGroup}
        )

        # Create a configuration-based feature
        feature = Feature(
            name="forecast_feature",
            options=Options(
                group={DefaultOptionKeys.reference_time.value: "time_filter"},
                context={
                    ForecastingFeatureGroup.ALGORITHM: "linear",
                    ForecastingFeatureGroup.HORIZON: 7,
                    ForecastingFeatureGroup.TIME_UNIT: "day",
                    DefaultOptionKeys.mloda_source_feature: "sales",
                },
            ),
        )

        # Run the API
        api = mlodaAPI(
            [feature],
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        api._batch_run()
        results = api.get_result()

        # Verify that the feature was created successfully
        assert len(results) == 1
        # The feature keeps its original name (no automatic name generation)
        assert "forecast_feature" in results[0].columns
        assert len(results[0]) > 0  # Should have data

    def test_both_approaches_produce_equivalent_results(self) -> None:
        """Test that both string-based and configuration-based approaches produce equivalent functionality."""
        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {ForecastingModernizationTestDataCreator, PandasForecastingFeatureGroup}
        )

        # Create string-based feature
        string_feature = Feature("linear_forecast_7day__sales")
        string_feature.options = Options({DefaultOptionKeys.reference_time.value: "time_filter"})

        # Create configuration-based feature
        config_feature = Feature(
            name="config_forecast",
            options=Options(
                group={DefaultOptionKeys.reference_time.value: "time_filter"},
                context={
                    ForecastingFeatureGroup.ALGORITHM: "linear",
                    ForecastingFeatureGroup.HORIZON: 7,
                    ForecastingFeatureGroup.TIME_UNIT: "day",
                    DefaultOptionKeys.mloda_source_feature: "sales",
                },
            ),
        )

        # Run both approaches
        api1 = mlodaAPI([string_feature], {PandasDataframe}, plugin_collector=plugin_collector)
        api1._batch_run()
        results1 = api1.get_result()

        api2 = mlodaAPI([config_feature], {PandasDataframe}, plugin_collector=plugin_collector)
        api2._batch_run()
        results2 = api2.get_result()

        # Both should produce results with their respective feature names
        assert "linear_forecast_7day__sales" in results1[0].columns
        assert "config_forecast" in results2[0].columns

        # Results should have the same structure (though feature names differ)
        assert len(results1[0]) == len(results2[0])
        # Both should have the same number of columns (original data + 1 forecast feature)
        assert len(results1[0].columns) == len(results2[0].columns)

    def test_parameter_validation_in_configuration_based_features(self) -> None:
        """Test that parameter validation works correctly for configuration-based features."""
        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {ForecastingModernizationTestDataCreator, PandasForecastingFeatureGroup}
        )

        # Test invalid algorithm
        with pytest.raises(Exception):  # Should fail during validation
            feature = Feature(
                name="placeholder",
                options=Options(
                    context={
                        ForecastingFeatureGroup.ALGORITHM: "invalid_algorithm",
                        ForecastingFeatureGroup.HORIZON: 7,
                        ForecastingFeatureGroup.TIME_UNIT: "day",
                        DefaultOptionKeys.mloda_source_feature: "sales",
                    }
                ),
            )
            api = mlodaAPI([feature], {PandasDataframe}, plugin_collector=plugin_collector)
            api._batch_run()

        # Test invalid time unit
        with pytest.raises(Exception):  # Should fail during validation
            feature = Feature(
                name="placeholder",
                options=Options(
                    context={
                        ForecastingFeatureGroup.ALGORITHM: "linear",
                        ForecastingFeatureGroup.HORIZON: 7,
                        ForecastingFeatureGroup.TIME_UNIT: "invalid_unit",
                        DefaultOptionKeys.mloda_source_feature: "sales",
                    }
                ),
            )
            api = mlodaAPI([feature], {PandasDataframe}, plugin_collector=plugin_collector)
            api._batch_run()

        # Test invalid horizon (negative)
        with pytest.raises(Exception):  # Should fail during validation
            feature = Feature(
                name="placeholder",
                options=Options(
                    context={
                        ForecastingFeatureGroup.ALGORITHM: "linear",
                        ForecastingFeatureGroup.HORIZON: -1,
                        ForecastingFeatureGroup.TIME_UNIT: "day",
                        DefaultOptionKeys.mloda_source_feature: "sales",
                    }
                ),
            )
            api = mlodaAPI([feature], {PandasDataframe}, plugin_collector=plugin_collector)
            api._batch_run()

    def test_multiple_algorithms_configuration_based(self) -> None:
        """Test multiple forecasting algorithms using configuration-based approach."""
        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {ForecastingModernizationTestDataCreator, PandasForecastingFeatureGroup}
        )

        algorithms = ["linear", "ridge", "randomforest"]
        features: List[Feature | str] = []

        for i, algorithm in enumerate(algorithms):
            feature = Feature(
                name=f"forecast_{algorithm}",
                options=Options(
                    group={DefaultOptionKeys.reference_time.value: "time_filter"},
                    context={
                        ForecastingFeatureGroup.ALGORITHM: algorithm,
                        ForecastingFeatureGroup.HORIZON: 5,
                        ForecastingFeatureGroup.TIME_UNIT: "day",
                        DefaultOptionKeys.mloda_source_feature: "sales",
                    },
                ),
            )
            features.append(feature)

        # Run the API with multiple features
        api = mlodaAPI(features, {PandasDataframe}, plugin_collector=plugin_collector)
        api._batch_run()
        results = api.get_result()

        # Verify all features were created
        assert len(results) == 1
        for algorithm in algorithms:
            feature_name = f"forecast_{algorithm}"
            assert feature_name in results[0].columns

    def test_context_parameters_dont_affect_feature_group_resolution(self) -> None:
        """Test that context parameters don't affect Feature Group resolution/splitting."""
        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {ForecastingModernizationTestDataCreator, PandasForecastingFeatureGroup}
        )

        # Create two features with different context parameters but same group parameters
        feature1 = Feature(
            name="forecast_linear",
            options=Options(
                group={DefaultOptionKeys.reference_time.value: "time_filter"},
                context={
                    ForecastingFeatureGroup.ALGORITHM: "linear",
                    ForecastingFeatureGroup.HORIZON: 7,
                    ForecastingFeatureGroup.TIME_UNIT: "day",
                    DefaultOptionKeys.mloda_source_feature: "sales",
                },
            ),
        )

        feature2 = Feature(
            name="forecast_ridge",
            options=Options(
                group={DefaultOptionKeys.reference_time.value: "time_filter"},
                context={
                    ForecastingFeatureGroup.ALGORITHM: "ridge",  # Different algorithm (context parameter)
                    ForecastingFeatureGroup.HORIZON: 14,  # Different horizon (context parameter)
                    ForecastingFeatureGroup.TIME_UNIT: "day",
                    DefaultOptionKeys.mloda_source_feature: "sales",
                },
            ),
        )

        # Run the API with both features
        api = mlodaAPI([feature1, feature2], {PandasDataframe}, plugin_collector=plugin_collector)
        api._batch_run()
        results = api.get_result()

        # Both features should be processed together (same Feature Group resolution)
        assert len(results) == 1
        assert "forecast_linear" in results[0].columns
        assert "forecast_ridge" in results[0].columns

    def test_match_feature_group_criteria_with_property_mapping(self) -> None:
        """Test that match_feature_group_criteria works with the new PROPERTY_MAPPING approach."""
        # Test string-based feature matching
        string_feature_name = "linear_forecast_7day__sales"
        string_options = Options({DefaultOptionKeys.reference_time.value: "time_filter"})

        assert ForecastingFeatureGroup.match_feature_group_criteria(string_feature_name, string_options) is True

        # Test configuration-based feature matching
        config_options = Options(
            context={
                ForecastingFeatureGroup.ALGORITHM: "linear",
                ForecastingFeatureGroup.HORIZON: 7,
                ForecastingFeatureGroup.TIME_UNIT: "day",
                DefaultOptionKeys.mloda_source_feature: "sales",
            }
        )

        assert ForecastingFeatureGroup.match_feature_group_criteria("forecast_feature", config_options) is True

        # Test invalid configuration-based feature (should raise exception during validation)
        invalid_config_options = Options(
            context={
                ForecastingFeatureGroup.ALGORITHM: "invalid_algorithm",
                ForecastingFeatureGroup.HORIZON: 7,
                ForecastingFeatureGroup.TIME_UNIT: "day",
                DefaultOptionKeys.mloda_source_feature: "sales",
            }
        )

        # This should return False due to validation failure
        try:
            result = ForecastingFeatureGroup.match_feature_group_criteria("forecast_feature", invalid_config_options)
            assert result is False
        except ValueError:
            # Validation failure is also acceptable behavior
            pass

        # Test non-matching string-based feature
        assert ForecastingFeatureGroup.match_feature_group_criteria("not_a_forecast_feature", string_options) is False
