"""
Runtime validation test for feature configuration integration JSON.
Tests all features from test_config_features.json with mlodaAPI.run_all.
"""

from pathlib import Path
from typing import Any, Dict
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.config.feature.loader import load_features_from_config
from mloda_plugins.feature_group.experimental.sklearn.scaling.pandas import PandasScalingFeatureGroup
from mloda_plugins.feature_group.experimental.data_quality.missing_value.pandas import PandasMissingValueFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.encoding.pandas import PandasEncodingFeatureGroup
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import PandasAggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.geo_distance.pandas import PandasGeoDistanceFeatureGroup
from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator


class IntegrationDataCreator(ATestDataCreator):
    """Provides test data for all columns in integration JSON."""

    compute_framework = PandasDataframe

    @classmethod
    def get_raw_data(cls) -> Dict[str, Any]:
        return {
            "age": [25, 30, 35, 40, 45],
            "weight": [150, 160, 170, 180, 190],
            "state": ["CA", "NY", "TX", "CA", "NY"],
            "category": ["X", "Y", "Z", "X", "Y"],
            "product": ["A", "B", "C", "A", "B"],
            "latitude": [37.7, 40.7, 29.7, 34.0, 41.8],
            "longitude": [-122.4, -74.0, -95.3, -118.2, -87.6],
            "sales": [1000, 1500, 2000, 2500, 3000],
            "revenue": [1200, 1800, 2400, 3000, 3600],
            "profit": [200, 300, 400, 500, 600],
            # Point features for geo distance (latitude, longitude)
            "customer_location": [(37.7, -122.4), (40.7, -74.0), (29.7, -95.3), (34.0, -118.2), (41.8, -87.6)],
            "store_location": [(37.8, -122.5), (40.8, -74.1), (29.8, -95.4), (34.1, -118.3), (41.9, -87.7)],
        }


def test_features_runtime_one_by_one() -> None:
    """
    Test all features from integration JSON with mlodaAPI.run_all.

    This test validates that features not only parse correctly,
    but also execute successfully through the full mloda pipeline.
    """

    json_path = Path(__file__).parent / "test_config_features.json"
    with open(json_path) as f:
        config_str = f.read()

    features = load_features_from_config(config_str, format="json")

    features_to_test = [
        features[0],  # ✅ Feature 0: "age" - simple string
        features[1],  # ✅ Feature 1: "weight" (with options) - object with flat options
        features[2],  # ✅ Feature 2: "age__mean_imputed__standard_scaled" - chained feature
        features[3],  # ✅ Feature 3: "weight__mean_imputed__max_aggr" - aggregation on single column
        features[4],  # ✅ Feature 4: "weight__mean_imputed__min_aggr" - min aggregation with timewindow
        features[5],  # ✅ Feature 5: "state__onehot_encoded~0" - column selector with mloda_source
        features[6],  # ✅ Feature 6: "state__onehot_encoded~1" - column selector with mloda_source
        features[7],  # ✅ Feature 7: "minmaxscaledage" - minmax scaling with mloda_source_features in options
        features[8],  # ✅ Feature 8: "age__max_aggr" - aggregation with mloda_source and window_size option
        features[9],  # ✅ Feature 9: "min_max" - scaling with mloda_source_features in options
        features[10],  # ✅ Feature 10: "customer_location__store_location__haversine_distance" - geo distance feature
        features[11],  # ✅ Feature 11: "custom_geo_distance" - config-based geo distance with mloda_sources and options
        features[
            12
        ],  # ✅ Feature 12: "min_max_nested" - nested mloda_source_features with recursive dependencies (weight -> minmaxscaledweght -> max_aggregated -> min_max_nested)
        features[13],  # ✅ Feature 13: "product__onehot_encoded" - multi-column producer (creates ~0, ~1, ~2)
        features[
            14
        ],  # ✅ Feature 14: "product__onehot_encoded__sum_aggr" - multi-column consumer using automatic discovery
    ]

    # Required plugins (expand as needed)
    plugins = {
        IntegrationDataCreator,
        PandasScalingFeatureGroup,
        PandasMissingValueFeatureGroup,
        PandasEncodingFeatureGroup,
        PandasAggregatedFeatureGroup,
        PandasGeoDistanceFeatureGroup,
        # Add more plugins as features require them
    }

    # Create plugin collector
    plugin_collector = PlugInCollector.enabled_feature_groups(plugins)

    # Run mlodaAPI with all features being tested
    results = mlodaAPI.run_all(
        features_to_test,
        compute_frameworks={PandasDataframe},
        plugin_collector=plugin_collector,
    )

    for res in results:
        print(res.columns)

    # Verify we got results
    assert len(results) > 0, "Expected at least one result DataFrame"

    # Verify each feature appears in results
    for i, feature in enumerate(features_to_test):
        feature_name = feature if isinstance(feature, str) else feature.name.name

        # Check if feature column exists in any result DataFrame
        found = any(feature_name in df.columns for df in results)

        if feature_name == "product__onehot_encoded":
            continue

        assert found, f"Feature {i}: {feature_name} not found in any result DataFrame"

        # Additional verification: check data is not all NaN
        for df in results:
            if feature_name in df.columns:
                assert not df[feature_name].isna().all(), f"Feature {i}: {feature_name} has all NaN values"
                break


def test_feature_3_step1_onehot_encoding() -> None:
    """
    Test Feature 3 Step 1: Create intermediate "state__onehot_encoded" feature as prerequisite.

    This test validates that we need to first create the "state__onehot_encoded" feature
    from "state" before we can use it in the chained feature "state__onehot_encoded__max_aggr".

    The problem: Feature 3 ("state__onehot_encoded__max_aggr" with mloda_source="state")
    expects to chain: state -> onehot_encoded -> max_aggr

    But OneHotEncoder creates MULTIPLE columns (state__onehot_encoded~0, ~1, ~2),
    not a single column named "state__onehot_encoded".

    Solution:
    1. Create "state__onehot_encoded" which produces multiple columns (~0, ~1, ~2)
    2. Create "state__onehot_encoded~0__max_aggr" targeting a specific OneHot column
       (using string-based feature name for proper parsing)
    """
    from mloda_core.abstract_plugins.components.feature import Feature
    from mloda_core.abstract_plugins.components.options import Options
    from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

    # Step 1: Create the intermediate feature "state__onehot_encoded" from "state"
    # This will create multiple columns: state__onehot_encoded~0, ~1, ~2
    intermediate_feature = Feature(
        name="state__onehot_encoded", options=Options(context={DefaultOptionKeys.mloda_source_features: "state"})
    )

    # Step 2: Create the chained feature "state__onehot_encoded~0__max_aggr"
    # Use string-based feature name so the aggregation plugin can parse it correctly
    chained_feature = "state__onehot_encoded~0__max_aggr"

    # Required plugins
    plugins = {
        IntegrationDataCreator,
        PandasEncodingFeatureGroup,
        PandasAggregatedFeatureGroup,
    }

    # Create plugin collector
    plugin_collector = PlugInCollector.enabled_feature_groups(plugins)

    # Run with BOTH features - the intermediate one must be created first
    results = mlodaAPI.run_all(
        [intermediate_feature, chained_feature],
        compute_frameworks={PandasDataframe},
        plugin_collector=plugin_collector,
    )

    # Verify we got results (should be 2 DataFrames, one for encoding, one for aggregation)
    assert len(results) > 0, "Expected at least one result DataFrame"

    # The aggregation result should be in one of the DataFrames
    found_aggregation = False
    for result_df in results:
        if "state__onehot_encoded~0__max_aggr" in result_df.columns:
            found_aggregation = True
            # Verify the column has data (not all NaN)
            assert not result_df["state__onehot_encoded~0__max_aggr"].isna().all(), (
                "Feature 'state__onehot_encoded~0__max_aggr' has all NaN values"
            )
            break

    assert found_aggregation, (
        f"Feature 'state__onehot_encoded~0__max_aggr' not found in any result DataFrame. "
        f"Result DataFrames: {[df.columns.tolist() for df in results]}"
    )
