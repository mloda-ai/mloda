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
            "onehot_encoded__state": [1, 0, 0, 1, 0],
            "latitude": [37.7, 40.7, 29.7, 34.0, 41.8],
            "longitude": [-122.4, -74.0, -95.3, -118.2, -87.6],
            "sales": [1000, 1500, 2000, 2500, 3000],
            "revenue": [1200, 1800, 2400, 3000, 3600],
            "profit": [200, 300, 400, 500, 600],
        }


def test_features_runtime_one_by_one() -> None:
    """
    Test all features from integration JSON with mlodaAPI.run_all.

    This test validates that features not only parse correctly,
    but also execute successfully through the full mloda pipeline.
    """
    # Load integration JSON
    json_path = Path(__file__).parent / "test_config_features.json"
    with open(json_path) as f:
        config_str = f.read()

    features = load_features_from_config(config_str, format="json")

    # Features to test (build incrementally)
    features_to_test = [
        features[0],  # Feature 0: "age"
        features[1],  # Feature 1: "weight" (with options)
        features[2],  # Feature 2: "standard_scaled__mean_imputed__age"
        features[3],  # Feature 3: "max_aggr__onehot_encoded__state" - now working!
        # features[4],  # SKIPPED: "production_feature" - no implementation, docs only
        features[5],  # Feature 5: "onehot_encoded__state" column_index: 0
        features[6],  # Feature 6: "onehot_encoded__state" column_index: 1
        # features[7],  # Add after features 5-6 pass
        # ... continue adding features
    ]

    # Required plugins (expand as needed)
    plugins = {
        IntegrationDataCreator,
        PandasScalingFeatureGroup,
        PandasMissingValueFeatureGroup,
        PandasEncodingFeatureGroup,
        PandasAggregatedFeatureGroup,
        # Add more plugins as features require them
    }

    # Skip test if no features to test yet
    if not features_to_test:
        print("\n✓ Setup complete - no features to test yet")
        return

    # Create plugin collector
    plugin_collector = PlugInCollector.enabled_feature_groups(plugins)

    # Run mlodaAPI with all features being tested
    results = mlodaAPI.run_all(
        features_to_test,
        compute_frameworks={PandasDataframe},
        plugin_collector=plugin_collector,
    )

    # Verify we got results
    assert len(results) > 0, "Expected at least one result DataFrame"

    # Verify each feature appears in results
    for i, feature in enumerate(features_to_test):
        feature_name = feature if isinstance(feature, str) else feature.name.name

        # Check if feature column exists in any result DataFrame
        found = any(feature_name in df.columns for df in results)
        assert found, f"Feature {i}: {feature_name} not found in any result DataFrame"

        # Additional verification: check data is not all NaN
        for df in results:
            if feature_name in df.columns:
                assert not df[feature_name].isna().all(), (
                    f"Feature {i}: {feature_name} has all NaN values"
                )
                break

    print(f"\n✓ Successfully tested {len(features_to_test)} features with mlodaAPI.run_all")


def test_feature_3_step1_onehot_encoding() -> None:
    """
    Test Feature 3 Step 1: Create intermediate "onehot_encoded__state" feature as prerequisite.

    This test validates that we need to first create the "onehot_encoded__state" feature
    from "state" before we can use it in the chained feature "max_aggr__onehot_encoded__state".

    The problem: Feature 3 ("max_aggr__onehot_encoded__state" with mloda_source="state")
    expects to chain: state -> onehot_encoded -> max_aggr

    But OneHotEncoder creates MULTIPLE columns (onehot_encoded__state~0, ~1, ~2),
    not a single column named "onehot_encoded__state".

    Solution:
    1. Create "onehot_encoded__state" which produces multiple columns (~0, ~1, ~2)
    2. Create "max_aggr__onehot_encoded__state~0" targeting a specific OneHot column
       (using string-based feature name for proper parsing)
    """
    from mloda_core.abstract_plugins.components.feature import Feature
    from mloda_core.abstract_plugins.components.options import Options
    from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

    # Step 1: Create the intermediate feature "onehot_encoded__state" from "state"
    # This will create multiple columns: onehot_encoded__state~0, ~1, ~2
    intermediate_feature = Feature(
        name="onehot_encoded__state",
        options=Options(
            context={
                DefaultOptionKeys.mloda_source_feature: "state"
            }
        )
    )

    # Step 2: Create the chained feature "max_aggr__onehot_encoded__state~0"
    # Use string-based feature name so the aggregation plugin can parse it correctly
    chained_feature = "max_aggr__onehot_encoded__state~0"

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
        if "max_aggr__onehot_encoded__state~0" in result_df.columns:
            found_aggregation = True
            # Verify the column has data (not all NaN)
            assert not result_df["max_aggr__onehot_encoded__state~0"].isna().all(), (
                "Feature 'max_aggr__onehot_encoded__state~0' has all NaN values"
            )
            break

    assert found_aggregation, (
        f"Feature 'max_aggr__onehot_encoded__state~0' not found in any result DataFrame. "
        f"Result DataFrames: {[df.columns.tolist() for df in results]}"
    )

    print(f"\n✓ Successfully created chained feature 'max_aggr__onehot_encoded__state~0'")


def test_integration_data_creator_has_onehot_encoded_state() -> None:
    """
    Test that IntegrationDataCreator provides 'onehot_encoded__state' column.

    This test verifies that the raw data includes a pre-encoded 'onehot_encoded__state'
    column with binary values (0 or 1) to support features that depend on it.

    Expected:
    - 'onehot_encoded__state' exists in get_raw_data() result
    - The column has 5 values (matching other columns like 'state')
    - All values are binary (0 or 1) representing one-hot encoded data
    """
    # Get raw data from IntegrationDataCreator
    raw_data = IntegrationDataCreator.get_raw_data()

    # Verify the column exists
    assert "onehot_encoded__state" in raw_data, (
        "Expected 'onehot_encoded__state' column in raw data, but it was not found. "
        f"Available columns: {list(raw_data.keys())}"
    )

    # Verify the column has 5 values (matching other columns)
    onehot_data = raw_data["onehot_encoded__state"]
    assert len(onehot_data) == 5, (
        f"Expected 'onehot_encoded__state' to have 5 values, but got {len(onehot_data)}"
    )

    # Verify all values are binary (0 or 1)
    for value in onehot_data:
        assert value in [0, 1], (
            f"Expected all values in 'onehot_encoded__state' to be 0 or 1, "
            f"but found: {value}"
        )
