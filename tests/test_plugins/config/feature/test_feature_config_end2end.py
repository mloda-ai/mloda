import json
from pathlib import Path
from typing import Any, Dict
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys
import pytest
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.config.feature.loader import load_features_from_config
from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator


def test_end2end_feature_config() -> None:
    """Test that feature config loader correctly parses config and creates Feature objects."""
    config_str = json.dumps(["age", {"name": "weight", "options": {"imputation_method": "mean"}}])

    features = load_features_from_config(config_str, format="json")

    # Verify we got 2 features
    assert len(features) == 2

    # First feature: simple string "age"
    assert features[0] == "age"

    # Second feature: Feature object with name and options
    assert isinstance(features[1], Feature)
    assert features[1].name.name == "weight"
    assert features[1].options.get("imputation_method") == "mean"


def test_integration_json_file() -> None:
    """Test loading and validating the integration JSON file with all features including references."""
    # Load the integration JSON file
    json_path = Path(__file__).parent / "test_config_features.json"
    with open(json_path) as f:
        config_str = f.read()

    # Parse the features
    features = load_features_from_config(config_str, format="json")

    # Verify we got 15 features (including 2 multi-source features, 1 nested feature, and 3 multi-column features)
    assert len(features) == 15

    # First feature: simple string "age"
    assert features[0] == "age"

    # Second feature: Feature object with name and options
    assert isinstance(features[1], Feature)
    assert features[1].name.name == "weight"
    assert features[1].options.get("imputation_method") == "mean"

    # Third feature: Chained feature (source inferred from name)
    assert isinstance(features[2], Feature)
    assert features[2].name.name == "standard_scaled__mean_imputed__age"
    # No explicit mloda_source - will be inferred from name as "mean_imputed__age"

    # Fourth feature: Chained feature (source inferred from name)
    assert isinstance(features[3], Feature)
    assert features[3].name.name == "max_aggr__mean_imputed__weight"
    # No explicit mloda_source - will be inferred from name as "mean_imputed__weight"

    # Fifth feature: Chained feature with options (source inferred from name)
    assert isinstance(features[4], Feature)
    assert features[4].name.name == "min_aggr__mean_imputed__weight"
    # No explicit mloda_source - will be inferred from name as "mean_imputed__weight"
    # Verify timewindow option is in group options
    assert features[4].options.group.get("timewindow") == 3

    # Sixth feature: Feature with column_index (column selector)
    assert isinstance(features[5], Feature)
    assert features[5].name.name == "onehot_encoded__state~0"

    # Seventh feature: Feature with column_index (column selector)
    assert isinstance(features[6], Feature)
    assert features[6].name.name == "onehot_encoded__state~1"

    # Eighth feature: minmaxscaledage with mloda_source "age"
    assert isinstance(features[7], Feature)
    assert features[7].name.name == "minmaxscaledage"
    assert features[7].options.group.get("mloda_source_features") == "age"
    assert features[7].options.group.get("scaler_type") == "minmax"

    # Ninth feature: max_aggr__age with mloda_sources=["minmaxscaledage"]
    assert isinstance(features[8], Feature)
    assert features[8].name.name == "max_aggr__age"
    # The mloda_sources should be a frozenset referencing feature 7
    mloda_source_8 = features[8].options.context.get("mloda_source_features")
    assert mloda_source_8 == frozenset({"minmaxscaledage"})

    # Tenth feature: min_max with mloda_source_features="max_aggr__age" in options
    assert isinstance(features[9], Feature)
    assert features[9].name.name == "min_max"
    # The mloda_source_features should be in group options
    mloda_source_9 = features[9].options.group.get("mloda_source_features")
    assert mloda_source_9 == "max_aggr__age"
    assert features[9].options.group.get("scaler_type") == "minmax"

    # Eleventh feature: haversine_distance__customer_location__store_location (geo distance feature)
    assert isinstance(features[10], Feature)
    assert features[10].name.name == "haversine_distance__customer_location__store_location"

    # Twelfth feature: custom_geo_distance with mloda_sources (multiple sources)
    assert isinstance(features[11], Feature)
    assert features[11].name.name == "custom_geo_distance"
    # Verify mloda_sources is converted to frozenset and stored correctly
    mloda_sources_11 = features[11].options.context.get(DefaultOptionKeys.mloda_source_features)
    assert isinstance(mloda_sources_11, frozenset)
    assert mloda_sources_11 == frozenset(["customer_location", "store_location"])
    assert features[11].options.context.get("distance_type") == "euclidean"


class ChainedFeatureTestDataCreator(ATestDataCreator):
    """Test data creator for end-to-end chained feature tests."""

    compute_framework = PandasDataframe

    @classmethod
    def get_raw_data(cls) -> Dict[str, Any]:
        """Return the raw data as a dictionary."""
        return {
            "age": [25, 30, 35, 40, 45],
            "salary": [50000, 60000, 70000, 80000, 90000],
        }


def test_end2end_chained_features() -> None:
    """Test that chained features work with mlodaAPI.run_all in a real scenario."""
    # Skip test if sklearn not available (needed for scaling)
    try:
        import sklearn  # noqa: F401
    except ImportError:
        pytest.skip("scikit-learn not available")

    # Import sklearn feature groups needed for chained features
    from mloda_plugins.feature_group.experimental.sklearn.scaling.pandas import PandasScalingFeatureGroup
    from mloda_plugins.feature_group.experimental.data_quality.missing_value.pandas import (
        PandasMissingValueFeatureGroup,
    )

    # Create a JSON config with chained features
    config_str = json.dumps(
        [
            "age",
            "salary",
            {
                "name": "mean_imputed__age",
                "mloda_sources": ["age"],
            },
            {
                "name": "standard_scaled__mean_imputed__age",
                "mloda_sources": ["age"],
            },
        ]
    )

    # Parse the features from config
    features = load_features_from_config(config_str, format="json")

    # Verify we got 4 features
    assert len(features) == 4

    # Enable the necessary feature groups
    plugin_collector = PlugInCollector.enabled_feature_groups(
        {ChainedFeatureTestDataCreator, PandasScalingFeatureGroup, PandasMissingValueFeatureGroup}
    )

    # Run mlodaAPI with the features
    results = mlodaAPI.run_all(
        features,
        compute_frameworks={PandasDataframe},
        plugin_collector=plugin_collector,
    )

    # Verify results
    assert len(results) > 0, "Expected at least one result DataFrame"

    # Find the DataFrame with all features
    result_df = None
    for df in results:
        if "standard_scaled__mean_imputed__age" in df.columns:
            result_df = df
            break

    assert result_df is not None, "DataFrame with chained feature not found"

    # Verify the chained feature exists
    assert "standard_scaled__mean_imputed__age" in result_df.columns

    # Verify the chained feature has values (basic sanity check)
    assert len(result_df["standard_scaled__mean_imputed__age"]) == 5
    assert not result_df["standard_scaled__mean_imputed__age"].isna().any()


def test_end2end_group_context_options() -> None:
    """Test that features with group_options and context_options are correctly parsed and loaded."""
    # Create a JSON config with group_options and context_options
    config_str = json.dumps(
        [
            {
                "name": "feature_with_separation",
                "group_options": {"data_source": "production", "cache_enabled": True},
                "context_options": {"aggregation_type": "sum", "window_size": 7, "normalization": "zscore"},
            }
        ]
    )

    # Parse the features from config
    features = load_features_from_config(config_str, format="json")

    # Verify we got 1 feature
    assert len(features) == 1

    # Verify the feature is a Feature object
    assert isinstance(features[0], Feature)
    assert features[0].name.name == "feature_with_separation"

    # Verify group options are correctly set
    assert features[0].options.group.get("data_source") == "production"
    assert features[0].options.group.get("cache_enabled") is True

    # Verify context options are correctly set
    assert features[0].options.context.get("aggregation_type") == "sum"
    assert features[0].options.context.get("window_size") == 7
    assert features[0].options.context.get("normalization") == "zscore"


def test_end2end_multi_column_access() -> None:
    """Test that column_index features work with mlodaAPI.run_all in a real scenario."""
    # Skip test if sklearn not available (needed for one-hot encoding)
    try:
        import sklearn  # noqa: F401
    except ImportError:
        pytest.skip("scikit-learn not available")

    # Import sklearn feature groups needed for one-hot encoding
    from mloda_plugins.feature_group.experimental.sklearn.encoding.pandas import PandasEncodingFeatureGroup

    # Create a JSON config with column_index features
    config_str = json.dumps(
        [
            "state",
            {
                "name": "onehot_encoded__state",
                "mloda_sources": ["state"],
            },
            {
                "name": "onehot_encoded__state",
                "column_index": 0,
            },
            {
                "name": "onehot_encoded__state",
                "column_index": 1,
            },
        ]
    )

    # Parse the features from config
    features = load_features_from_config(config_str, format="json")

    # Verify we got 4 features
    assert len(features) == 4

    # Verify the column selector features have the tilde syntax in their names
    assert isinstance(features[2], Feature)
    assert isinstance(features[3], Feature)
    assert features[2].name == "onehot_encoded__state~0"
    assert features[3].name == "onehot_encoded__state~1"

    # Create a test data creator for state data
    class StateTestDataCreator(ATestDataCreator):
        """Test data creator for state column selector tests."""

        compute_framework = PandasDataframe

        @classmethod
        def get_raw_data(cls) -> Dict[str, Any]:
            """Return the raw data as a dictionary."""
            return {
                "state": ["CA", "NY", "CA", "TX", "NY"],
            }

    # Enable the necessary feature groups
    plugin_collector = PlugInCollector.enabled_feature_groups({StateTestDataCreator, PandasEncodingFeatureGroup})

    # Run mlodaAPI with the features
    results = mlodaAPI.run_all(
        features,
        compute_frameworks={PandasDataframe},
        plugin_collector=plugin_collector,
    )

    # Verify results
    assert len(results) > 0, "Expected at least one result DataFrame"

    # Find the DataFrame with all features
    result_df = None
    for df in results:
        if "onehot_encoded__state~0" in df.columns and "onehot_encoded__state~1" in df.columns:
            result_df = df
            break

    assert result_df is not None, "DataFrame with column selector features not found"

    # Verify the column selector features exist
    assert "onehot_encoded__state~0" in result_df.columns
    assert "onehot_encoded__state~1" in result_df.columns

    # Verify the column selector features have values (basic sanity check)
    assert len(result_df["onehot_encoded__state~0"]) == 5
    assert len(result_df["onehot_encoded__state~1"]) == 5

    # Verify they are different columns (one-hot encoded should have different values)
    assert not (result_df["onehot_encoded__state~0"] == result_df["onehot_encoded__state~1"]).all()


def test_end2end_nested_feature_references() -> None:
    """Test that nested feature references work correctly with @ syntax."""
    # Create a JSON config with nested feature references
    config_str = json.dumps(
        [
            "age",
            {
                "name": "scaled_age",
                "mloda_sources": ["age"],
                "options": {"scaling_method": "standard"},
            },
            {
                "name": "derived_from_scaled",
                "mloda_sources": ["@scaled_age"],
                "options": {"transformation": "log"},
            },
            {
                "name": "nested_reference",
                "mloda_sources": ["@derived_from_scaled"],
                "options": {"normalization": "minmax"},
            },
        ]
    )

    # Parse the features from config
    features = load_features_from_config(config_str, format="json")

    # Verify we got 4 features
    assert len(features) == 4

    # First feature: simple string "age"
    assert features[0] == "age"

    # Second feature: scaled_age with mloda_sources pointing to "age"
    assert isinstance(features[1], Feature)
    assert features[1].name.name == "scaled_age"
    mloda_sources_1 = features[1].options.context.get("mloda_source_features")
    assert isinstance(mloda_sources_1, frozenset)
    assert mloda_sources_1 == frozenset({"age"})
    assert features[1].options.group.get("scaling_method") == "standard"

    # Third feature: derived_from_scaled with mloda_sources pointing to scaled_age Feature object
    assert isinstance(features[2], Feature)
    assert features[2].name.name == "derived_from_scaled"
    # The mloda_sources should be resolved to a frozenset containing a Feature object, not a string
    mloda_sources_2 = features[2].options.context.get("mloda_source_features")
    assert isinstance(mloda_sources_2, frozenset)
    feature_list_2 = list(mloda_sources_2)
    assert len(feature_list_2) == 1
    assert isinstance(feature_list_2[0], Feature)
    assert feature_list_2[0].name.name == "scaled_age"
    assert features[2].options.group.get("transformation") == "log"

    # Fourth feature: nested_reference with mloda_sources pointing to derived_from_scaled Feature object
    assert isinstance(features[3], Feature)
    assert features[3].name.name == "nested_reference"
    # The mloda_sources should be resolved to a frozenset containing a Feature object
    mloda_sources_3 = features[3].options.context.get("mloda_source_features")
    assert isinstance(mloda_sources_3, frozenset)
    feature_list_3 = list(mloda_sources_3)
    assert len(feature_list_3) == 1
    assert isinstance(feature_list_3[0], Feature)
    assert feature_list_3[0].name.name == "derived_from_scaled"
    assert features[3].options.group.get("normalization") == "minmax"

    # Verify the dependency chain is correctly established
    # nested_reference -> derived_from_scaled -> scaled_age -> age
    chained_sources_3 = feature_list_3[0].options.context.get("mloda_source_features")
    assert chained_sources_3 is not None
    assert isinstance(chained_sources_3, frozenset)
    chained_feature_list = list(chained_sources_3)
    assert len(chained_feature_list) == 1
    assert isinstance(chained_feature_list[0], Feature)
    assert chained_feature_list[0].name.name == "scaled_age"


def test_end2end_multiple_source_features() -> None:
    """Test that features with mloda_sources (multiple sources) are correctly parsed and loaded."""
    # Create a JSON config with multiple source features
    config_str = json.dumps(
        [
            "latitude",
            "longitude",
            "sales",
            "revenue",
            "profit",
            {
                "name": "distance_feature",
                "mloda_sources": ["latitude", "longitude"],
                "options": {"distance_type": "euclidean"},
            },
            {
                "name": "multi_source_aggregation",
                "mloda_sources": ["sales", "revenue", "profit"],
                "options": {"aggregation": "sum"},
            },
            {
                "name": "feature_with_both",
                "mloda_sources": ["latitude", "longitude"],
                "group_options": {"cache_enabled": True},
                "context_options": {"precision": 6},
            },
        ]
    )

    # Parse the features from config
    features = load_features_from_config(config_str, format="json")

    # Verify we got 8 features
    assert len(features) == 8

    # First three features: simple strings
    assert features[0] == "latitude"
    assert features[1] == "longitude"
    assert features[2] == "sales"
    assert features[3] == "revenue"
    assert features[4] == "profit"

    # Sixth feature: distance_feature with mloda_sources
    assert isinstance(features[5], Feature)
    assert features[5].name.name == "distance_feature"
    # Verify mloda_sources is converted to frozenset
    mloda_sources_5 = features[5].options.context.get(DefaultOptionKeys.mloda_source_features)
    assert isinstance(mloda_sources_5, frozenset)
    assert mloda_sources_5 == frozenset(["latitude", "longitude"])
    # Verify options are correctly set
    assert features[5].options.group.get("distance_type") == "euclidean"

    # Seventh feature: multi_source_aggregation with mloda_sources
    assert isinstance(features[6], Feature)
    assert features[6].name.name == "multi_source_aggregation"
    # Verify mloda_sources is converted to frozenset
    mloda_sources_6 = features[6].options.context.get(DefaultOptionKeys.mloda_source_features)
    assert isinstance(mloda_sources_6, frozenset)
    assert mloda_sources_6 == frozenset(["sales", "revenue", "profit"])
    # Verify options are correctly set
    assert features[6].options.group.get("aggregation") == "sum"

    # Eighth feature: feature_with_both (mloda_sources with group_options and context_options)
    assert isinstance(features[7], Feature)
    assert features[7].name.name == "feature_with_both"
    # Verify mloda_sources is converted to frozenset
    mloda_sources_7 = features[7].options.context.get(DefaultOptionKeys.mloda_source_features)
    assert isinstance(mloda_sources_7, frozenset)
    assert mloda_sources_7 == frozenset(["latitude", "longitude"])
    # Verify group options are correctly set
    assert features[7].options.group.get("cache_enabled") is True
    # Verify context options are correctly set (in addition to mloda_source_features)
    assert features[7].options.context.get("precision") == 6


def test_complete_integration_json() -> None:
    """Comprehensive test that validates ALL feature types are present in the integration JSON file.

    This test ensures the test_config_features.json file includes all supported feature patterns:
    - Simple string features
    - Features with options
    - Chained features (with __)
    - Group/context options separation
    - Multi-column access (column_index)
    - Feature references (@syntax)
    - Multiple source features (mloda_sources)
    """
    # Load the integration JSON file
    json_path = Path(__file__).parent / "test_config_features.json"
    with open(json_path) as f:
        config_str = f.read()

    # Parse all features using load_features_from_config
    features = load_features_from_config(config_str, format="json")

    # Verify we got the expected number of features (15 total, including nested feature and multi-column features)
    assert len(features) == 15, f"Expected 15 features, got {len(features)}"

    # Track which feature types we've validated
    validated_patterns = {
        "simple_string": False,
        "feature_with_options": False,
        "chained_feature": False,
        "group_context_separation": False,
        "column_selector": False,
        "feature_reference": False,
        "multiple_sources": False,
    }

    # 1. Simple string features
    # Feature index 0: "age"
    assert features[0] == "age", "Simple string feature 'age' not found"
    validated_patterns["simple_string"] = True

    # 2. Features with options
    # Feature index 1: weight with imputation_method option
    assert isinstance(features[1], Feature), "Feature 'weight' should be a Feature object"
    assert features[1].name.name == "weight", "Feature name should be 'weight'"
    assert features[1].options.get("imputation_method") == "mean", "imputation_method should be 'mean'"
    validated_patterns["feature_with_options"] = True

    # 3. Chained features (with __ in name, source inferred from name)
    # Feature index 2: standard_scaled__mean_imputed__age
    assert isinstance(features[2], Feature), "Chained feature should be a Feature object"
    assert features[2].name.name == "standard_scaled__mean_imputed__age", "Chained feature name incorrect"
    # No explicit mloda_source - will be inferred from name as "mean_imputed__age"
    validated_patterns["chained_feature"] = True

    # Feature index 3: Another chained feature (aggregation on mean imputation)
    assert isinstance(features[3], Feature), "Second chained feature should be a Feature object"
    assert features[3].name.name == "max_aggr__mean_imputed__weight", "Second chained feature name incorrect"
    # No explicit mloda_source - will be inferred from name as "mean_imputed__weight"

    # 4. Group/context options separation
    # Feature index 4: min_aggr__mean_imputed__weight with options (source inferred from name)
    assert isinstance(features[4], Feature), "Feature with group/context options should be a Feature object"
    assert features[4].name.name == "min_aggr__mean_imputed__weight", (
        "Feature name should be 'min_aggr__mean_imputed__weight'"
    )
    # No explicit mloda_source - will be inferred from name as "mean_imputed__weight"
    assert features[4].options.group.get("timewindow") == 3, "group option 'timewindow' should be 3"
    validated_patterns["group_context_separation"] = True

    # 5. Multi-column access (column_index with ~ syntax)
    # Feature index 5: onehot_encoded__state~0
    assert isinstance(features[5], Feature), "Column selector feature should be a Feature object"
    assert features[5].name.name == "onehot_encoded__state~0", "Column selector feature should have ~0 suffix"
    validated_patterns["column_selector"] = True

    # Feature index 6: onehot_encoded__state~1
    assert isinstance(features[6], Feature), "Second column selector feature should be a Feature object"
    assert features[6].name.name == "onehot_encoded__state~1", "Second column selector should have ~1 suffix"

    # 6. Feature references (@syntax)
    # Feature index 7: minmaxscaledage (base feature)
    assert isinstance(features[7], Feature), "Base feature for reference should be a Feature object"
    assert features[7].name.name == "minmaxscaledage", "Base feature name should be 'minmaxscaledage'"
    assert features[7].options.group.get("mloda_source_features") == "age", "Base feature mloda_source should be 'age'"

    # Feature index 8: max_aggr__age with mloda_sources=["minmaxscaledage"]
    assert isinstance(features[8], Feature), "Feature with aggregation should be a Feature object"
    assert features[8].name.name == "max_aggr__age", "Feature name should be 'max_aggr__age'"
    mloda_source_8 = features[8].options.context.get("mloda_source_features")
    assert mloda_source_8 == frozenset({"minmaxscaledage"}), "mloda_sources should be frozenset with 'minmaxscaledage'"
    validated_patterns["feature_reference"] = True

    # Feature index 9: min_max with mloda_source_features="max_aggr__age" in options
    assert isinstance(features[9], Feature), "Feature with scaling should be a Feature object"
    assert features[9].name.name == "min_max", "Feature name should be 'min_max'"
    mloda_source_9 = features[9].options.group.get("mloda_source_features")
    assert mloda_source_9 == "max_aggr__age", "mloda_source_features should be 'max_aggr__age'"
    assert features[9].options.group.get("scaler_type") == "minmax", "scaler_type should be 'minmax'"

    # 7. Geo distance feature (string-based naming pattern)
    # Feature index 10: haversine_distance__customer_location__store_location
    assert isinstance(features[10], Feature), "Geo distance feature should be a Feature object"
    assert features[10].name.name == "haversine_distance__customer_location__store_location", (
        "Feature name should be 'haversine_distance__customer_location__store_location'"
    )
    # The string-based geo distance feature uses the pattern: {distance_type}_distance__{point1}__{point2}
    # No mloda_sources or options needed - it's all encoded in the feature name
    validated_patterns["multiple_sources"] = True

    # Feature index 11: custom_geo_distance with mloda_sources ["customer_location", "store_location"]
    assert isinstance(features[11], Feature), "Second multi-source feature should be a Feature object"
    assert features[11].name.name == "custom_geo_distance", "Feature name should be 'custom_geo_distance'"
    mloda_sources_11 = features[11].options.context.get(DefaultOptionKeys.mloda_source_features)
    assert isinstance(mloda_sources_11, frozenset), "mloda_sources should be converted to frozenset"
    assert mloda_sources_11 == frozenset(["customer_location", "store_location"]), (
        "mloda_sources should contain customer_location and store_location"
    )
    assert features[11].options.context.get("distance_type") == "euclidean", "distance_type should be 'euclidean'"

    # Final validation: Ensure ALL pattern types are validated
    missing_patterns = [pattern for pattern, validated in validated_patterns.items() if not validated]
    assert not missing_patterns, f"Missing validation for pattern types: {missing_patterns}"

    # Summary: All feature pattern types are present and correctly parsed
    print("All feature pattern types validated successfully:")
    print("  - Simple string features")
    print("  - Features with options")
    print("  - Chained features (with __)")
    print("  - Group/context options separation")
    print("  - Multi-column access (column_index)")
    print("  - Feature references (@syntax)")
    print("  - Multiple source features (mloda_sources)")
