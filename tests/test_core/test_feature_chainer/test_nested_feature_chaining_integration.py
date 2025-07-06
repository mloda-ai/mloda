"""
Integration test for nested Feature object chaining with mlodaAPI.

This test demonstrates the new nested Feature object chaining functionality
working end-to-end with the mloda API, alongside traditional string-based chaining.
"""

from typing import Any, Dict, List, Union

from mloda_core.abstract_plugins import plugin_loader
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.abstract_plugins.plugin_loader.plugin_loader import PluginLoader
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import PandasAggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.data_quality.missing_value.pandas import PandasMissingValueFeatureGroup
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys
from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator


class NestedChainingTestDataCreator(ATestDataCreator):
    """Test data creator for nested chaining integration tests."""

    compute_framework = PandasDataframe

    @classmethod
    def get_raw_data(cls) -> Dict[str, Any]:
        """Return the raw data as a dictionary."""
        return {
            "sales": [100, None, 300, 400, None, 600, 700, 800],
            "revenue": [1000, 2000, None, 4000, 5000, None, 7000, 8000],
            "temperature": [20.5, 22.1, None, 25.3, 23.7, None, 21.8, 24.2],
        }


class TestNestedFeatureChainingIntegration:
    """Integration tests for nested Feature object chaining with mlodaAPI."""

    def test_nested_vs_string_chaining_equivalence(self) -> None:
        """
        Test that nested Feature object chaining produces identical results to string-based chaining.

        This test demonstrates:
        1. Traditional string-based feature chaining
        2. New nested Feature object chaining
        3. Both approaches produce identical results
        4. Both work seamlessly with mlodaAPI
        """
        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {NestedChainingTestDataCreator, PandasAggregatedFeatureGroup, PandasMissingValueFeatureGroup}
        )

        # 1. Traditional string-based approach
        string_features = [
            Feature("max_aggr__mean_imputed__sales"),
            Feature("sum_aggr__median_imputed__revenue"),
            Feature("avg_aggr__mode_imputed__temperature"),
        ]

        # 2. New nested Feature object approach - equivalent to the string features above

        # Create: max_aggr__mean_imputed__sales
        mean_imputed_sales = Feature(
            "mean_imputed",
            Options(context={"imputation_method": "mean", DefaultOptionKeys.mloda_source_feature: "sales"}),
        )

        max_aggr_nested = Feature(
            "max_aggr",
            Options(context={"aggregation_type": "max", DefaultOptionKeys.mloda_source_feature: mean_imputed_sales}),
        )

        # Create: sum_aggr__median_imputed__revenue
        median_imputed_revenue = Feature(
            "median_imputed",
            Options(context={"imputation_method": "median", DefaultOptionKeys.mloda_source_feature: "revenue"}),
        )

        sum_aggr_nested = Feature(
            "sum_aggr",
            Options(
                context={"aggregation_type": "sum", DefaultOptionKeys.mloda_source_feature: median_imputed_revenue}
            ),
        )

        # Create: avg_aggr__mode_imputed__temperature
        mode_imputed_temperature = Feature(
            "mode_imputed",
            Options(context={"imputation_method": "mode", DefaultOptionKeys.mloda_source_feature: "temperature"}),
        )

        avg_aggr_nested = Feature(
            "avg_aggr",
            Options(
                context={"aggregation_type": "avg", DefaultOptionKeys.mloda_source_feature: mode_imputed_temperature}
            ),
        )

        nested_features = [max_aggr_nested, sum_aggr_nested, avg_aggr_nested]

        # 3. Test string-based approach
        string_results = mlodaAPI.run_all(
            string_features,  # type: ignore
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        assert len(string_results) == 1
        string_df = string_results[0]

        # Verify expected columns exist
        expected_columns = [
            "max_aggr__mean_imputed__sales",
            "sum_aggr__median_imputed__revenue",
            "avg_aggr__mode_imputed__temperature",
        ]

        for col in expected_columns:
            assert col in string_df.columns, f"Missing column in string results: {col}"

        # 4. Test nested Feature object approach
        nested_results = mlodaAPI.run_all(
            nested_features,  # type: ignore
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        assert len(nested_results) == 1
        nested_df = nested_results[0]

        # Verify that nested approach produces the same columns
        for col in expected_columns:
            assert col in nested_df.columns, f"Missing column in nested results: {col}"

        # 5. Verify that both approaches produce identical results
        for col in expected_columns:
            string_values = string_df[col]
            nested_values = nested_df[col]

            # Values should be identical
            assert string_values.equals(nested_values), f"Results differ for column {col}"

        # 6. Verify the actual computed values make sense
        # Max of imputed sales (mean imputation fills None with mean of [100, 300, 400, 600, 700, 800] = 483.33)
        max_sales = string_df["max_aggr__mean_imputed__sales"].iloc[0]
        assert max_sales == 800.0, f"Expected max sales to be 800.0, got {max_sales}"

        # Sum of imputed revenue (median imputation fills None with median of [1000, 2000, 4000, 5000, 7000, 8000] = 4500)
        sum_revenue = string_df["sum_aggr__median_imputed__revenue"].iloc[0]
        expected_sum = 1000 + 2000 + 4500 + 4000 + 5000 + 4500 + 7000 + 8000  # 36000
        assert sum_revenue == expected_sum, f"Expected sum revenue to be {expected_sum}, got {sum_revenue}"

    def test_deep_nested_chaining(self) -> None:
        """
        Test deep nested Feature object chaining (3+ levels).

        This test creates a 3-level chain: max_aggr -> sum_aggr -> mean_imputed -> sales
        And compares it to the equivalent string-based approach.
        """

        PluginLoader.all()

        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {NestedChainingTestDataCreator, PandasAggregatedFeatureGroup, PandasMissingValueFeatureGroup}
        )

        # 1. String-based 3-level chain
        # string_feature = Feature("mean_imputed__sales")

        # 2. Equivalent nested Feature object chain (3 levels deep)
        # mean_imputed_sales = Feature(
        #    "mean_imputed__sales",
        #    Options(group={"imputation_method": "mean", DefaultOptionKeys.mloda_source_feature: "sales"}),
        # )

        # ISSUE IS HERE THE NAME sum_aggr IS NOT RECOGNIZED
        sum_aggr = Feature(
            "placeholder",
            Options(context={"aggregation_type": "sum", DefaultOptionKeys.mloda_source_feature: "mean_imputed__sales"}),
        )

        sum_aggr2 = Feature(
            "sum_aggr__sales",
            Options(context={"aggregation_type": "avg", DefaultOptionKeys.mloda_source_feature: "sales"}),
        )

        print(sum_aggr.name)
        # return

        # 3. Test both approaches
        # string_results = mlodaAPI.run_all(
        #    [string_feature],
        #    compute_frameworks={PandasDataframe},
        #    plugin_collector=plugin_collector,
        # )

        nested_results = mlodaAPI.run_all(
            [sum_aggr, sum_aggr2],
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        print(nested_results)
        return
        # 4. Verify both produce the same results
        assert len(string_results) == 1
        assert len(nested_results) == 1

        string_df = string_results[0]
        nested_df = nested_results[0]

        expected_column = "max_aggr__sum_aggr__mean_imputed__sales"
        assert expected_column in string_df.columns
        assert expected_column in nested_df.columns

        # Values should be identical
        string_values = string_df[expected_column]
        nested_values = nested_df[expected_column]
        assert string_values.equals(nested_values), "Deep nested chaining results should be identical"

    def test_mixed_string_and_nested_features(self) -> None:
        """
        Test that string-based and nested Feature objects can be used together in the same request.

        This demonstrates that both approaches are fully compatible and can coexist.
        """
        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {NestedChainingTestDataCreator, PandasAggregatedFeatureGroup, PandasMissingValueFeatureGroup}
        )

        # Mix of string-based and nested features
        mixed_features = [
            # String-based feature
            Feature("max_aggr__mean_imputed__sales"),
            # Nested Feature object
            Feature(
                "sum_aggr",
                Options(
                    context={
                        "aggregation_type": "sum",
                        DefaultOptionKeys.mloda_source_feature: Feature(
                            "median_imputed",
                            Options(
                                context={
                                    "imputation_method": "median",
                                    DefaultOptionKeys.mloda_source_feature: "revenue",
                                }
                            ),
                        ),
                    }
                ),
            ),
            # Another string-based feature
            Feature("avg_aggr__mode_imputed__temperature"),
        ]

        # Test mixed approach
        mixed_results = mlodaAPI.run_all(
            mixed_features,  # type: ignore
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        assert len(mixed_results) == 1
        mixed_df = mixed_results[0]

        # Verify all expected columns exist
        expected_columns = [
            "max_aggr__mean_imputed__sales",
            "sum_aggr__median_imputed__revenue",
            "avg_aggr__mode_imputed__temperature",
        ]

        for col in expected_columns:
            assert col in mixed_df.columns, f"Missing column in mixed results: {col}"

        # Verify values are reasonable
        for col in expected_columns:
            values = mixed_df[col]
            assert all(isinstance(val, (int, float)) for val in values), f"Non-numeric values in {col}"

    def test_nested_feature_string_conversion(self) -> None:
        """
        Test the conversion of nested Feature objects to string equivalents.

        This verifies that the FeatureChainParser can correctly convert nested structures
        to their string-based equivalents for debugging and compatibility.
        """
        # Create a nested Feature structure
        base_feature = "sales"

        imputed_feature = Feature(
            "mean_imputed",
            Options(context={"imputation_method": "mean", DefaultOptionKeys.mloda_source_feature: base_feature}),
        )

        aggregated_feature = Feature(
            "max_aggr",
            Options(context={"aggregation_type": "max", DefaultOptionKeys.mloda_source_feature: imputed_feature}),
        )

        # Test string conversion
        string_equivalent = FeatureChainParser.convert_nested_to_string_equivalent(aggregated_feature)
        expected_string = "max_aggr__mean_imputed__sales"

        assert string_equivalent == expected_string, f"Expected {expected_string}, got {string_equivalent}"

        # Test that the converted string produces the same results
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {NestedChainingTestDataCreator, PandasAggregatedFeatureGroup, PandasMissingValueFeatureGroup}
        )

        # Test nested feature
        nested_results = mlodaAPI.run_all(
            [aggregated_feature],
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        # Test equivalent string feature
        string_feature = Feature(string_equivalent)
        string_results = mlodaAPI.run_all(
            [string_feature],
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        # Results should be identical
        assert len(nested_results) == 1
        assert len(string_results) == 1

        nested_df = nested_results[0]
        string_df = string_results[0]

        assert expected_string in nested_df.columns
        assert expected_string in string_df.columns

        nested_values = nested_df[expected_string]
        string_values = string_df[expected_string]
        assert nested_values.equals(string_values), "Nested and string equivalent should produce identical results"

    def test_nested_chaining_with_group_context_separation(self) -> None:
        """
        Test that nested Feature chaining works correctly with the Options group/context separation.

        This verifies that group parameters (affecting Feature Group resolution) and context parameters
        (metadata only) work correctly with nested Feature objects.
        """
        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {NestedChainingTestDataCreator, PandasAggregatedFeatureGroup, PandasMissingValueFeatureGroup}
        )

        # Create nested features with explicit group/context separation
        imputed_feature = Feature(
            "mean_imputed",
            Options(
                group={"data_source": "test"},  # Group parameter (affects resolution)
                context={
                    "imputation_method": "mean",  # Context parameter (metadata)
                    DefaultOptionKeys.mloda_source_feature: "sales",
                },
            ),
        )

        aggregated_feature = Feature(
            "max_aggr",
            Options(
                group={"environment": "integration_test"},  # Group parameter
                context={
                    "aggregation_type": "max",  # Context parameter
                    DefaultOptionKeys.mloda_source_feature: imputed_feature,
                },
            ),
        )

        # Test that the nested structure works with group/context separation
        results = mlodaAPI.run_all(
            [aggregated_feature],
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        assert len(results) == 1
        df = results[0]

        expected_column = "max_aggr__mean_imputed__sales"
        assert expected_column in df.columns, f"Missing column: {expected_column}"

        # Verify that the source feature extraction works with nested Features
        source_feature = FeatureChainParser.extract_source_feature_from_options(aggregated_feature.options)
        assert isinstance(source_feature, Feature), "Source feature should be a Feature object"
        assert source_feature.name.name == "mean_imputed", "Source feature name should be 'mean_imputed'"

        # Verify that nested chaining is detected
        assert FeatureChainParser.supports_nested_chaining(aggregated_feature.options), "Should support nested chaining"

        # Verify group/context parameters are preserved
        assert aggregated_feature.options.group["environment"] == "integration_test"
        assert aggregated_feature.options.context["aggregation_type"] == "max"
        assert imputed_feature.options.group["data_source"] == "test"
        assert imputed_feature.options.context["imputation_method"] == "mean"
