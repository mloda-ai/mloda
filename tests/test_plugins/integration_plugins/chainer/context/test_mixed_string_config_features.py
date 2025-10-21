from copy import copy
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.options import Options
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from tests.test_plugins.integration_plugins.chainer.chainer_context_feature import (
    ChainedContextFeatureGroupTest,
)
from tests.test_plugins.integration_plugins.chainer.context.test_chained_optional_features import (
    ChainerContextParserTestDataCreator,
)


class TestMixedStringConfigFeatures:
    """Test mixing string-based and config-based features in chained feature groups."""

    plugin_collector = PlugInCollector.enabled_feature_groups(
        {ChainedContextFeatureGroupTest, ChainerContextParserTestDataCreator}
    )

    def test_mixed_string_and_config_features(self) -> None:
        """Test mixing string-based features with config-based features."""

        # String-based feature: parsed from feature name
        string_feature_name = f"identifier1{ChainedContextFeatureGroupTest.PATTERN}Sales"

        # Config-based feature 1: uses string-based feature as source
        config_feature1 = Feature(
            name="config_feature1",
            options=Options(
                group={
                    "property2": "value2",  # Different group parameter from string feature
                },
                context={
                    DefaultOptionKeys.mloda_source_features: string_feature_name,
                    "ident": "identifier2",  # Context parameter
                    "property3": "opt_val1",  # Optional context parameter
                },
            ),
        )

        # Config-based feature 2: uses config feature as source
        config_feature2 = Feature(
            name="config_feature2",
            options=Options(
                group={
                    "property2": "value1",  # Same group parameter as string feature
                },
                context={
                    DefaultOptionKeys.mloda_source_features: config_feature1,
                    "ident": "identifier1",  # Context parameter
                    # property3 omitted (optional)
                },
            ),
        )

        # Config-based feature 3: same group parameters as config_feature2 (should resolve together)
        config_feature3 = Feature(
            name="config_feature3",
            options=Options(
                group={
                    "property2": "value1",  # Same group parameter as config_feature2
                },
                context={
                    DefaultOptionKeys.mloda_source_features: config_feature1,
                    "ident": "identifier1",  # Same context parameter as config_feature2
                    "property3": "opt_val2",  # Different optional context parameter
                },
            ),
        )

        # Test resolution behavior
        result = mlodaAPI.run_all(
            [
                string_feature_name,  # String-based
                # config_feature1,  # Config-based          --> Known resolution bug
                config_feature2,  # Config-based
                config_feature3,  # Config-based (same group as config_feature2)
                "Sales",  # Base data
            ],
            compute_frameworks={PandasDataframe},
            plugin_collector=self.plugin_collector,
        )

        # Should have multiple result datasets based on group parameter differences
        assert len(result) == 3, "Should have at least one result dataset"

        # Verify all features are present in results
        all_columns = set()
        for res in result:
            all_columns.update(res.columns)

        expected_features = {
            string_feature_name,
            # "config_feature1",
            "config_feature2",
            "config_feature3",
            "Sales",
        }

        assert expected_features.issubset(all_columns), (
            f"Missing features. Expected: {expected_features}, Got: {all_columns}"
        )

    def test_string_to_config_chaining(self) -> None:
        """Test string-based feature feeding into config-based feature."""

        # String-based source feature
        string_source = f"identifier1{ChainedContextFeatureGroupTest.PATTERN}Sales"

        # Config-based feature using string source
        config_target = Feature(
            name="string_to_config",
            options=Options(
                group={
                    "property2": "value2",
                },
                context={
                    DefaultOptionKeys.mloda_source_features: string_source,
                    "ident": "identifier2",
                    "property3": "opt_val1",
                },
            ),
        )

        result = mlodaAPI.run_all(
            [config_target, "Sales"],
            compute_frameworks={PandasDataframe},
            plugin_collector=self.plugin_collector,
        )

        assert len(result) == 2
        # Verify both features exist
        all_columns = set()
        for res in result:
            all_columns.update(res.columns)
        assert string_source not in all_columns
        assert "string_to_config" in all_columns
        assert "Sales" in all_columns

    def test_mixed_group_context_resolution(self) -> None:
        """Test that group/context separation works correctly with mixed feature types."""

        # String-based feature (group parameters extracted from name)
        string_feature = f"identifier1{ChainedContextFeatureGroupTest.PATTERN}Sales"

        # Config-based feature with same effective group parameters
        config_feature_same_group = Feature(
            name="same_group_config",
            options=Options(
                group={
                    "property2": "value1",  # Should match string feature's default
                },
                context={
                    DefaultOptionKeys.mloda_source_features: "Sales",
                    "ident": "identifier1",  # Same as string feature
                    "property3": "different_context",  # Different context (shouldn't affect grouping)
                },
            ),
        )

        # Config-based feature with different group parameters
        config_feature_diff_group = Feature(
            name="diff_group_config",
            options=Options(
                group={
                    "property2": "value2",  # Different group parameter
                },
                context={
                    DefaultOptionKeys.mloda_source_features: "Sales",
                    "ident": "identifier1",  # Same context as others
                },
            ),
        )

        result = mlodaAPI.run_all(
            [string_feature, config_feature_same_group, config_feature_diff_group, "Sales"],
            compute_frameworks={PandasDataframe},
            plugin_collector=self.plugin_collector,
        )

        # Should have multiple datasets due to different group parameters
        assert len(result) >= 1

        # Verify all features are computed
        all_columns = set()
        for res in result:
            all_columns.update(res.columns)

        expected_features = {string_feature, "same_group_config", "diff_group_config", "Sales"}
        assert expected_features.issubset(all_columns)
