from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.options import Options
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from tests.test_plugins.integration_plugins.chainer.chainer_context_feature import (
    ChainedContextFeatureGroupTest,
)
from tests.test_plugins.integration_plugins.chainer.context.test_chained_optional_features import (
    ChainerContextParserTestDataCreator,
)


class TestChainedFeatures:
    plugin_collector = PlugInCollector.enabled_feature_groups(
        {ChainedContextFeatureGroupTest, ChainerContextParserTestDataCreator}
    )

    def test_context_chained_features(self) -> None:
        """Test context/group separation in chained features."""

        # Test 1: Default behavior - context parameters go to context, group parameters to group
        feature1 = Feature(
            name="placeholder1",
            options=Options(
                group={
                    "property2": "value1",  # Group parameter by default (not marked as context)
                },
                context={
                    DefaultOptionKeys.in_features: "Sales",  # Context by default
                    "ident": "identifier1",  # Context by default
                    "property3": "opt_val1",  # Context by default
                },
            ),
        )

        # Test 2: Same group parameters, different context parameters - should resolve together
        feature2 = Feature(
            name="placeholder2",
            options=Options(
                group={
                    "property2": "value2",  # Same group parameter as feature1
                },
                context={
                    DefaultOptionKeys.in_features: feature1,
                    "ident": "identifier1",  # Different context parameter (shouldn't affect resolution)
                    # property3 omitted (optional context parameter)
                },
            ),
        )

        # Test 3: User override - force context parameter into group
        feature3 = Feature(
            name="placeholder3",
            options=Options(
                group={
                    "property2": "value1",  # Same group parameter
                    "ident": "identifier1",  # USER OVERRIDE: context parameter forced to group
                },
                context={
                    DefaultOptionKeys.in_features: frozenset([feature2]),
                    "property3": "opt_val2",  # Context parameter
                },
            ),
        )

        # Test 4: Different group parameters - should resolve separately
        feature4 = Feature(
            name="placeholder4",
            options=Options(
                group={
                    "property2": "value1",  # Different group parameter
                },
                context={
                    DefaultOptionKeys.in_features: frozenset([feature2]),
                    "ident": "identifier2",  # Context parameter
                    # property3 omitted (optional)
                },
            ),
        )

        # Test resolution behavior
        result = mlodaAPI.run_all(
            [
                feature4,
                feature3,
                feature2,
                feature1,
                "Sales",
            ],
            compute_frameworks={PandasDataFrame},
            plugin_collector=self.plugin_collector,
        )

        # Should have 5 results: Sales + 4 features
        assert len(result) == 5

        feature5 = Feature(
            name="placeholder4_5",
            options=Options(
                group={
                    "property2": "specific_val_3_test",  # Different group parameter
                },
                context={
                    DefaultOptionKeys.in_features: frozenset([feature2]),
                    "ident": "identifier2",  # Context parameter
                },
            ),
        )

        # This feature should be the same es feature 4, but with different context parameter
        # This means it should be resolved in the same feature group as 4, but with diffrent outcome
        feature6 = Feature(
            name="placeholder4_6",
            options=Options(
                group={
                    "property2": "value1",  # Different group parameter
                },
                context={
                    DefaultOptionKeys.in_features: frozenset([feature2]),
                    "ident": "identifier1",  # Context parameter
                    # property3 omitted (optional)
                },
            ),
        )

        result = mlodaAPI.run_all(
            [feature4, feature5, feature6],
            compute_frameworks={PandasDataFrame},
            plugin_collector=self.plugin_collector,
        )

        assert len(result) == 2, "Expected 2 result datasets for feature4 and feature5/6"

        for res in result:
            if "placeholder4" in res:
                assert all(res["placeholder4"].values != res["placeholder4_6"].values), "Feature 4 and 6 should differ"
                assert "placeholder4_5" not in res
            else:
                assert "placeholder4_5" in res, "Feature 5 should be present"
