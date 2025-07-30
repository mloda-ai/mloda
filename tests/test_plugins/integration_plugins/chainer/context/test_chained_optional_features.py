from typing import Any, Dict

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from tests.test_plugins.integration_plugins.chainer.chainer_context_feature import (
    ChainedContextFeatureGroupTest,
)

from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator


class ChainerContextParserTestDataCreator(ATestDataCreator):
    """Test data creator for aggregation parser tests."""

    compute_framework = PandasDataframe

    @classmethod
    def get_raw_data(cls) -> Dict[str, Any]:
        """Return the raw data as a dictionary."""
        return {
            "Sales": [100, 200, 300, 400, 500],
            "Revenue": [1000, 2000, 3000, 4000, 5000],
        }


class TestChainedFeatures:
    plugin_collector = PlugInCollector.enabled_feature_groups(
        {ChainedContextFeatureGroupTest, ChainerContextParserTestDataCreator}
    )

    def test_optional_chained_features(self) -> None:
        feature1 = Feature(
            name="placeholder1",
            options={
                DefaultOptionKeys.mloda_source_feature: "Sales",
                "ident": "identifier1",
                "property2": "value1",
                "property3": "opt_val1",  # Include optional property
            },
        )

        feature2 = Feature(
            name="placeholder2",
            options={
                DefaultOptionKeys.mloda_source_feature: feature1,
                "ident": "identifier2",
                "property2": "value2",
                # property3 omitted - should still work since it's optional
            },
        )

        feature3 = Feature(
            name="placeholder3",
            options={
                DefaultOptionKeys.mloda_source_feature: frozenset([feature2]),
                "ident": "identifier1",
                "property2": "value1",
                "property3": "opt_val2",  # Include optional property
            },
        )

        feature4 = Feature(
            name="placeholder4",
            options={
                DefaultOptionKeys.mloda_source_feature: frozenset([feature2]),
                "ident": "identifier2",
                "property2": "value1",
                # property3 omitted - should still work since it's optional
            },
        )

        result = mlodaAPI.run_all(
            [
                feature4,
                feature3,
                feature2,
                feature1,
                "Sales",
            ],
            compute_frameworks={PandasDataframe},
            plugin_collector=self.plugin_collector,
        )
        # Currently, we duplicate here the data. This can be changed in the future.
        assert len(result) == 5

        result = mlodaAPI.run_all(
            [
                feature4,
            ],
            compute_frameworks={PandasDataframe},
            plugin_collector=self.plugin_collector,
        )
        assert len(result) == 1
