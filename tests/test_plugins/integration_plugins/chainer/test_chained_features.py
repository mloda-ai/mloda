from typing import Any, Dict

from mloda_core.abstract_plugins.components.feature import Feature
import pytest

from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from tests.test_plugins.integration_plugins.chainer.chainer_test_feature import (
    ChainedFeatureGroupTest,
    ChainedFeatureGroupTest_B,
)
from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator


class ChainerParserTestDataCreator(ATestDataCreator):
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
        {ChainedFeatureGroupTest, ChainerParserTestDataCreator, ChainedFeatureGroupTest_B}
    )

    def test_chained_features(self) -> None:
        feature = Feature(f"identifier1{ChainedFeatureGroupTest.PATTERN}Sales")
        feature2 = Feature(
            f"identifier2{ChainedFeatureGroupTest_B.PATTERN}identifier1{ChainedFeatureGroupTest.PATTERN}Sales",
        )

        result = mlodaAPI.run_all(
            [
                feature,
                f"identifier2{ChainedFeatureGroupTest.PATTERN}Sales",
                feature2,
                f"identifier2{ChainedFeatureGroupTest_B.PATTERN}identifier2{ChainedFeatureGroupTest.PATTERN}Sales",
            ],
            compute_frameworks={PandasDataframe},
            plugin_collector=self.plugin_collector,
        )
        # Currently, we duplicate here the data. This can be changed in the future.
        assert len(result) == 2

    def test_invalid_prefix_configuration(self) -> None:
        with pytest.raises(Exception) as exc_info:
            mlodaAPI.run_all(
                [f"invalid_prefix{ChainedFeatureGroupTest.PATTERN}Sales"],
                compute_frameworks={PandasDataframe},
                plugin_collector=self.plugin_collector,
            )
        assert "invalid_prefix" in str(exc_info.value)
