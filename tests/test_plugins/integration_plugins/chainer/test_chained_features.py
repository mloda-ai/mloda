from typing import Any, Dict

from mloda import Feature
import pytest

from mloda.user import PluginCollector
import mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from tests.test_plugins.integration_plugins.chainer.chainer_test_feature import (
    ChainedFeatureGroupTest,
    ChainedFeatureGroupTest_B,
)
from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator


class ChainerParserTestDataCreator(ATestDataCreator):
    """Test data creator for aggregation parser tests."""

    compute_framework = PandasDataFrame

    @classmethod
    def get_raw_data(cls) -> Dict[str, Any]:
        """Return the raw data as a dictionary."""
        return {
            "Sales": [100, 200, 300, 400, 500],
            "Revenue": [1000, 2000, 3000, 4000, 5000],
        }


class TestChainedFeatures:
    plugin_collector = PluginCollector.enabled_feature_groups(
        {ChainedFeatureGroupTest, ChainerParserTestDataCreator, ChainedFeatureGroupTest_B}
    )

    def test_chained_features(self) -> None:
        feature = Feature(f"Sales__{ChainedFeatureGroupTest.OPERATION_ID}identifier1")
        feature2 = Feature(
            f"Sales__{ChainedFeatureGroupTest.OPERATION_ID}identifier1__{ChainedFeatureGroupTest_B.OPERATION_ID}identifier2",
        )

        result = mloda.run_all(
            [
                feature,
                f"Sales__{ChainedFeatureGroupTest.OPERATION_ID}identifier2",
                feature2,
                f"Sales__{ChainedFeatureGroupTest.OPERATION_ID}identifier2__{ChainedFeatureGroupTest_B.OPERATION_ID}identifier2",
            ],
            compute_frameworks={PandasDataFrame},
            plugin_collector=self.plugin_collector,
        )
        # Currently, we duplicate here the data. This can be changed in the future.
        assert len(result) == 2

    def test_invalid_suffix_configuration(self) -> None:
        with pytest.raises(Exception) as exc_info:
            mloda.run_all(
                [f"Sales__{ChainedFeatureGroupTest.OPERATION_ID}invalid_suffix"],
                compute_frameworks={PandasDataFrame},
                plugin_collector=self.plugin_collector,
            )
        assert "invalid_suffix" in str(exc_info.value)
