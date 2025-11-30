"""
Simple demonstration of encoding + scaling chaining.
"""

import pytest
from typing import Any, Dict

from mloda_core.abstract_plugins.plugin_loader.plugin_loader import PluginLoader
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.experimental.sklearn.encoding.pandas import PandasEncodingFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.scaling.pandas import PandasScalingFeatureGroup
from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator


class SimpleChainTestDataCreator(ATestDataCreator):
    """Test data creator for simple chaining tests."""

    compute_framework = PandasDataframe

    @classmethod
    def get_raw_data(cls) -> Dict[str, Any]:
        """Return the raw data as a dictionary."""
        return {
            "customer_id": [1, 2, 3, 4, 5],
            "category": ["Premium", "Standard", "Basic", "Premium", "Standard"],
            "sales": [1000, 500, 250, 1200, 600],
        }


class TestSimpleChaining:
    """Simple chaining test to understand the issue."""

    def test_step_by_step_chaining(self) -> None:
        """Test chaining step by step to understand the issue."""
        # Skip test if sklearn not available
        try:
            import sklearn
        except ImportError:
            pytest.skip("scikit-learn not available")

        PluginLoader().all()

        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {SimpleChainTestDataCreator, PandasEncodingFeatureGroup, PandasScalingFeatureGroup}
        )

        # Step 1: Create OneHot encoding first
        print("Step 1: Creating OneHot encoding...")
        # L→R: category__onehot_encoded~0__standard_scaled
        onehot_feature = Feature("category__onehot_encoded~0__standard_scaled")
        api1 = mlodaAPI([onehot_feature], {PandasDataframe}, plugin_collector=plugin_collector)
        api1._batch_run()
        results1 = api1.get_result()
        df1 = results1[0]

        print("Columns after OneHot encoding:")
        print(list(df1.columns))
        print("Data:")
        print(df1)
