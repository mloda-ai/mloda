"""End-to-end test for list-valued options in PROPERTY_MAPPING (issue #228).

Verifies that list-valued options pass through the mloda pipeline without
TypeError and that element order is preserved via tuple conversion.
"""

import ast
from typing import Any, Dict, Optional, Set, Type, Union

from mloda.provider import ComputeFramework
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.provider import DataCreator
from mloda.provider import BaseInputData
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Options
from mloda.user import PluginCollector
from mloda.user import mloda
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

import pandas as pd


class ListValuedTestDataCreator(FeatureGroup):
    """Creates test data with three columns."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"col_a", "col_b", "col_c"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame(
            {
                "col_a": [1, 2, 3],
                "col_b": [10, 20, 30],
                "col_c": [100, 200, 300],
            }
        )

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PandasDataFrame}


class ListValuedFeatureGroup(FeatureGroup):
    """Feature group that accepts a list-valued 'columns' option.

    Computes an order-dependent weighted sum:
      result = columns[0]*1 + columns[1]*10 + columns[2]*100
    """

    PROPERTY_MAPPING = {
        "columns": {
            "explanation": "List of columns to combine in order",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source features",
            DefaultOptionKeys.context: True,
        },
    }

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Optional[Any] = None,
    ) -> bool:
        _name = feature_name.name if isinstance(feature_name, FeatureName) else feature_name
        return FeatureChainParser.match_configuration_feature_chain_parser(
            _name,
            options,
            property_mapping=cls.PROPERTY_MAPPING,
        )

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        source_features = options.get_in_features()
        return set(source_features)

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        for feature in features.features:
            columns_raw = feature.options.get("columns")

            if isinstance(columns_raw, str):
                columns = ast.literal_eval(columns_raw)
            elif isinstance(columns_raw, (list, tuple)):
                columns = list(columns_raw)
            else:
                columns = list(columns_raw)

            weights = [1, 10, 100]
            result = data[columns[0]] * weights[0]
            for i in range(1, len(columns)):
                result = result + data[columns[i]] * weights[i]

            data[feature.get_name()] = result
        return data


class TestListValuedOptionsE2E:
    """End-to-end tests for list-valued options through the mloda pipeline."""

    plugin_collector = PluginCollector.enabled_feature_groups({ListValuedTestDataCreator, ListValuedFeatureGroup})

    def test_list_valued_option_order_preserved(self) -> None:
        """List-valued option order is preserved through the pipeline.

        Order [col_a, col_b, col_c] with weights [1, 10, 100] gives:
          row 0: 1*1 + 10*10 + 100*100 = 10101
        Order [col_c, col_b, col_a] with weights [1, 10, 100] gives:
          row 0: 100*1 + 10*10 + 1*100 = 300
        """
        feature_abc = Feature(
            name="weighted_abc",
            options=Options(
                context={
                    DefaultOptionKeys.in_features: "col_a",
                    "columns": ["col_a", "col_b", "col_c"],
                },
            ),
        )

        result = mloda.run_all(
            [feature_abc],
            compute_frameworks={PandasDataFrame},
            plugin_collector=self.plugin_collector,
        )

        assert len(result) >= 1

        for df in result:
            if "weighted_abc" in df.columns:
                abc_values = df["weighted_abc"].tolist()
                # col_a=1, col_b=10, col_c=100 with weights [1, 10, 100]:
                # abc: 1*1 + 10*10 + 100*100 = 10101
                assert abc_values[0] == 10101, f"Expected 10101, got {abc_values[0]}"
                return

        raise AssertionError("weighted_abc not found in results")

    def test_list_valued_option_different_order(self) -> None:
        """Reversed column order produces different results, proving order preservation."""
        feature_cba = Feature(
            name="weighted_cba",
            options=Options(
                context={
                    DefaultOptionKeys.in_features: "col_a",
                    "columns": ["col_c", "col_b", "col_a"],
                },
            ),
        )

        result = mloda.run_all(
            [feature_cba],
            compute_frameworks={PandasDataFrame},
            plugin_collector=self.plugin_collector,
        )

        assert len(result) >= 1

        for df in result:
            if "weighted_cba" in df.columns:
                cba_values = df["weighted_cba"].tolist()
                # col_c=100, col_b=10, col_a=1 with weights [1, 10, 100]:
                # cba: 100*1 + 10*10 + 1*100 = 300
                assert cba_values[0] == 300, f"Expected 300, got {cba_values[0]}"
                return

        raise AssertionError("weighted_cba not found in results")

    def test_list_valued_in_features(self) -> None:
        """in_features passed as a list works through the pipeline."""
        feature = Feature(
            name="weighted_list_in",
            options=Options(
                context={
                    DefaultOptionKeys.in_features: ["col_a", "col_b"],
                    "columns": ["col_a", "col_b", "col_c"],
                },
            ),
        )

        result = mloda.run_all(
            [feature],
            compute_frameworks={PandasDataFrame},
            plugin_collector=self.plugin_collector,
        )

        assert len(result) >= 1

        for df in result:
            if "weighted_list_in" in df.columns:
                values = df["weighted_list_in"].tolist()
                # col_a=1, col_b=10, col_c=100 with weights [1, 10, 100]:
                # 1*1 + 10*10 + 100*100 = 10101
                assert values[0] == 10101, f"Expected 10101, got {values[0]}"
                return

        raise AssertionError("weighted_list_in not found in results")
