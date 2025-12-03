from typing import Any, Dict, List

from mloda_core.abstract_plugins.components.link import JoinType
import pandas as pd

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys
from mloda_plugins.feature_group.experimental.dynamic_feature_group_factory.dynamic_feature_group_factory import (
    DynamicFeatureGroupCreator,
)
from mloda_plugins.feature_group.experimental.source_input_feature import (
    SourceInputFeature,
)
from mloda_core.abstract_plugins.components.input_data.creator.data_creator import DataCreator


class TestDynamicFeatureGroupFactoryIntegration:
    def test_dynamic_feature_group_integration_simple_calculation(self) -> None:
        # 1. Define Dynamic Feature Group Properties
        properties: Dict[str, Any] = {
            "calculate_feature": lambda cls, data, features: pd.DataFrame({"input_feature": [5]}),
            "match_feature_group_criteria": lambda cls, feature_name, options, data_access_collection: feature_name
            == FeatureName("input_feature"),
            "input_data": lambda: DataCreator({"input_feature"}),
            "compute_framework_rule": lambda: {PandasDataFrame},
        }

        # 2. Create the Dynamic Feature Group
        DynamicFeatureGroupCreator.create(properties, class_name="DynamicTestFeatureGroupSimpleCalc")

        # 3. Define the Features
        features: List[Feature | str] = [
            Feature(name="input_feature"),
        ]

        # 4. Run mlodaAPI with the Dynamic Feature Group
        result = mlodaAPI.run_all(features=features)

        # 5. Verification
        assert result
        assert len(result) == 1
        assert isinstance(result[0], pd.DataFrame)
        assert result[0].iloc[0, 0] == 5

    def test_dynamic_feature_group_integration_source_input_feature_composite(self) -> None:
        """
        Test case for creating a dynamic feature group using SourceInputFeatureComposite,
        where an aggregated feature is created from two source features.
        """

        # 1. Create DataCreators for Source Features
        class SourceFeature1(AbstractFeatureGroup):
            @classmethod
            def input_data(cls) -> BaseInputData | None:
                return DataCreator({"source_feature_1"})

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return pd.DataFrame({"idx": [1], "source_feature_1": [2]})

        class SourceFeature2(AbstractFeatureGroup):
            @classmethod
            def input_data(cls) -> BaseInputData | None:
                return DataCreator({"source_feature_2"})

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return pd.DataFrame({"idx": [1], "source_feature_2": [5]})

        # 2. Define Dynamic Feature Group Properties
        properties: Dict[str, Any] = {
            "calculate_feature": lambda cls, data, features: pd.DataFrame(
                {"AggregatedFeature": [data["source_feature_1"].iloc[0] + data["source_feature_2"].iloc[0]]}
            ),
            "compute_framework_rule": lambda: {PandasDataFrame},
        }

        # 3. Create the Dynamic Feature Group
        DynamicFeatureGroupCreator.create(
            properties, class_name="AggregatedFeature", feature_group_cls=SourceInputFeature
        )

        # 4. Define the Features with Source Definitions
        features: List[Feature] = [
            Feature(
                name="AggregatedFeature",
                options={
                    DefaultOptionKeys.in_features: frozenset(
                        [
                            (
                                "source_feature_1",
                                None,
                                None,
                                (SourceFeature1, "idx"),
                                (SourceFeature2, "idx"),
                                JoinType.OUTER,
                                None,
                            ),
                            ("source_feature_2", None, None, None, None, None, None),
                        ]
                    )
                },
            )
        ]

        # 5. Run mlodaAPI with the Dynamic Feature Group
        result = mlodaAPI.run_all(features=features, compute_frameworks={PandasDataFrame})  # type: ignore

        # 6. Verification
        assert result
        assert len(result) == 1
        assert isinstance(result[0], pd.DataFrame)
        assert result[0].iloc[0, 0] == 7
