from typing import Any, Optional, Set, Union
from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.api.request import mlodaAPI

from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe


class NFeatureNameBase(AbstractFeatureGroup):
    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"NFeatureNameBase~1": [1, 2, 3], "NFeatureNameBase~2": [1, 2, 3]}


class NFeatureConsumer(AbstractFeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {Feature.not_typed("NFeatureNameBase")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # Access the columns from NFeatureNameBase using the naming convention
        col1 = data["NFeatureNameBase~1"] + data["NFeatureNameBase~2"]

        feature_name = features.get_name_of_one_feature().name
        return {feature_name: col1}


class TestNFeature:
    def test_n_feature_name(self) -> None:
        result = mlodaAPI.run_all(
            ["NFeatureNameBase"],
            compute_frameworks=["PyarrowTable"],
        )

        for k, v in result[0].to_pydict().items():
            assert k == "NFeatureNameBase~1" or k == "NFeatureNameBase~2"
            assert len(v) == 3
            assert v[0] == 1 or v[0] == 2 or v[0] == 3
            assert v[1] == 1 or v[1] == 2 or v[1] == 3
            assert v[2] == 1 or v[2] == 2 or v[2] == 3

    def test_n_feature_as_input(self) -> None:
        # Run the API with NFeatureConsumer, which depends on NFeatureNameBase
        result = mlodaAPI.run_all(["NFeatureConsumer"], compute_frameworks={PandasDataframe})

        # Verify the results
        res = result[0]
        assert list(res["NFeatureConsumer"].values) == [2, 4, 6]
