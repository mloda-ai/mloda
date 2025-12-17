from typing import Any, Optional, Set
from mloda import FeatureGroup
from mloda import Feature
from mloda.user import FeatureName
from mloda.provider import FeatureSet
from mloda import Options
import mloda

from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame


class NFeatureNameBase(FeatureGroup):
    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"NFeatureNameBase~1": [1, 2, 3], "NFeatureNameBase~2": [1, 2, 3]}


class NFeatureConsumer(FeatureGroup):
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
        result = mloda.run_all(
            ["NFeatureNameBase"],
            compute_frameworks=["PyArrowTable"],
        )

        for k, v in result[0].to_pydict().items():
            assert k == "NFeatureNameBase~1" or k == "NFeatureNameBase~2"
            assert len(v) == 3
            assert v[0] == 1 or v[0] == 2 or v[0] == 3
            assert v[1] == 1 or v[1] == 2 or v[1] == 3
            assert v[2] == 1 or v[2] == 2 or v[2] == 3

    def test_n_feature_as_input(self) -> None:
        # Run the API with NFeatureConsumer, which depends on NFeatureNameBase
        result = mloda.run_all(["NFeatureConsumer"], compute_frameworks={PandasDataFrame})

        # Verify the results
        res = result[0]
        assert list(res["NFeatureConsumer"].values) == [2, 4, 6]
