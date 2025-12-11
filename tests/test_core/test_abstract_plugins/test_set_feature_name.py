from typing import Any, Optional, Set, Union
from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.api.request import mlodaAPI


class ATestSetFeatureNameBase(AbstractFeatureGroup):
    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"ATestSetFeatureNameBaseL": [12, 2, 3], "ATestSetFeatureNameBaseR": [1, 2, 3]}

    def set_feature_name(self, config: Options, feature_name: FeatureName) -> FeatureName:
        return FeatureName(self.resolve_name(feature_name.name, config))

    def resolve_name(self, feature_name: str, config: Options) -> str:
        if "1" in feature_name:
            return "ATestSetFeatureNameBaseL"
        if "2" in feature_name:
            return "ATestSetFeatureNameBaseR"
        raise ValueError("No fitting name found.")

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        if isinstance(feature_name, FeatureName):
            feature_name = feature_name.name
        if "ATestSetFeatureNameBase" in feature_name:
            return True
        return False


class ATestSetFeatureNameFeature(AbstractFeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {Feature.int64_of("ATestSetFeatureNameBase1"), Feature.int64_of("ATestSetFeatureNameBase2")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"ATestSetFeatureNameFeature": [12, 2, 3]}


class TestSetFeatureName:
    def test_set_feature_name(self) -> None:
        mlodaAPI.run_all(
            ["ATestSetFeatureNameFeature"],
            compute_frameworks=["PyArrowTable"],
        )
