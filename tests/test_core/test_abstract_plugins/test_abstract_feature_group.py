from typing import Optional, Set, Union
from mloda_core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda_core.abstract_plugins.components.data_types import DataType
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.options import Options


class BaseTestFeatureGroup1(AbstractFeatureGroup):
    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        if "BaseTestFeature" in feature_name.name and "1" in feature_name.name:  # type: ignore
            return True
        return False


class BaseTestFeatureGroup2(AbstractFeatureGroup):
    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        if "BaseTestFeature" in feature_name.name and "2" in feature_name.name:  # type: ignore
            return True
        return False

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """This function should return the input features for the feature group
        if this feature is dependent on other features.
        Else, return None"""
        return {Feature.str_of("BaseTestFeature1"), Feature.int32_of("BaseTestFeature1")}

    @classmethod
    def return_data_type_rule(cls, feature: Feature) -> Optional[DataType]:
        if "BaseTestFeature" in feature.name and "2" in feature.name:
            return DataType.STRING
        return None
