from typing import Any, Optional
from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.input_data.api.api_input_data import ApiInputData
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData


class ApiInputDataFeature(AbstractFeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return ApiInputData()

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data
