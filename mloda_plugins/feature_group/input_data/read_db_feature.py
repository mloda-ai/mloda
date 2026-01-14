from typing import Any, Optional

from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.provider import BaseInputData
from mloda_plugins.feature_group.input_data.read_db import ReadDB


class ReadDBFeature(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return ReadDB()

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        reader = cls.input_data()
        if reader is not None:
            return reader.load(features)
        raise ValueError(f"Reading file failed for feature {features.get_name_of_one_feature()}.")
