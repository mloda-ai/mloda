from typing import Any, Optional

from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.provider import BaseInputData
from mloda_plugins.feature_group.input_data.read_file import ReadFile


class ReadFileFeature(FeatureGroup):
    """Load features from file-based data sources (CSV, Parquet, JSON, etc.).

    This is the default input FeatureGroup for file reads. It delegates to
    ``ReadFile``, which selects a reader based on the file extension provided
    in the feature options.
    """

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return ReadFile()

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        reader = cls.input_data()
        if reader is None:
            raise ValueError(f"No reader available for feature {features.get_name_of_one_feature()}.")

        data = reader.load(features)
        return data
