from pathlib import Path
from typing import Any, List, Optional, Tuple

from mloda.provider import BaseInputData, FeatureSet
from mloda.user import Options


class ReadDocument(BaseInputData):
    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        raise NotImplementedError

    @classmethod
    def suffix(cls) -> Tuple[str, ...]:
        raise NotImplementedError

    def load(self, features: FeatureSet) -> Any:
        _options = None

        for feature in features.features:
            if _options:
                if _options != feature.options:
                    raise ValueError("All features must have the same options.")
            _options = feature.options

        reader, data_access = self.init_reader(_options)
        data = reader.load_data(data_access, features)

        if data is None:
            raise ValueError(f"Loading data failed for feature {features.get_name_of_one_feature()}.")

        return data

    def init_reader(self, options: Optional[Options]) -> Tuple["ReadDocument", Any]:
        if options is None:
            raise ValueError("Options were not set.")

        reader_data_access = options.get("BaseInputData")

        if reader_data_access is None:
            raise ValueError("Reader data access was not set.")

        reader, data_access = reader_data_access
        return reader(), data_access

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: List[str]) -> Any:
        if isinstance(data_access, (str, Path)):
            return data_access
        return None
