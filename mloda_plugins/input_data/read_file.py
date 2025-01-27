import os
from pathlib import Path
from typing import Any, List, Optional, Tuple
from mloda_core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda_core.abstract_plugins.components.options import Options


class ReadFile(BaseInputData):
    """
    ReadFile is responsible for loading and processing input data files.

    This class should be inherited by all classes that are responsible for reading files.

    The following methods should be implemented in the child classes:
    - load_data
    - suffix
    - get_column_names

    If get_column_names is not implemented, the class will assume the columns are there.
    """

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        """
        This function should be implemented from child classes.
        """
        raise NotImplementedError

    @classmethod
    def suffix(cls) -> Tuple[str, ...]:
        raise NotImplementedError

    @classmethod
    def get_column_names(cls, file_name: str) -> List[str]:
        raise NotImplementedError

    def load(self, features: FeatureSet) -> Any:
        reader, data_access = self.init_reader(features.options)
        data = reader.load_data(data_access, features)

        if data is None:
            raise ValueError(f"Loading data failed for feature {features.get_name_of_one_feature()}.")

        return data

    def init_reader(self, options: Optional[Options]) -> Tuple["ReadFile", Any]:
        if options is None:
            raise ValueError("Options were not set.")

        reader_data_access = options.get("BaseInputData")

        if reader_data_access is None:
            raise ValueError("Reader data access was not set.")

        reader, data_access = reader_data_access
        return reader(), data_access

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: List[str]) -> Any:
        if isinstance(data_access, DataAccessCollection):
            data_accesses = list(data_access.files | data_access.folders)
        elif isinstance(data_access, str):
            data_accesses = [data_access]
        elif isinstance(data_access, Path):
            data_accesses = [str(data_access)]
        else:
            return False

        matched_data_access = cls.match_read_file_data_access(data_accesses, feature_names)
        if matched_data_access is False:
            return False
        return matched_data_access

    @classmethod
    def match_read_file_data_access(cls, data_accesses: List[str], feature_names: List[str]) -> Any:
        for data_access in data_accesses:
            # We assume that the given path contains the feature name.
            if data_access.endswith(cls.suffix()):
                if cls.validate_columns(data_access, feature_names) is False:
                    continue

                return data_access

            # We assume that the given path is a folder containing the feature name.
            if os.path.isdir(data_access):
                for file in os.listdir(data_access):
                    if file.endswith(cls.suffix()):
                        file_name = os.path.join(data_access, file)

                        if cls.validate_columns(file_name, feature_names) is False:
                            continue

                        return file_name
        return False

    @classmethod
    def validate_columns(cls, file_name: str, feature_names: List[str]) -> bool:
        try:
            columns = cls.get_column_names(file_name)
        except NotImplementedError:
            return True

        for feature in feature_names:
            if feature not in columns:
                return False
        return True
