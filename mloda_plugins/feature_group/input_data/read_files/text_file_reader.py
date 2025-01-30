from copy import deepcopy
from typing import Any, Tuple

from mloda_plugins.feature_group.input_data.read_file import ReadFile
from pyarrow import fs as pyarrow_fs

from mloda_core.abstract_plugins.components.feature_set import FeatureSet

try:
    import pandas as pd
except ImportError:
    pd = None


class TextFileReader(ReadFile):
    @classmethod
    def suffix(cls) -> Tuple[str, ...]:
        return (".text",)

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        if pd is None:
            raise ImportError("pandas is not installed.")

        file_path = deepcopy(features.get_options_key(cls.__name__))
        local_fs = pyarrow_fs.LocalFileSystem()

        with local_fs.open_input_file(file_path) as file:
            content = file.read().decode("utf-8")

        data = {cls.get_class_name(): [content]}

        return pd.DataFrame(data)


class PyFileReader(TextFileReader):
    @classmethod
    def suffix(cls) -> Tuple[str, ...]:
        return (".py",)
