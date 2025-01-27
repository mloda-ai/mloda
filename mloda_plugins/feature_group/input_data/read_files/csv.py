from typing import Any, Tuple

from pyarrow import csv as pyarrow_csv

from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_plugins.feature_group.input_data.read_file import ReadFile


class CsvReader(ReadFile):
    @classmethod
    def suffix(cls) -> Tuple[str, ...]:
        return (
            ".csv",
            ".CSV",
        )

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        result = pyarrow_csv.read_csv(data_access)
        return result.select(list(features.get_all_names()))

    @classmethod
    def get_column_names(cls, file_name: str) -> Any:
        read_options = pyarrow_csv.ReadOptions(skip_rows_after_names=1)
        table = pyarrow_csv.read_csv(file_name, read_options=read_options)
        return table.schema.names
