from typing import Any, Tuple

from pyarrow import parquet as pyarrow_parquet

from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_plugins.feature_group.input_data.read_file import ReadFile


class ParquetReader(ReadFile):
    @classmethod
    def suffix(cls) -> Tuple[str, ...]:
        return (
            ".parquet",
            ".PARQUET",
            ".pqt",
            ".PQT",
        )

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return pyarrow_parquet.read_table(data_access, columns=list(features.get_all_names()))

    @classmethod
    def get_column_names(cls, file_name: str) -> Any:
        parquet_file = pyarrow_parquet.ParquetFile(file_name)
        return [column.name for column in parquet_file.schema]
