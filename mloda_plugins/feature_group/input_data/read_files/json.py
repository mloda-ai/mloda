from typing import Any, Tuple

from pyarrow import json as pyarrow_json

from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_plugins.feature_group.input_data.read_file import ReadFile


class JsonReader(ReadFile):
    @classmethod
    def suffix(cls) -> Tuple[str, ...]:
        return (
            ".json",
            ".JSON",
        )

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        result = pyarrow_json.read_json(
            data_access,
            parse_options=pyarrow_json.ParseOptions(
                explicit_schema=None,
                unexpected_field_behavior="error",
            ),
        )
        return result.select(list(features.get_all_names()))

    @classmethod
    def get_column_names(cls, file_name: str) -> Any:
        # Read only the first batch of rows to infer the schema
        read_options = pyarrow_json.ReadOptions(block_size=65536)  # Reads a small sample
        table = pyarrow_json.read_json(file_name, read_options=read_options)
        return table.schema.names
