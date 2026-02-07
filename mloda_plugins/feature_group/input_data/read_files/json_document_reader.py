import json
from typing import Any, Tuple

from mloda_plugins.feature_group.input_data.read_document import ReadDocument

from mloda.provider import FeatureSet


class JsonDocumentReader(ReadDocument):
    """Load entire JSON file as a single document value for RAG pipelines."""

    @classmethod
    def suffix(cls) -> Tuple[str, ...]:
        return (".json", ".JSON")

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        file_path = features.get_options_key(cls.__name__)

        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)

        json_string = json.dumps(content)
        file_type = cls.suffix()[0].lstrip(".")

        return [{cls.get_class_name(): json_string, "source": file_path, "file_type": file_type}]
