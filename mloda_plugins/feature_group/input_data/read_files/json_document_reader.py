import json
from typing import Any

from mloda_plugins.feature_group.input_data.read_document import ReadDocument


class JsonDocumentReader(ReadDocument):
    """Load entire JSON file as a single document value for RAG pipelines."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".json", ".JSON")

    @classmethod
    def produce_document(cls, file_path: str) -> Any:
        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)
        return json.dumps(content)
