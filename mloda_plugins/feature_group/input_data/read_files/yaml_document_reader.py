import yaml
from pathlib import Path
from typing import Any

from mloda_plugins.feature_group.input_data.read_document import ReadDocument


class YamlDocumentReader(ReadDocument):
    """Load entire YAML file as a single document value for RAG pipelines."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".yaml", ".yml")

    @classmethod
    def produce_document(cls, file_path: str) -> Any:
        with open(file_path, "r", encoding="utf-8") as f:
            documents = list(yaml.safe_load_all(f))
        content = documents[0] if len(documents) == 1 else documents
        return yaml.dump(content)

    @classmethod
    def document_file_type(cls, file_path: str) -> str:
        return Path(file_path).suffix.lstrip(".")
