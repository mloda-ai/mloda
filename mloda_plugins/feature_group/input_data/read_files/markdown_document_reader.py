from typing import Any

from mloda_plugins.feature_group.input_data.read_document import ReadDocument


class MarkdownDocumentReader(ReadDocument):
    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".md",)

    @classmethod
    def produce_document(cls, file_path: str) -> Any:
        return cls._read_text(file_path)
