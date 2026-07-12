from typing import Any

from mloda_plugins.feature_group.input_data.read_document import ReadDocument


class TextFileReader(ReadDocument):
    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".text",)

    @classmethod
    def produce_document(cls, file_path: str) -> Any:
        return cls._read_text(file_path)


class PyFileReader(TextFileReader):
    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".py",)
