import json

from mloda.provider import ReadDocumentFG
from mloda_plugins.feature_group.input_data.file_suffixes import JSON_SUFFIXES


class JsonDocumentFG(ReadDocumentFG):
    @classmethod
    def suffixes(cls) -> tuple[str, ...]:
        return JSON_SUFFIXES

    @classmethod
    def handover_suffixes(cls) -> tuple[str, ...]:
        return JSON_SUFFIXES

    @classmethod
    def read_text(cls, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return json.dumps(json.load(f))
