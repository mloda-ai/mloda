from mloda.provider import ReadDocumentFG
from mloda_plugins.feature_group.input_data.file_suffixes import TEXT_SUFFIXES, PY_SUFFIXES


class TextFG(ReadDocumentFG):
    @classmethod
    def suffixes(cls) -> tuple[str, ...]:
        return TEXT_SUFFIXES


class PyFG(ReadDocumentFG):
    @classmethod
    def suffixes(cls) -> tuple[str, ...]:
        return PY_SUFFIXES
