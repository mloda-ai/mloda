from mloda.provider import ReadDocumentFG
from mloda_plugins.feature_group.input_data.file_suffixes import MARKDOWN_SUFFIXES


class MarkdownFG(ReadDocumentFG):
    @classmethod
    def suffixes(cls) -> tuple[str, ...]:
        return MARKDOWN_SUFFIXES
