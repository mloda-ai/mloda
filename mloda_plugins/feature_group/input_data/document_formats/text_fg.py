from typing import Any, ClassVar

from mloda.core.abstract_plugins.components.input_data.claim_route import SourceMatch
from mloda.provider import ComputeFramework, PropertySpec, ReadDocumentFG
from mloda_plugins.feature_group.input_data.file_suffixes import TEXT_SUFFIXES, PY_SUFFIXES

TEXT_LINES_OPTION = "text_lines"
_SOURCE_SUFFIX = "~source"
_FILE_TYPE_SUFFIX = "~file_type"


class TextFG(ReadDocumentFG):
    PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {
        TEXT_LINES_OPTION: PropertySpec(
            "When true, return one row per non-blank line instead of the whole text in one row.",
            default=False,
            context=True,
        ),
    }

    @classmethod
    def suffixes(cls) -> tuple[str, ...]:
        return TEXT_SUFFIXES

    @classmethod
    def _non_blank_lines(cls, path: str) -> list[str]:
        return [line for line in cls.read_text(path).splitlines() if line.strip()]

    @classmethod
    def load_neutral(cls, match: SourceMatch, features: Any) -> Any:
        if features.options is None or cls.read_option(TEXT_LINES_OPTION, features.options) is not True:
            return super().load_neutral(match, features)
        lines = cls._non_blank_lines(match.access)
        out: dict[str, list[str]] = {}
        for name in sorted(features.get_all_names()):
            if name.endswith(_SOURCE_SUFFIX):
                out[name] = [match.access] * len(lines)
            elif name.endswith(_FILE_TYPE_SUFFIX):
                out[name] = [cls.file_type(match.access)] * len(lines)
            else:
                out[name] = list(lines)
        return out

    @classmethod
    def count_rows(cls, match: SourceMatch, compute_framework: type[ComputeFramework]) -> int | None:
        return max(1, len(cls._non_blank_lines(match.access)))


class PyFG(ReadDocumentFG):
    @classmethod
    def suffixes(cls) -> tuple[str, ...]:
        return PY_SUFFIXES
