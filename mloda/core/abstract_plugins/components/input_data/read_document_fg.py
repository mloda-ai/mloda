"""Abstract base of one-FeatureGroup-per-document-format: one file in, three outputs (text, source, file_type).

A subclass declares ``suffixes()``. Third-party groups should run the contract mixins in mloda's
``tests/mixins/reader_feature_groups/`` (``document_format_feature_group_test_mixin``).
"""

import os
from collections.abc import Collection
from typing import Any, ClassVar

from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.core.abstract_plugins.components.input_data.claim_route import ClaimRoute, NamePolicy, SourceMatch
from mloda.core.abstract_plugins.components.input_data.read_file_fg import ReadFileFG
from mloda.core.abstract_plugins.compute_framework import ComputeFramework

SOURCE_SUFFIX = "~source"
FILE_TYPE_SUFFIX = "~file_type"


class ReadDocumentFG(ReadFileFG):
    """Document format group; run tests/mixins/reader_feature_groups/document_format_feature_group_test_mixin.py."""

    CLAIM_ROUTES: ClassVar[tuple[ClaimRoute, ...]] = (
        ClaimRoute("file", NamePolicy.DECLARED, True),
        ClaimRoute("folder", NamePolicy.DECLARED, True),
    )

    @classmethod
    def handover_suffixes(cls) -> tuple[str, ...]:
        return ()

    @classmethod
    def read_text(cls, path: str) -> str:
        with open(path, encoding="utf-8") as handle:
            return handle.read()

    @classmethod
    def file_type(cls, path: str) -> str:
        return os.path.splitext(path)[1].lstrip(".").lower()

    @classmethod
    def column_names(cls, path: str) -> Collection[str]:
        return sorted(cls.declared_names())

    @classmethod
    def declared_names(cls) -> frozenset[str]:
        names: set[str] = set()
        for key in cls.pointer_keys():
            names |= {key, f"{key}{SOURCE_SUFFIX}", f"{key}{FILE_TYPE_SUFFIX}"}
        return frozenset(names)

    @classmethod
    def _declared_name(cls, base_name: str) -> bool:
        return base_name in cls.declared_names()

    @classmethod
    def _owns(cls, path: str, document_suffixes: tuple[str, ...]) -> bool:
        if not path.endswith(cls.suffixes()):
            return False
        handed_over = path.endswith(cls.handover_suffixes()) if cls.handover_suffixes() else False
        listed = bool(document_suffixes) and path.endswith(document_suffixes)
        return not (handed_over and not listed)

    @classmethod
    def load_neutral(cls, match: SourceMatch, features: Any) -> Any:
        out: dict[str, list[str]] = {}
        text: str | None = None
        for name in sorted(features.get_all_names()):
            if name.endswith(SOURCE_SUFFIX):
                out[name] = [match.access]
            elif name.endswith(FILE_TYPE_SUFFIX):
                out[name] = [cls.file_type(match.access)]
            else:
                if text is None:
                    text = cls.read_text(match.access)
                out[name] = [text]
        return out

    @classmethod
    def describe_columns(cls, match: SourceMatch) -> dict[str, DataType | None]:
        name = cls.get_class_name()
        return {
            name: DataType.STRING,
            f"{name}{SOURCE_SUFFIX}": DataType.STRING,
            f"{name}{FILE_TYPE_SUFFIX}": DataType.STRING,
        }

    @classmethod
    def count_rows(cls, match: SourceMatch, compute_framework: type[ComputeFramework]) -> int | None:
        return 1
