"""Abstract base of one-FeatureGroup-per-document-format: one file in, three outputs (text, source, file_type).

A subclass declares ``suffixes()``. Third-party groups should run the contract mixins in mloda's
``tests/mixins/reader_feature_groups/`` (``document_format_feature_group_test_mixin``); the mixins need the ``rejection_window`` fixture from mloda's
``tests/conftest.py``.
"""

import os
from typing import Any, ClassVar

from mloda.core.abstract_plugins.components.credential_scrub import scrub_credentials
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.core.abstract_plugins.components.input_data.claim_route import ClaimRoute, NamePolicy, SourceMatch
from mloda.core.abstract_plugins.components.input_data.format_feature_group import HANDLE_OPTION
from mloda.core.abstract_plugins.components.input_data.file_source_fg import _FileSourceFG, not_a_file_reason
from mloda.core.abstract_plugins.components.match_rejection import INPUT_DATA_STAGE, record_match_rejection
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework

SOURCE_SUFFIX = "~source"
FILE_TYPE_SUFFIX = "~file_type"


class ReadDocumentFG(_FileSourceFG):
    """Document format group; run tests/mixins/reader_feature_groups/document_format_feature_group_test_mixin.py.

    The mixin needs the rejection_window fixture from tests/conftest.py.
    """

    CLAIM_ROUTES: ClassVar[tuple[ClaimRoute, ...]] = (
        ClaimRoute("file", NamePolicy.DECLARED, True),
        ClaimRoute("folder", NamePolicy.DECLARED, True),
    )

    @classmethod
    def handover_suffixes(cls) -> tuple[str, ...]:
        return ()

    @classmethod
    def default_owned_suffixes(cls) -> tuple[str, ...]:
        handed_over = cls.handover_suffixes()
        return tuple(suffix for suffix in cls.suffixes() if suffix not in handed_over)

    @classmethod
    def read_text(cls, path: str) -> str:
        with open(path, encoding="utf-8") as handle:
            return handle.read()

    @classmethod
    def file_type(cls, path: str) -> str:
        return os.path.splitext(path)[1].lstrip(".").lower()

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
    def find_sources(
        cls,
        route: ClaimRoute,
        feature_name: str,
        options: Options,
        data_access_collection: DataAccessCollection | None,
    ) -> list[SourceMatch]:
        if feature_name not in cls.declared_names():
            return []
        name = cls.get_class_name()
        found = super().find_sources(route, feature_name, options, data_access_collection)
        if route.source_kind == "file":
            cls._record_handed_over(feature_name, options, data_access_collection)
        gone = [match for match in found if not os.path.isfile(match.access)]
        if gone:
            reason = not_a_file_reason(os.path.abspath(gone[0].access))
            record_match_rejection(
                name, scrub_credentials(f"{name} matched {gone[0].source} but {reason}"), stage=INPUT_DATA_STAGE
            )
        kept = [match for match in found if match not in gone]
        if not kept and cls._pointer_key(options) is None:
            cls._record_collection_handed_over(route, options, data_access_collection)
        return kept

    @classmethod
    def _record_handed_over(
        cls, feature_name: str, options: Options, data_access_collection: DataAccessCollection | None
    ) -> None:
        path = cls._pointer_value(options) or cls._pinned_path(feature_name, data_access_collection)
        if path is not None:
            cls._record_if_handed_over(path, options)

    @classmethod
    def _record_collection_handed_over(
        cls, route: ClaimRoute, options: Options, data_access_collection: DataAccessCollection | None
    ) -> None:
        if data_access_collection is None:
            return
        files = data_access_collection.files
        folders = data_access_collection.folders
        handle = cls.read_option(HANDLE_OPTION, options)
        if isinstance(handle, str):
            files = {handle: files[handle]} if handle in files else {}
            folders = {handle: folders[handle]} if handle in folders else {}
        if route.source_kind == "file":
            paths = list(files.values())
        else:
            paths = [
                os.path.join(f, entry)
                for f in folders.values()
                if os.path.isdir(f)
                for entry in sorted(os.listdir(f))
                if cls._inside(os.path.realpath(f), os.path.join(f, entry))
            ]
        for path in paths:
            if os.path.isfile(path):
                cls._record_if_handed_over(path, options)

    @classmethod
    def _record_if_handed_over(cls, path: str, options: Options) -> None:
        if path.endswith(cls.suffixes()) and not cls._owns(path, cls._document_suffixes(options)):
            record_match_rejection(
                cls.get_class_name(),
                scrub_credentials(
                    f"{path} is handed over to another reader; list its suffix in document_suffixes to read it here"
                ),
                stage=INPUT_DATA_STAGE,
            )

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
