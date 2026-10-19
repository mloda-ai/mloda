"""Abstract base shared by file format groups and document groups: pointers, pins, owned files and folders."""

import inspect
import os
from abc import abstractmethod
from typing import Any, ClassVar, cast

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.input_data.claim_route import ClaimRoute, SourceMatch
from mloda.core.abstract_plugins.components.input_data.format_feature_group import (
    HANDLE_OPTION,
    HANDLE_SPEC,
    FormatFeatureGroup,
)
from mloda.core.abstract_plugins.components.match_rejection import INPUT_DATA_STAGE, record_match_rejection
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.property_spec import PropertySpec

_DOCUMENT_SUFFIXES_OPTION = "document_suffixes"


def _is_suffix_value(value: Any) -> bool:
    if isinstance(value, str):
        return True
    return isinstance(value, (list, tuple, set, frozenset)) and all(isinstance(item, str) for item in value)


class _FileSourceFG(FormatFeatureGroup):
    """Abstract plumbing for groups whose sources are files and folders of files."""

    PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {
        HANDLE_OPTION: HANDLE_SPEC,
        _DOCUMENT_SUFFIXES_OPTION: PropertySpec(
            "Suffixes handed over to document readers; a file group never claims them, a document group claims "
            "the ones listed here.",
            default=None,
            strict_validation=True,
            context=True,
            element_validator=_is_suffix_value,
            match_guard=_is_suffix_value,
            expected="a str or a collection of str suffixes",
        ),
    }
    SAMPLE_SIZE_BYTES: ClassVar[int] = 0

    @classmethod
    @abstractmethod
    def suffixes(cls) -> tuple[str, ...]:
        """File suffixes (with dot) this format owns."""

    @classmethod
    def default_owned_suffixes(cls) -> tuple[str, ...]:
        """Suffixes this group claims when no document_suffixes option is set."""
        return cls.suffixes()

    @classmethod
    def file_format(cls) -> str:
        return cls.suffixes()[0].lstrip(".").lower()

    @classmethod
    def is_pointed(
        cls, feature_name: str, options: Options, data_access_collection: DataAccessCollection | None
    ) -> bool:
        if super().is_pointed(feature_name, options, data_access_collection):
            return True
        if inspect.isabstract(cls):
            return False
        pinned = cls._pinned_path(feature_name, data_access_collection)
        return pinned is not None and cls._owns(pinned, cls._document_suffixes(options))

    @classmethod
    def find_sources(
        cls,
        route: ClaimRoute,
        feature_name: str,
        options: Options,
        data_access_collection: DataAccessCollection | None,
    ) -> list[SourceMatch]:
        excluded = cls._document_suffixes(options)
        is_file_route = route.source_kind == "file"
        pointer = cls._pointer_value(options)
        if pointer is None and cls._pointer_key(options) is not None:
            if is_file_route:
                bad = type(options.get(cast(str, cls._pointer_key(options)))).__name__
                record_match_rejection(
                    cls.get_class_name(), f"the pointer value must be a path, got {bad}", stage=INPUT_DATA_STAGE
                )
            return []
        if pointer is not None:
            if os.path.isdir(pointer):
                return [] if is_file_route else cls._folder_entries(pointer, excluded)
            return [cls._source(pointer)] if is_file_route and cls._owns(pointer, excluded) else []
        pinned = cls._pinned_path(feature_name, data_access_collection)
        if pinned is not None:
            return [cls._source(pinned)] if is_file_route and cls._owns(pinned, excluded) else []
        dac = data_access_collection
        if dac is None:
            return []
        files: list[str] = list(dac.files.values())
        folders: list[str] = list(dac.folders.values())
        handle = options.get(HANDLE_OPTION)
        if isinstance(handle, str):
            if handle in dac.files:
                files, folders = [dac.files[handle]], []
            elif handle in dac.folders:
                files, folders = [], [dac.folders[handle]]
            else:
                files, folders = [], []
                if is_file_route and handle not in dac.handles():
                    record_match_rejection(
                        cls.get_class_name(),
                        f"data_access_handle '{handle}' is unknown; file handles: {sorted(dac.files)}, "
                        f"folder handles: {sorted(dac.folders)}",
                        stage=INPUT_DATA_STAGE,
                    )
        if is_file_route:
            return [cls._source(path) for path in files if cls._owns(path, excluded)]
        return [match for folder in folders for match in cls._folder_entries(folder, excluded)]

    @staticmethod
    def _source(path: str) -> SourceMatch:
        return SourceMatch(source=os.path.abspath(path), access=path)

    @staticmethod
    def _document_suffixes(options: Options) -> tuple[str, ...]:
        value = options.get(_DOCUMENT_SUFFIXES_OPTION)
        if value is None or not _is_suffix_value(value):
            return ()
        return (value,) if isinstance(value, str) else tuple(value)

    @classmethod
    def _pointer_key(cls, options: Options) -> str | None:
        return next((key for key in cls.pointer_keys() if key in options), None)

    @classmethod
    def _pointer_value(cls, options: Options) -> str | None:
        key = cls._pointer_key(options)
        value = None if key is None else options.get(key)
        return str(value) if isinstance(value, (str, os.PathLike)) else None

    @staticmethod
    def _pinned_path(feature_name: str, data_access_collection: DataAccessCollection | None) -> str | None:
        if data_access_collection is None or data_access_collection.column_to_file is None:
            return None
        handle = data_access_collection.column_to_file.get(feature_name)
        return None if handle is None else data_access_collection.files.get(handle)

    @classmethod
    def _owns(cls, path: str, document_suffixes: tuple[str, ...]) -> bool:
        return path.endswith(cls.suffixes()) and not (document_suffixes and path.endswith(document_suffixes))

    @classmethod
    def _folder_entries(cls, folder: str, document_suffixes: tuple[str, ...]) -> list[SourceMatch]:
        if not os.path.isdir(folder):
            return []
        paths = (os.path.join(folder, name) for name in sorted(os.listdir(folder)))
        return [cls._source(p) for p in paths if cls._owns(p, document_suffixes) and os.path.isfile(p)]
