"""Abstract base of one-FeatureGroup-per-file-format: finds files and folders, lists columns, caches per run.

A subclass declares ``suffixes()`` and ``column_names(path)``; the base claims a feature when a file has the column.
"""

import os
from abc import abstractmethod
from collections.abc import Collection
from typing import Any, ClassVar

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.input_data.claim_route import ClaimRoute, NamePolicy, SourceMatch
from mloda.core.abstract_plugins.components.input_data.file_source import FileSource
from mloda.core.abstract_plugins.components.input_data.format_feature_group import FormatFeatureGroup
from mloda.core.abstract_plugins.components.input_data.match_cache import run_cached
from mloda.core.abstract_plugins.components.match_rejection import INPUT_DATA_STAGE, record_match_rejection
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.property_spec import PropertySpec
from mloda.core.abstract_plugins.components.utils import is_match_abort

_HANDLE_OPTION = "data_access_handle"
_DOCUMENT_SUFFIXES_OPTION = "document_suffixes"


class ReadFileFG(FormatFeatureGroup):
    """Abstract file format group: claims features found as columns of owned files and folder entries."""

    CLAIM_ROUTES: ClassVar[tuple[ClaimRoute, ...]] = (
        ClaimRoute("file", NamePolicy.CHECKED, True),
        ClaimRoute("folder", NamePolicy.CHECKED, True),
    )
    PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {
        _HANDLE_OPTION: PropertySpec(
            "Name of the DataAccessCollection file or folder handle to read from.",
            default=None,
            strict_validation=True,
            context=True,
            element_validator=lambda value: isinstance(value, str),
            match_guard=lambda value: isinstance(value, str),
            expected="a str handle name",
        ),
        _DOCUMENT_SUFFIXES_OPTION: PropertySpec(
            "Suffixes handed over to document readers; this group never claims them.", default=None, context=True
        ),
    }
    SAMPLE_SIZE_BYTES: ClassVar[int] = 0

    @classmethod
    @abstractmethod
    def suffixes(cls) -> tuple[str, ...]:
        """File suffixes (with dot) this format owns."""

    @classmethod
    @abstractmethod
    def column_names(cls, path: str) -> Collection[str]:
        """All column names of the file; OSError or ValueError when unreadable, ImportError without the backend."""

    @classmethod
    def sample_column_names(cls, path: str) -> Collection[str] | None:
        """Cheap partial listing from the first SAMPLE_SIZE_BYTES bytes, None when the format has none."""
        return None

    @classmethod
    def file_format(cls) -> str:
        return cls.suffixes()[0].lstrip(".").lower()

    @classmethod
    def is_pointed(
        cls, feature_name: str, options: Options, data_access_collection: DataAccessCollection | None
    ) -> bool:
        if super().is_pointed(feature_name, options, data_access_collection):
            return True
        pinned = cls._pinned_path(feature_name, data_access_collection)
        return pinned is not None and cls._owns(pinned, cls._document_suffixes(options))

    @classmethod
    def ambiguity_fix(
        cls, feature_name: str, matches: list[SourceMatch], data_access_collection: DataAccessCollection | None
    ) -> str:
        return (
            f"pin one source with a column_to_file entry, point {cls.get_class_name()} at one file with "
            f"options={{{cls.get_class_name()!r}: <path>}}, or select one with data_access_handle."
        )

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
        handle = options.get(_HANDLE_OPTION)
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

    @classmethod
    def columns(cls, match: SourceMatch) -> Collection[str] | None:
        return cls._listing("full", match)

    @classmethod
    def has_column(cls, match: SourceMatch, column: str) -> bool | None:
        sample = cls._listing("sample", match)
        if sample is not None:
            if column in sample:
                return True
            if os.path.getsize(match.access) <= cls.SAMPLE_SIZE_BYTES:
                return False
        return super().has_column(match, column)

    @classmethod
    def load_neutral(cls, match: SourceMatch, features: Any) -> Any:
        return FileSource(path=match.access, format=cls.file_format(), columns=tuple(sorted(features.get_all_names())))

    @staticmethod
    def _source(path: str) -> SourceMatch:
        return SourceMatch(source=os.path.abspath(path), access=path)

    @staticmethod
    def _document_suffixes(options: Options) -> tuple[str, ...]:
        value = options.get(_DOCUMENT_SUFFIXES_OPTION)
        if value is None:
            return ()
        return (value,) if isinstance(value, str) else tuple(value)

    @classmethod
    def _pointer_value(cls, options: Options) -> str | None:
        for key in cls.pointer_keys():
            if key in options:
                value = options.get(key)
                return str(value) if isinstance(value, (str, os.PathLike)) else None
        return None

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

    @classmethod
    def _listing(cls, kind: str, match: SourceMatch) -> Collection[str] | None:
        path = os.path.abspath(match.access)
        if not os.path.isfile(path):
            cls._reject_listing(path, "it is not a regular file")
            return None
        stat_result = os.stat(path)
        key = (cls, kind, path, stat_result.st_mtime_ns, stat_result.st_size)
        names, failure = run_cached(key, lambda: cls._read_listing(kind, match.access))
        if failure is not None:
            cls._reject_listing(path, failure)
        return names

    @classmethod
    def _read_listing(cls, kind: str, access: str) -> tuple[tuple[str, ...] | None, str | None]:
        try:
            names = cls.column_names(access) if kind == "full" else cls.sample_column_names(access)
        except (OSError, ValueError, ImportError) as exc:
            if is_match_abort(exc):
                raise
            return None, f"could not read its columns: {exc}"
        return (None if names is None else tuple(names)), None

    @classmethod
    def _reject_listing(cls, path: str, reason: str) -> None:
        record_match_rejection(
            cls.get_class_name(), f"{cls.get_class_name()} matched {path} but {reason}", stage=INPUT_DATA_STAGE
        )
