"""Abstract base of one-FeatureGroup-per-file-format: finds files and folders, lists columns, caches per run.

A subclass declares ``suffixes()`` and ``column_names(path)``; the base claims a feature when a file has the column.
"""

import inspect
import os
from abc import abstractmethod
from collections.abc import Collection
from typing import Any, ClassVar, cast

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.input_data.claim_route import ClaimRoute, NamePolicy, SourceMatch
from mloda.core.abstract_plugins.components.input_data.file_source import FileSource
from mloda.core.abstract_plugins.components.input_data.format_feature_group import (
    HANDLE_OPTION,
    HANDLE_SPEC,
    FormatFeatureGroup,
)
from mloda.core.abstract_plugins.components.input_data.match_cache import run_cached
from mloda.core.abstract_plugins.components.match_rejection import INPUT_DATA_STAGE, record_match_rejection
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.property_spec import PropertySpec
from mloda.core.abstract_plugins.components.utils import is_match_abort

_DOCUMENT_SUFFIXES_OPTION = "document_suffixes"


def _is_suffix_value(value: Any) -> bool:
    if isinstance(value, str):
        return True
    return isinstance(value, (list, tuple, set, frozenset)) and all(isinstance(item, str) for item in value)


class ReadFileFG(FormatFeatureGroup):
    """Abstract file format group: claims features found as columns of owned files and folder entries."""

    CLAIM_ROUTES: ClassVar[tuple[ClaimRoute, ...]] = (
        ClaimRoute("file", NamePolicy.CHECKED, True),
        ClaimRoute("folder", NamePolicy.CHECKED, True),
    )
    PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {
        HANDLE_OPTION: HANDLE_SPEC,
        _DOCUMENT_SUFFIXES_OPTION: PropertySpec(
            "Suffixes handed over to document readers; this group never claims them.",
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
        if inspect.isabstract(cls):
            return False
        pinned = cls._pinned_path(feature_name, data_access_collection)
        return pinned is not None and cls._owns(pinned, cls._document_suffixes(options))

    @classmethod
    def ambiguity_fix(
        cls, feature_name: str, matches: list[SourceMatch], data_access_collection: DataAccessCollection | None
    ) -> str:
        sources = {match.source for match in matches}
        handles = sorted(
            handle
            for handle, path in (data_access_collection.files.items() if data_access_collection else ())
            if os.path.abspath(path) in sources
        )
        name = cls.get_class_name()
        if handles:
            return (
                f"pin one source with a column_to_file entry, point {name} at one file with "
                f"options={{{name!r}: <path>}}, or select one with data_access_handle "
                f"(file handles: {', '.join(repr(h) for h in handles)})."
            )
        return (
            f"point {name} at one file with options={{{name!r}: <path>}}, or add the file as a files handle "
            "and pin it with a column_to_file entry."
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

    @classmethod
    def _overrides_load_neutral(cls, base: type) -> bool:
        """True when a class below ``base`` in the MRO defines its own load_neutral."""
        return any("load_neutral" in klass.__dict__ for klass in cls.__mro__[: cls.__mro__.index(base)])

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

    @classmethod
    def _listing(cls, kind: str, match: SourceMatch) -> Collection[str] | None:
        names, failure = cls._listing_result(kind, match)
        if failure is not None:
            cls._reject_listing(os.path.abspath(match.access), failure)
        return names

    @classmethod
    def unknown_columns_reason(cls, match: SourceMatch) -> str | None:
        return cls._listing_result("full", match)[1]

    @classmethod
    def _listing_result(cls, kind: str, match: SourceMatch) -> tuple[Collection[str] | None, str | None]:
        path = os.path.abspath(match.access)
        if not os.path.isfile(path):
            return None, "it is not a regular file"
        stat_result = os.stat(path)
        key = (cls, kind, path, stat_result.st_mtime_ns, stat_result.st_size)
        return run_cached(key, lambda: cls._read_listing(kind, match.access))

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
