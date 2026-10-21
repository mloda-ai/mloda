"""Abstract base shared by file format groups and document groups: pointers, pins, owned files and folders."""

import inspect
import os
from abc import abstractmethod
from typing import Any, ClassVar, cast
from urllib.parse import urlsplit

from mloda.core.abstract_plugins.components.credential_scrub import _drop_free_text_userinfo, scrub_credentials
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.input_data.claim_route import ClaimRoute, SourceMatch, first_report_of_pin
from mloda.core.abstract_plugins.components.input_data.format_feature_group import (
    FOLDER_ESCAPE_MARK,
    HANDLE_OPTION,
    HANDLE_SPEC,
    FormatFeatureGroup,
)
from mloda.core.abstract_plugins.components.match_rejection import INPUT_DATA_STAGE, record_match_rejection
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.components.property_spec import PropertySpec

_DOCUMENT_SUFFIXES_OPTION = "document_suffixes"
_UNOWNED_PIN_OWNER = "unowned pin"


def _is_suffix_value(value: Any) -> bool:
    if isinstance(value, str):
        return True
    return isinstance(value, (list, tuple, set, frozenset)) and all(isinstance(item, str) for item in value)


def not_a_file_reason(path: str) -> str:
    return "it is not a regular file" if os.path.exists(path) else "it does not exist"


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

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for route in cls.__dict__.get("CLAIM_ROUTES", ()):
            if route.source_kind not in ("file", "folder"):
                raise TypeError(
                    f"{cls.__name__} route source_kind must be 'file' or 'folder', got {route.source_kind!r}"
                )
        names = {n for base in cls.__bases__ for n in getattr(base, "__abstractmethods__", ())} | set(cls.__dict__)
        if not any(getattr(getattr(cls, n, None), "__isabstractmethod__", False) for n in names):
            suffixes = cls.suffixes()
            if not isinstance(suffixes, tuple) or not suffixes or not all(isinstance(s, str) and s for s in suffixes):
                raise TypeError(f"{cls.__name__}.suffixes() must return a non-empty tuple of non-empty str")

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
            if is_file_route and not pointer.endswith(cls.suffixes()):
                record_match_rejection(
                    cls.get_class_name(),
                    scrub_credentials(
                        f"{pointer} does not end in a {cls.get_class_name()} suffix ({', '.join(cls.suffixes())})"
                    ),
                    stage=INPUT_DATA_STAGE,
                )
            return [cls._source(pointer)] if is_file_route and cls._owns(pointer, excluded) else []
        pinned = cls._pinned_path(feature_name, data_access_collection)
        if pinned is not None:
            if is_file_route and not cls._owns(pinned, excluded):
                cls._record_unowned_pin(pinned)
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
        if "://" in path:
            parts = urlsplit(_drop_free_text_userinfo(path))
            return SourceMatch(source=scrub_credentials(f"{parts.scheme}://{parts.netloc}") + parts.path, access=path)
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
    def _record_unowned_pin(cls, pinned_path: str) -> None:
        """Names a pinned file whose suffix no file or document group owns, once per resolution."""
        if any(
            not inspect.isabstract(group) and pinned_path.endswith(group.suffixes())
            for group in get_all_subclasses(_FileSourceFG)
        ):
            return
        if not first_report_of_pin(pinned_path):
            return
        record_match_rejection(
            _UNOWNED_PIN_OWNER,
            scrub_credentials(f"pinned file {pinned_path} has a suffix no registered reader owns"),
            stage=INPUT_DATA_STAGE,
        )

    @classmethod
    def _owns(cls, path: str, document_suffixes: tuple[str, ...]) -> bool:
        return path.endswith(cls.suffixes()) and not (document_suffixes and path.endswith(document_suffixes))

    @staticmethod
    def _inside(root: str, path: str) -> bool:
        """Whether path resolves under the already resolved root; a prefix check, safe across drives."""
        prefix = os.path.normcase(root).rstrip(os.sep) + os.sep
        return os.path.normcase(os.path.realpath(path)).startswith(prefix)

    @classmethod
    def _folder_entries(cls, folder: str, document_suffixes: tuple[str, ...]) -> list[SourceMatch]:
        if not os.path.isdir(folder):
            return []
        root = os.path.realpath(folder)
        paths = (os.path.join(folder, name) for name in sorted(os.listdir(folder)))
        kept: list[SourceMatch] = []
        for path in paths:
            if not (cls._owns(path, document_suffixes) and os.path.isfile(path)):
                continue
            if not cls._inside(root, path):
                record_match_rejection(
                    cls.get_class_name(),
                    scrub_credentials(
                        f"folder entry {os.path.abspath(path)} {FOLDER_ESCAPE_MARK}, so it is skipped; "
                        "add it as a files handle or point at it"
                    ),
                    stage=INPUT_DATA_STAGE,
                )
                continue
            kept.append(cls._source(path))
        return kept
