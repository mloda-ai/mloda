"""Abstract base of one-FeatureGroup-per-file-format: finds files and folders, lists columns, caches per run.

A subclass declares ``suffixes()`` and ``column_names(path)``; the base claims a feature when a file has the column.
Third-party groups should run the contract mixins in mloda's ``tests/mixins/reader_feature_groups/``
(``file_format_feature_group_test_mixin``); the mixins need the ``rejection_window`` fixture from mloda's
``tests/conftest.py``.
"""

import os
from abc import abstractmethod
from collections.abc import Collection
from typing import Any, ClassVar

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.input_data.claim_route import ClaimRoute, NamePolicy, SourceMatch
from mloda.core.abstract_plugins.components.input_data.file_source import FileSource
from mloda.core.abstract_plugins.components.input_data.file_source_fg import _FileSourceFG
from mloda.core.abstract_plugins.components.input_data.match_cache import run_cached
from mloda.core.abstract_plugins.components.match_rejection import INPUT_DATA_STAGE, record_match_rejection
from mloda.core.abstract_plugins.components.utils import is_match_abort


class ReadFileFG(_FileSourceFG):
    """Abstract file format group: claims features found as columns of owned files and folder entries."""

    CLAIM_ROUTES: ClassVar[tuple[ClaimRoute, ...]] = (
        ClaimRoute("file", NamePolicy.CHECKED, True),
        ClaimRoute("folder", NamePolicy.CHECKED, True),
    )

    @classmethod
    @abstractmethod
    def column_names(cls, path: str) -> Collection[str]:
        """All column names of the file; OSError or ValueError when unreadable, ImportError without the backend."""

    @classmethod
    def sample_column_names(cls, path: str) -> Collection[str] | None:
        """Cheap partial listing from the first SAMPLE_SIZE_BYTES bytes, None when the format has none."""
        return None

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
