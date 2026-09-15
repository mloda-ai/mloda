import os
from pathlib import Path
from typing import Any, ClassVar
from mloda.core.abstract_plugins.components.utils import is_match_abort
from mloda.user import DataAccessCollection
from mloda.provider import FeatureSet
from mloda.provider import (
    BaseInputData,
    CHAIN_SEPARATOR,
    COLUMN_SEPARATOR,
    INPUT_DATA_STAGE,
    PropertySpec,
    record_match_rejection,
)
from mloda.user import DataType
from mloda.user import Options


class ReadFile(BaseInputData):
    """
    ReadFile is responsible for loading and processing structured input data files.

    _auto_load_group triggers lazy plugin discovery when no ReadFile subclasses
    are found in the process (i.e. when the user has not imported CsvReader etc.).
    Only the read_files subdirectory is loaded, not the entire feature_group tree.

    To suppress auto-loading:
        PluginLoader.disable_auto_load("feature_group/input_data/read_files")

    This class should be inherited by all classes that are responsible for reading files.

    _structured_suffixes lists file extensions that ReadFile owns by default.
    ReadDocument skips these suffixes unless overridden via the per-feature
    ``document_suffixes`` option. When that option is set, ReadFile auto-excludes
    the listed suffixes so that ReadDocument can claim them instead.

    The following methods should be implemented in the child classes:
    - load_data
    - suffix
    - get_column_names
    - describe_columns (optional; default wraps get_column_names with unknown types)

    A ReadFile subclass classifies as a final reader by overriding ``load_data``
    wholesale. It may return its table directly, or a descriptor materialized by
    the target compute framework (CsvReader returns a ``FileSource``).

    A reader that cannot enumerate columns (get_column_names not overridden, or raising NotImplementedError
    or ImportError) assumes plain columns are present but declines a chain- or column-separated name while
    matching; an explicit column_to_file pin is exempt. An unpinned file whose columns cannot be read
    (OSError, ValueError) is declined; a pinned one raises. A match_subclass_data_access override should
    route through _file_matches, and through _pin_applies/_resolve_pinned_file for pinned requests, to
    keep these rules.
    """

    _auto_load_group: str = "feature_group/input_data/read_files"

    READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
        "document_suffixes": PropertySpec(
            "Suffixes handed over to document readers; ReadFile auto-excludes them from its own matching.",
            default=frozenset(),
        ),
        "data_access_handle": PropertySpec(
            "Hint naming which DataAccessCollection file handle to prefer while matching.",
            default=None,
            element_validator=lambda value: isinstance(value, str),
            strict_validation=True,
            scalar_only=True,
        ),
    }

    _structured_suffixes: "frozenset[str]" = frozenset(
        {
            ".csv",
            ".CSV",
            ".json",
            ".JSON",
            ".parquet",
            ".PARQUET",
            ".pqt",
            ".PQT",
            ".orc",
            ".ORC",
            ".feather",
        }
    )

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        """
        This function should be implemented by child classes.
        """
        raise NotImplementedError

    @classmethod
    def _final_reader_requires(cls) -> tuple[str, ...]:
        # ReadFile anchors its own subtree so a subclass classifies as a final reader only
        # by overriding load_data wholesale; there is no hook-based classification here.
        return ()

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        raise NotImplementedError

    @classmethod
    def get_column_names(cls, file_name: str) -> list[str]:
        raise NotImplementedError

    @staticmethod
    def _file_path(data_access: Any) -> str:
        """Coerce a str or Path data_access to a plain path string; anything else is a caller error."""
        if isinstance(data_access, (str, Path)):
            return str(data_access)
        raise ValueError(f"describe_columns requires a file path (str or Path), got {type(data_access).__name__}.")

    @classmethod
    def describe_columns(cls, data_access: Any) -> dict[str, DataType | None]:
        """Family default: wraps get_column_names, mapping every column name to an unknown (None) type."""
        return {name: None for name in cls.get_column_names(cls._file_path(data_access))}

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
        document_suffixes: "frozenset[str]" = cls.reader_option("document_suffixes", options)

        if isinstance(data_access, DataAccessCollection):
            if cls._pin_applies(data_access, feature_names):
                return cls._resolve_pinned_file(data_access, feature_names)
            hint = options.get("data_access_handle")
            if hint is not None:
                handle_kind = data_access.handles().get(hint)
                if handle_kind not in (None, "file"):
                    hint = None
                elif handle_kind == "file" and not cls._file_matches(
                    data_access.files[hint], feature_names, document_suffixes
                ):
                    return None
            file_match = data_access.resolve(
                "file",
                predicate=lambda p: cls._file_matches(p, feature_names, document_suffixes),
                hint=hint,
            )
            if file_match is not None:
                return file_match
            folder_paths = list(data_access.folders.values())
            return cls.match_read_file_data_access(folder_paths, feature_names, document_suffixes)
        elif isinstance(data_access, str):
            data_accesses = [data_access]
        elif isinstance(data_access, Path):
            data_accesses = [str(data_access)]
        else:
            return None

        matched_data_access = cls.match_read_file_data_access(data_accesses, feature_names, document_suffixes)
        if matched_data_access is None:
            return None
        return matched_data_access

    @classmethod
    def _file_matches(cls, path: str, feature_names: list[str], document_suffixes: "frozenset[str]") -> bool:
        if not path.endswith(cls.suffix()):
            return False
        if document_suffixes and any(path.endswith(s) for s in document_suffixes):
            return False
        try:
            if cls.validate_columns(path, feature_names) is False:
                return False
            return not cls._declines_unvalidated_separator_name(path, feature_names)
        except (OSError, ValueError) as exc:
            if is_match_abort(exc):
                raise
            # An unreadable unpinned file declines so sibling readers still match; a pinned file is read
            # first by _resolve_pinned_file through validate_columns, and once a pin applies there is no
            # fallback path for its error to bypass.
            record_match_rejection(
                cls.get_class_name(),
                f"{cls.get_class_name()} matched the suffix of {path} but could not read its columns: {exc}",
                stage=INPUT_DATA_STAGE,
            )
            return False

    @classmethod
    def _declines_unvalidated_separator_name(cls, file_name: str, feature_names: list[str]) -> bool:
        """Declines a chain/column-separated name this reader cannot confirm by enumerating columns."""
        feature = next((name for name in feature_names if CHAIN_SEPARATOR in name or COLUMN_SEPARATOR in name), None)
        if feature is None:
            return False
        if cls._column_names_or_none(file_name) is not None:
            return False
        record_match_rejection(
            cls.get_class_name(),
            f"{cls.get_class_name()} matched the suffix of {file_name} but cannot enumerate its columns via "
            f"get_column_names, so it cannot confirm the chain/column-separated name '{feature}'",
            stage=INPUT_DATA_STAGE,
        )
        return True

    @classmethod
    def match_read_file_data_access(
        cls, data_accesses: list[str], feature_names: list[str], document_suffixes: "frozenset[str]" = frozenset()
    ) -> Any:
        for data_access in data_accesses:
            if data_access.endswith(cls.suffix()):
                if cls._file_matches(data_access, feature_names, document_suffixes):
                    return data_access
                continue

            if os.path.isdir(data_access):
                for file in os.listdir(data_access):
                    file_name = os.path.join(data_access, file)
                    if cls._file_matches(file_name, feature_names, document_suffixes):
                        return file_name
        return None

    @classmethod
    def _column_names_or_none(cls, file_name: str) -> "list[str] | None":
        """get_column_names' result, or None when it cannot enumerate (NotImplementedError or ImportError)."""
        try:
            return cls.get_column_names(file_name)
        except (NotImplementedError, ImportError) as exc:
            if is_match_abort(exc):
                raise
            return None

    @classmethod
    def validate_columns(cls, file_name: str, feature_names: list[str]) -> bool:
        """A suffix-owned file lacking a requested column records an attributable decline before returning False."""
        columns = cls._column_names_or_none(file_name)
        if columns is None:
            return True

        missing = [feature for feature in feature_names if feature not in columns]
        if missing:
            # Attributable decline: ownership established, content failed; plain non-matches stay silent.
            record_match_rejection(
                cls.get_class_name(),
                f"{cls.get_class_name()} matched the suffix of {file_name} but it lacks the column(s): "
                f"{', '.join(missing)}",
                stage=INPUT_DATA_STAGE,
            )
            return False
        return True
