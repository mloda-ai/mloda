import os
from pathlib import Path
from typing import Any
from mloda.user import DataAccessCollection
from mloda.provider import FeatureSet
from mloda.provider import BaseInputData
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

    A ReadFile subclass classifies as a final reader by overriding ``load_data``
    wholesale. It may return its table directly, or a descriptor materialized by
    the target compute framework (CsvReader returns a ``FileSource``).

    If get_column_names is not implemented, the class will assume the columns are there.
    """

    _auto_load_group: str = "feature_group/input_data/read_files"

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

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
        document_suffixes: "frozenset[str]" = options.get("document_suffixes") or frozenset()

        if isinstance(data_access, DataAccessCollection):
            if data_access.column_to_file is not None:
                pinned = cls._resolve_pinned_file(data_access, feature_names)
                if pinned is not None:
                    return pinned
            hint = options.get("data_access_handle")
            if hint is not None:
                handle_kind = data_access.handles().get(hint)
                if handle_kind not in (None, "file"):
                    hint = None
                elif handle_kind == "file" and not cls._file_matches(
                    data_access.files[hint], feature_names, document_suffixes
                ):
                    hint = None
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
        return cls.validate_columns(path, feature_names) is not False

    @classmethod
    def match_read_file_data_access(
        cls, data_accesses: list[str], feature_names: list[str], document_suffixes: "frozenset[str]" = frozenset()
    ) -> Any:
        for data_access in data_accesses:
            if data_access.endswith(cls.suffix()):
                if document_suffixes and any(data_access.endswith(s) for s in document_suffixes):
                    continue
                if cls.validate_columns(data_access, feature_names) is False:
                    continue

                return data_access

            if os.path.isdir(data_access):
                for file in os.listdir(data_access):
                    if file.endswith(cls.suffix()):
                        if document_suffixes and any(file.endswith(s) for s in document_suffixes):
                            continue
                        file_name = os.path.join(data_access, file)

                        if cls.validate_columns(file_name, feature_names) is False:
                            continue

                        return file_name
        return None

    @classmethod
    def validate_columns(cls, file_name: str, feature_names: list[str]) -> bool:
        try:
            columns = cls.get_column_names(file_name)
        except NotImplementedError:
            return True

        for feature in feature_names:
            if feature not in columns:
                return False
        return True
