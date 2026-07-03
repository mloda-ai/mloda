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

    load_data is a template method exposing an opt-in lifecycle seam: a new
    backend implements ``produce_table``, ``_pyarrow_module``, and
    ``_file_format_label`` (optionally ``get_column_names``) plus ``suffix``
    instead of overriding ``load_data`` wholesale; ``produce_table``,
    ``suffix``, and ``_pyarrow_module`` are all required for the class to be
    discovered as a final reader. Overriding ``load_data`` directly is still
    supported.

    If get_column_names is not implemented, the class will assume the columns are there.
    """

    _auto_load_group: str = "feature_group/input_data/read_files"

    _file_format_label: str = "structured"

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
    def produce_table(cls, data_access: Any, column_names: list[str]) -> Any:
        """The per-format read hook; a new backend implements THIS (plus _pyarrow_module and
        _file_format_label) instead of overriding load_data wholesale."""
        raise NotImplementedError

    @classmethod
    def _pyarrow_module(cls) -> Any:
        """Return the pyarrow submodule used by this reader, or None when pyarrow is absent."""
        raise NotImplementedError

    @classmethod
    def check_pyarrow_backend(cls) -> None:
        """Raises the centralized ImportError when the reader's pyarrow backend is missing."""
        if cls._pyarrow_module() is None:
            raise ImportError(
                f"pyarrow is required to read {cls._file_format_label} files. "
                "Install it with: pip install 'mloda[pyarrow]'"
            )

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        """Template method: probe the read hook, check the pyarrow backend, then produce the table."""
        if not cls._is_overridden(ReadFile, "produce_table"):
            raise NotImplementedError

        cls.check_pyarrow_backend()
        return cls.produce_table(data_access, list(features.get_all_names()))

    @classmethod
    def supports_scoped_data_access(cls) -> bool:
        # A ReadFile subclass is a final scoped reader if it overrides load_data
        # wholesale, or implements ALL of the per-format read hook (produce_table),
        # suffix, and _pyarrow_module. Decided structurally via _is_overridden() so
        # plugin discovery never executes the lifecycle (no _pyarrow_module()/
        # produce_table() side effects, no escaping exceptions). Requiring suffix and
        # _pyarrow_module screens out intermediate bases that share produce_table but
        # leave suffix or the backend guard abstract, mirroring ReadDB's connect
        # requirement.
        if cls._is_overridden(ReadFile, "load_data"):
            return True
        return (
            cls._is_overridden(ReadFile, "produce_table")
            and cls._is_overridden(ReadFile, "suffix")
            and cls._is_overridden(ReadFile, "_pyarrow_module")
        )

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
