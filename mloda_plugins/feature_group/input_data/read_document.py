import os
from pathlib import Path
from typing import Any, Optional

from mloda.provider import BaseInputData, FeatureSet
from mloda.user import DataAccessCollection, Options
from mloda_plugins.feature_group.input_data.read_file import ReadFile


class ReadDocument(BaseInputData):
    """
    Base class for document file readers (text, Markdown, JSON, YAML, etc.).

    _auto_load_group triggers lazy plugin discovery when no ReadDocument subclasses
    are found in the process. Only the read_files subdirectory is loaded.

    To suppress auto-loading:
        PluginLoader.disable_auto_load("feature_group/input_data/read_files")

    By default, ReadDocument skips file types owned by ReadFile (CSV, JSON,
    Parquet, etc.) to avoid conflicts. To read a structured file type as a
    document, set the ``document_suffixes`` option on the Feature:

        Feature("content", options={"document_suffixes": frozenset({".json"})})

    This tells ReadDocument to include .json files and ReadFile to
    auto-exclude them for that feature.
    """

    _auto_load_group: str = "feature_group/input_data/read_files"

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        raise NotImplementedError

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        raise NotImplementedError

    def load(self, features: FeatureSet) -> Any:
        _options = None

        for feature in features.features:
            if _options:
                if _options != feature.options:
                    raise ValueError("All features must have the same options.")
            _options = feature.options

        reader, data_access = self.init_reader(_options)
        data = reader.load_data(data_access, features)

        if data is None:
            raise ValueError(f"Loading data failed for feature {features.get_name_of_one_feature()}.")

        return data

    def init_reader(self, options: Optional[Options]) -> tuple["ReadDocument", Any]:
        if options is None:
            raise ValueError("Options were not set.")

        reader_data_access = options.get("BaseInputData")

        if reader_data_access is None:
            raise ValueError("Reader data access was not set.")

        reader, data_access = reader_data_access
        return reader(), data_access

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
        if isinstance(data_access, DataAccessCollection):
            if data_access.column_to_file is not None:
                pinned = cls._resolve_pinned_file(data_access, feature_names)
                if pinned is not None:
                    return pinned
            document_suffixes = options.get("document_suffixes") or frozenset()
            hint = options.get("data_access_handle")
            if hint is not None:
                handle_kind = data_access.handles().get(hint)
                if handle_kind not in (None, "file"):
                    hint = None
                elif handle_kind == "file" and not cls._document_file_matches(
                    data_access.files[hint], document_suffixes
                ):
                    hint = None
            file_match = data_access.resolve(
                "file",
                predicate=lambda p: cls._document_file_matches(p, document_suffixes),
                hint=hint,
            )
            if file_match is not None:
                return file_match
            folder_paths = list(data_access.folders.values())
            return cls.match_document_data_access(folder_paths, feature_names, document_suffixes)
        if isinstance(data_access, (str, Path)):
            path_str = str(data_access)
            result = cls.match_document_data_access([path_str], feature_names)
            if result is None and cls is ReadDocument:
                return data_access
            return result
        return None

    @classmethod
    def match_document_data_access(
        cls,
        data_accesses: list[str],
        feature_names: list[str],
        document_suffixes: Optional["frozenset[str]"] = None,
    ) -> Any:
        try:
            suffix = cls.suffix()
        except NotImplementedError:
            return None
        for da in data_accesses:
            if da.endswith(suffix):
                if document_suffixes is not None and cls._is_structured_suffix(da, document_suffixes):
                    continue
                return da
            if os.path.isdir(da):
                for file in os.listdir(da):
                    if file.endswith(suffix):
                        if document_suffixes is not None and cls._is_structured_suffix(file, document_suffixes):
                            continue
                        return os.path.join(da, file)
        return None

    @classmethod
    def _document_file_matches(cls, path: str, document_suffixes: "frozenset[str]") -> bool:
        if not cls._has_suffix():
            return False
        if not path.endswith(cls.suffix()):
            return False
        if document_suffixes is not None and cls._is_structured_suffix(path, document_suffixes):
            return False
        return True

    @classmethod
    def _is_structured_suffix(cls, filename: str, document_suffixes: "frozenset[str]") -> bool:
        """Return True if filename has a structured suffix not overridden by document_suffixes."""
        for s in ReadFile._structured_suffixes:
            if filename.endswith(s) and s not in document_suffixes:
                return True
        return False
