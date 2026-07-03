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

    load_data is a template method exposing an opt-in lifecycle seam: a new
    reader implements ``produce_document`` (optionally ``document_file_type``)
    plus ``suffix`` instead of overriding ``load_data`` wholesale; both are
    required for the class to be discovered as a final reader. Overriding
    ``load_data`` directly is still supported.
    """

    _auto_load_group: str = "feature_group/input_data/read_files"

    @classmethod
    def produce_document(cls, file_path: str) -> Any:
        """The per-format parse hook; a new reader implements THIS instead of overriding load_data wholesale."""
        raise NotImplementedError

    @classmethod
    def document_file_type(cls, file_path: str) -> str:
        """Overridable hook for the envelope's file_type; default is the first declared suffix without the dot."""
        return cls.suffix()[0].lstrip(".")

    @classmethod
    def _read_text(cls, file_path: str) -> str:
        """Shared helper for text-based readers: read the whole file as UTF-8 text."""
        with open(file_path, encoding="utf-8") as file:
            return file.read()

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        """Template method: probe the parse hook, resolve the path, parse, then wrap the one-row envelope."""
        if not cls._is_overridden(ReadDocument, "produce_document"):
            raise NotImplementedError

        file_path = features.get_options_key(cls.__name__)
        content = cls.produce_document(file_path)
        return [{cls.get_class_name(): content, "source": file_path, "file_type": cls.document_file_type(file_path)}]

    @classmethod
    def supports_scoped_data_access(cls) -> bool:
        # A ReadDocument subclass is a final scoped reader if it overrides load_data
        # wholesale, or implements BOTH the per-format parse hook (produce_document)
        # and suffix. Decided structurally via _is_overridden() so plugin discovery
        # never executes the lifecycle (no produce_document() side effects, no
        # escaping exceptions) and works for classmethod/staticmethod/plain overrides
        # alike. Requiring suffix screens out intermediate bases that share
        # produce_document but leave suffix abstract, mirroring ReadDB's connect
        # requirement.
        if cls._is_overridden(ReadDocument, "load_data"):
            return True
        return cls._is_overridden(ReadDocument, "produce_document") and cls._is_overridden(ReadDocument, "suffix")

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        raise NotImplementedError

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
