"""Tests for suffix ownership: ReadFile owns structured suffixes, ReadDocument skips them by default.

The ``document_suffixes`` per-feature option overrides this default, letting ReadDocument
claim specific structured suffixes while ReadFile auto-excludes them.
"""

import os
import tempfile
from typing import Any, List, Optional, Tuple

import pytest

from mloda.user import DataAccessCollection, Options
from mloda_plugins.feature_group.input_data.read_file import ReadFile
from mloda_plugins.feature_group.input_data.read_document import ReadDocument


class StubJsonReader(ReadFile):
    """ReadFile subclass that handles .json files with column validation."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".json", ".JSON")

    @classmethod
    def get_column_names(cls, file_name: str) -> list[str]:
        return ["id", "value"]

    @classmethod
    def load_data(cls, data_access: Any, features: Any) -> Any:
        return None


class StubJsonDocReader(ReadDocument):
    """ReadDocument subclass that handles .json files as documents."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".json", ".JSON")

    @classmethod
    def load_data(cls, data_access: Any, features: Any) -> Any:
        return None


class TestDefaultSuffixOwnership:
    """Without document_suffixes option, ReadFile owns .json, ReadDocument skips it."""

    def test_readfile_matches_json_by_default(self) -> None:
        dac = DataAccessCollection(files={"data.json"})
        options = Options()
        result = StubJsonReader.match_subclass_data_access(dac, ["id"], options=options)
        assert result == "data.json"

    def test_readdocument_skips_json_by_default(self) -> None:
        dac = DataAccessCollection(files={"data.json"})
        options = Options()
        result = StubJsonDocReader.match_subclass_data_access(dac, ["content"], options=options)
        assert result is None

    def test_readdocument_matches_non_structured_suffix(self) -> None:
        """ReadDocument should still match suffixes not in _structured_suffixes."""

        class StubMdReader(ReadDocument):
            @classmethod
            def suffix(cls) -> tuple[str, ...]:
                return (".md",)

            @classmethod
            def load_data(cls, data_access: Any, features: Any) -> Any:
                return None

        dac = DataAccessCollection(files={"readme.md"})
        options = Options()
        result = StubMdReader.match_subclass_data_access(dac, ["content"], options=options)
        assert result == "readme.md"


class TestDocumentSuffixesOverride:
    """With document_suffixes option, ReadDocument claims the suffix, ReadFile auto-excludes."""

    def test_readdocument_matches_json_with_override(self) -> None:
        dac = DataAccessCollection(files={"data.json"})
        options = Options({"document_suffixes": frozenset({".json"})})
        result = StubJsonDocReader.match_subclass_data_access(dac, ["content"], options=options)
        assert result == "data.json"

    def test_readfile_excludes_json_with_override(self) -> None:
        dac = DataAccessCollection(files={"data.json"})
        options = Options({"document_suffixes": frozenset({".json"})})
        result = StubJsonReader.match_subclass_data_access(dac, ["id"], options=options)
        assert result is None

    def test_override_is_suffix_specific(self) -> None:
        """Overriding .json does not affect .csv ownership."""

        class StubCsvReader(ReadFile):
            @classmethod
            def suffix(cls) -> tuple[str, ...]:
                return (".csv",)

            @classmethod
            def get_column_names(cls, file_name: str) -> list[str]:
                return ["col"]

            @classmethod
            def load_data(cls, data_access: Any, features: Any) -> Any:
                return None

        dac = DataAccessCollection(files={"data.csv"})
        options = Options({"document_suffixes": frozenset({".json"})})
        result = StubCsvReader.match_subclass_data_access(dac, ["col"], options=options)
        assert result == "data.csv"


class TestFeatureScopeUnaffected:
    """Feature scope (string path) should not be affected by suffix filtering."""

    def test_readdocument_string_path_still_works(self) -> None:
        result = StubJsonDocReader.match_subclass_data_access("doc.json", ["content"], options=Options({}))
        assert result == "doc.json"

    def test_readfile_string_path_still_works(self) -> None:
        result = StubJsonReader.match_subclass_data_access("data.json", ["id"], options=Options({}))
        assert result == "data.json"


class TestNoOptionsBackwardCompatible:
    """Calling without options (as existing tests do) should preserve old behavior."""

    def test_readfile_no_options(self) -> None:
        dac = DataAccessCollection(files={"data.json"})
        result = StubJsonReader.match_subclass_data_access(dac, ["id"], options=Options({}))
        assert result == "data.json"

    def test_readdocument_no_options(self) -> None:
        dac = DataAccessCollection(files={"data.json"})
        result = StubJsonDocReader.match_subclass_data_access(dac, ["content"], options=Options({}))
        assert result is None


class TestFolderTraversal:
    """Suffix ownership applies to files discovered inside folders too."""

    def test_readdocument_skips_structured_in_folder(self) -> None:
        tmp_dir = tempfile.mkdtemp()
        json_path = os.path.join(tmp_dir, "data.json")
        with open(json_path, "w") as f:
            f.write("{}")

        try:
            dac = DataAccessCollection(folders={tmp_dir})
            options = Options()
            result = StubJsonDocReader.match_subclass_data_access(dac, ["content"], options=options)
            assert result is None
        finally:
            os.remove(json_path)
            os.rmdir(tmp_dir)

    def test_readdocument_matches_in_folder_with_override(self) -> None:
        tmp_dir = tempfile.mkdtemp()
        json_path = os.path.join(tmp_dir, "data.json")
        with open(json_path, "w") as f:
            f.write("{}")

        try:
            dac = DataAccessCollection(folders={tmp_dir})
            options = Options({"document_suffixes": frozenset({".json"})})
            result = StubJsonDocReader.match_subclass_data_access(dac, ["content"], options=options)
            assert result == json_path
        finally:
            os.remove(json_path)
            os.rmdir(tmp_dir)


class TestStructuredSuffixesAttribute:
    """Verify _structured_suffixes contains all expected extensions."""

    def test_csv_in_structured(self) -> None:
        assert ".csv" in ReadFile._structured_suffixes

    def test_json_in_structured(self) -> None:
        assert ".json" in ReadFile._structured_suffixes

    def test_parquet_in_structured(self) -> None:
        assert ".parquet" in ReadFile._structured_suffixes

    def test_orc_in_structured(self) -> None:
        assert ".orc" in ReadFile._structured_suffixes

    def test_feather_in_structured(self) -> None:
        assert ".feather" in ReadFile._structured_suffixes

    def test_md_not_in_structured(self) -> None:
        assert ".md" not in ReadFile._structured_suffixes

    def test_text_not_in_structured(self) -> None:
        assert ".text" not in ReadFile._structured_suffixes
