"""Tests for suffix ownership: the file format groups own structured suffixes, ReadDocument skips them by default.

The ``document_suffixes`` per-feature option overrides this default, letting ReadDocument
claim specific structured suffixes while the format groups auto-exclude them.
"""

import ast
import importlib
import inspect
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

from mloda.user import DataAccessCollection, Options
from mloda_plugins.feature_group.input_data.read_document import ReadDocument
from mloda_plugins.feature_group.input_data.read_files.json_document_reader import JsonDocumentReader
from tests.mixins.reader_feature_groups.format_file_writers import write_csv, write_json
from tests.mixins.reader_feature_groups.lazy_format_group import load_group

SUFFIX_COLUMN = "suffixown_id"


def _json_group() -> Any:
    return load_group("json_fg", "JsonFG")


@pytest.fixture
def json_file(tmp_path: Path) -> str:
    path = tmp_path / "data.json"
    write_json(path, {SUFFIX_COLUMN: [1, 2], "suffixown_value": [3, 4]})
    return str(path)


class StubJsonDocReader(ReadDocument):
    """ReadDocument subclass that handles .json files as documents."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".json", ".JSON")

    @classmethod
    def load_data(cls, data_access: Any, features: Any) -> Any:
        return None


class TestDefaultSuffixOwnership:
    """Without document_suffixes option, JsonFG owns .json, ReadDocument skips it."""

    def test_json_fg_matches_json_by_default(self, json_file: str) -> None:
        dac = DataAccessCollection(files={json_file})
        assert _json_group().match_feature_group_criteria(SUFFIX_COLUMN, Options(), dac)

    def test_readdocument_skips_json_by_default(self) -> None:
        dac = DataAccessCollection(files={"data.json"})
        options = Options()
        result = StubJsonDocReader.match_subclass_data_access(dac, ["content"], options=options)
        assert result is None

    def test_readdocument_matches_non_structured_suffix(self) -> None:
        """ReadDocument should still match suffixes not in STRUCTURED_SUFFIXES."""

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
    """With document_suffixes option, ReadDocument claims the suffix, the format group auto-excludes."""

    def test_readdocument_matches_json_with_override(self) -> None:
        dac = DataAccessCollection(files={"data.json"})
        options = Options({"document_suffixes": frozenset({".json"})})
        result = StubJsonDocReader.match_subclass_data_access(dac, ["content"], options=options)
        assert result == "data.json"

    def test_json_fg_excludes_json_with_override(self, json_file: str) -> None:
        dac = DataAccessCollection(files={json_file})
        options = Options({"document_suffixes": frozenset({".json"})})
        assert not _json_group().match_feature_group_criteria(SUFFIX_COLUMN, options, dac)

    def test_a_json_handed_to_document_readers_is_declined_by_json_fg_and_claimed_by_the_json_document_reader(
        self, json_file: str
    ) -> None:
        dac = DataAccessCollection(files={json_file})
        options = Options({"document_suffixes": frozenset({".json"})})

        assert not _json_group().match_feature_group_criteria(SUFFIX_COLUMN, options, dac)
        assert JsonDocumentReader.match_subclass_data_access(dac, ["content"], options=options) == json_file

    def test_override_is_suffix_specific(self, tmp_path: Path) -> None:
        """Overriding .json does not affect .csv ownership."""
        csv_path = tmp_path / "data.csv"
        write_csv(csv_path, {"suffixown_csv_col": [1]})
        dac = DataAccessCollection(files={str(csv_path)})
        options = Options({"document_suffixes": frozenset({".json"})})

        assert load_group("csv_fg", "CsvFG").match_feature_group_criteria("suffixown_csv_col", options, dac)


class TestFeatureScopeUnaffected:
    """Feature scope (string path) should not be affected by suffix filtering."""

    def test_readdocument_string_path_still_works(self) -> None:
        result = StubJsonDocReader.match_subclass_data_access("doc.json", ["content"], options=Options({}))
        assert result == "doc.json"

    def test_json_fg_pointer_path_still_works(self, json_file: str) -> None:
        options = Options({"JsonFG": json_file})
        assert _json_group().match_feature_group_criteria(SUFFIX_COLUMN, options, None)


class TestNoOptionsBackwardCompatible:
    """Calling without options (as existing tests do) should preserve old behavior."""

    def test_json_fg_no_options(self, json_file: str) -> None:
        dac = DataAccessCollection(files={json_file})
        assert _json_group().match_feature_group_criteria(SUFFIX_COLUMN, Options({}), dac)

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


def _suffixes() -> Any:
    """Imported lazily so a missing module fails these tests, not the collection of the whole file."""
    return importlib.import_module("mloda_plugins.feature_group.input_data.file_suffixes")


class TestStructuredSuffixesAttribute:
    """Verify STRUCTURED_SUFFIXES contains all expected extensions."""

    def test_csv_in_structured(self) -> None:
        assert ".csv" in _suffixes().STRUCTURED_SUFFIXES

    def test_json_in_structured(self) -> None:
        assert ".json" in _suffixes().STRUCTURED_SUFFIXES

    def test_parquet_in_structured(self) -> None:
        assert ".parquet" in _suffixes().STRUCTURED_SUFFIXES

    def test_orc_in_structured(self) -> None:
        assert ".orc" in _suffixes().STRUCTURED_SUFFIXES

    def test_feather_in_structured(self) -> None:
        assert ".feather" in _suffixes().STRUCTURED_SUFFIXES

    def test_md_not_in_structured(self) -> None:
        assert ".md" not in _suffixes().STRUCTURED_SUFFIXES

    def test_text_not_in_structured(self) -> None:
        assert ".text" not in _suffixes().STRUCTURED_SUFFIXES


class TestSharedSuffixConstants:
    """file_suffixes holds the single definition of each format's suffixes and their union."""

    def test_per_format_constants(self) -> None:
        assert _suffixes().CSV_SUFFIXES == (".csv", ".CSV")
        assert _suffixes().PARQUET_SUFFIXES == (".parquet", ".PARQUET", ".pqt", ".PQT")
        assert _suffixes().JSON_SUFFIXES == (".json", ".JSON")
        assert _suffixes().FEATHER_SUFFIXES == (".feather",)
        assert _suffixes().ORC_SUFFIXES == (".orc", ".ORC")

    def test_structured_suffixes_is_the_union_of_the_formats(self) -> None:
        union = frozenset(
            _suffixes().CSV_SUFFIXES
            + _suffixes().PARQUET_SUFFIXES
            + _suffixes().JSON_SUFFIXES
            + _suffixes().FEATHER_SUFFIXES
            + _suffixes().ORC_SUFFIXES
        )
        assert isinstance(_suffixes().STRUCTURED_SUFFIXES, frozenset)
        assert _suffixes().STRUCTURED_SUFFIXES == union

    def test_structured_suffixes_equals_the_union_of_the_five_groups_suffixes(self) -> None:
        groups = [
            load_group("csv_fg", "CsvFG"),
            load_group("parquet_fg", "ParquetFG"),
            load_group("json_fg", "JsonFG"),
            load_group("feather_fg", "FeatherFG"),
            load_group("orc_fg", "OrcFG"),
        ]
        assert frozenset(suffix for group in groups for suffix in group.suffixes()) == _suffixes().STRUCTURED_SUFFIXES


class TestReadDocumentIndependentOfReadFile:
    """ReadDocument reads the shared constant and does not import the old ReadFile module."""

    def test_read_document_source_does_not_import_read_file(self) -> None:
        tree = ast.parse(inspect.getsource(importlib.import_module(ReadDocument.__module__)))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.append(node.module)
                imported.extend(f"{node.module}.{alias.name}" for alias in node.names)
            elif isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)

        assert not [name for name in imported if name.split(".")[-1] == "read_file"]
        assert "ReadFile" not in {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}

    def test_read_document_skips_every_structured_suffix(self) -> None:
        for suffix in sorted(_suffixes().STRUCTURED_SUFFIXES):
            assert ReadDocument._is_structured_suffix(f"data{suffix}", frozenset()) is True
            assert ReadDocument._is_structured_suffix(f"data{suffix}", frozenset({suffix})) is False
