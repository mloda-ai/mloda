"""Contract tests for the ``data_access_handle`` Options key flowing through
the file and document consumers of ``DataAccessCollection`` (databases: the database contract mixin).

In each case, multi-entry without a hint must raise ``ValueError`` naming the
candidates, the hint must disambiguate, and single-entry behavior is
preserved. See ``docs/docs/in_depth/named-data-access-handles.md``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass, resolve_or_raise
from mloda.user import Feature, Options
from mloda_plugins.feature_group.input_data.read_document import ReadDocument
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.feature_group.input_data.file_formats.csv_fg import CsvFG


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------


@pytest.fixture
def two_csv_files(tmp_path: Path) -> tuple[str, str]:
    """Two distinct CSV file paths in an isolated tmp dir."""
    a = tmp_path / "transactions.csv"
    b = tmp_path / "users.csv"
    a.write_text("id,amount\n1,10\n")
    b.write_text("id,amount\n2,20\n")
    return str(a), str(b)


@pytest.fixture
def two_txt_files(tmp_path: Path) -> tuple[str, str]:
    """Two distinct .txt document paths in an isolated tmp dir."""
    a = tmp_path / "notes_a.txt"
    b = tmp_path / "notes_b.txt"
    a.write_text("hello")
    b.write_text("world")
    return str(a), str(b)


@pytest.fixture
def csv_and_txt_files(tmp_path: Path) -> tuple[str, str]:
    """A .csv path and a .txt path in an isolated tmp dir, for mixed-suffix hinting."""
    csv_path = tmp_path / "data.csv"
    txt_path = tmp_path / "notes.txt"
    csv_path.write_text("id,amount\n1,10\n")
    txt_path.write_text("hello")
    return str(csv_path), str(txt_path)


# ----------------------------------------------------------------------------
# Concrete reader subclasses used only by these tests.
# ----------------------------------------------------------------------------


class _TxtDocReader(ReadDocument):
    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".txt",)


# ----------------------------------------------------------------------------
# CsvFG: multi-file ambiguity aborts, data_access_handle narrows
# ----------------------------------------------------------------------------

_CSV_PLUGINS: Any = {CsvFG: {PyArrowTable}}


class TestCsvFGHint:
    def test_multiple_files_without_hint_raises(self, two_csv_files: tuple[str, str]) -> None:
        path_a, path_b = two_csv_files
        dac = DataAccessCollection(files={"transactions": path_a, "users": path_b})
        with pytest.raises(ValueError) as excinfo:
            resolve_or_raise(Feature("id"), _CSV_PLUGINS, None, dac)
        msg = str(excinfo.value)
        assert os.path.abspath(path_a) in msg
        assert os.path.abspath(path_b) in msg
        assert "data_access_handle" in msg


# ----------------------------------------------------------------------------
# ReadDocument: multi-file ambiguity raises, data_access_handle disambiguates
# ----------------------------------------------------------------------------


class TestReadDocumentHint:
    def test_multiple_documents_without_hint_raises(self, two_txt_files: tuple[str, str]) -> None:
        path_a, path_b = two_txt_files
        dac = DataAccessCollection(files={"notes_a": path_a, "notes_b": path_b})
        with pytest.raises(ValueError) as excinfo:
            _TxtDocReader.match_subclass_data_access(dac, feature_names=["content"], options=Options())
        msg = str(excinfo.value)
        assert "notes_a" in msg
        assert "notes_b" in msg

    def test_hint_disambiguates_to_named_document(self, two_txt_files: tuple[str, str]) -> None:
        path_a, path_b = two_txt_files
        dac = DataAccessCollection(files={"notes_a": path_a, "notes_b": path_b})
        options = Options(context={"data_access_handle": "notes_a"})
        resolved = _TxtDocReader.match_subclass_data_access(dac, feature_names=["content"], options=options)
        assert resolved == path_a

    def test_single_document_no_hint_resolves(self, two_txt_files: tuple[str, str]) -> None:
        path_a, _ = two_txt_files
        dac = DataAccessCollection(files={"notes_a": path_a})
        resolved = _TxtDocReader.match_subclass_data_access(dac, feature_names=["content"], options=Options())
        assert resolved == path_a

    def test_hint_at_foreign_file_declines_instead_of_rescanning(self, csv_and_txt_files: tuple[str, str]) -> None:
        """A hint naming a "file" handle this reader's own predicate rejects
        (a .csv file, which ReadDocument excludes as a structured suffix by default) must
        make the reader decline (None), not fall back to an unhinted rescan that silently
        binds the .txt file the caller never named.
        """
        csv_path, txt_path = csv_and_txt_files
        dac = DataAccessCollection(files={"notes": txt_path, "data": csv_path})
        options = Options(context={"data_access_handle": "data"})
        resolved = _TxtDocReader.match_subclass_data_access(dac, feature_names=["content"], options=options)
        # Crux of the bug: today this rescans the collection and wrongly returns txt_path
        # (the OTHER file, which the caller never hinted at) instead of declining.
        assert resolved != txt_path
        assert resolved is None


class TestDataAccessHandleRejectsCollectionsOutright:
    """A collection-shaped data_access_handle must not reach dict.get() as an unhashable key (#1165)."""

    def test_file_reader_rejects_a_list_handle_instead_of_crashing(self, two_csv_files: tuple[str, str]) -> None:
        path_a, path_b = two_csv_files
        dac = DataAccessCollection(files={"transactions": path_a, "users": path_b})
        feature = Feature("id", Options(context={"data_access_handle": ["users", "transactions"]}))
        result = IdentifyFeatureGroupClass.evaluate(feature, _CSV_PLUGINS, None, dac)
        assert CsvFG not in result.identified
        assert "data_access_handle" in result.eliminations[CsvFG].reason

    def test_a_pointed_file_missing_the_column_declines_with_the_option_rejection_not_an_abort(
        self, two_csv_files: tuple[str, str]
    ) -> None:
        path_a, _ = two_csv_files
        options = Options({"CsvFG": path_a}, context={"data_access_handle": ["users", "transactions"]})
        feature = Feature("hint_absent_column", options)
        result = IdentifyFeatureGroupClass.evaluate(feature, _CSV_PLUGINS, None, None)
        assert CsvFG not in result.identified
        assert "data_access_handle" in result.eliminations[CsvFG].reason


# Sanity check that fixtures are isolated (parallel-safety smoke).
def test_tmp_files_are_per_test(tmp_path: Path) -> None:
    assert tmp_path.exists()
    assert not os.listdir(tmp_path)
