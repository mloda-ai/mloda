"""Contract tests for the ``data_access_handle`` Options key flowing through
the file and document groups of ``DataAccessCollection`` (databases: the database contract mixin).

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
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.feature_group.input_data.file_formats.csv_fg import CsvFG
from tests.mixins.reader_feature_groups.lazy_format_group import load_document_group


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
# TextFG: multi-file ambiguity aborts, a handle that names a foreign file declines
# ----------------------------------------------------------------------------


class TestTextFGHint:
    def test_multiple_documents_without_hint_raises_naming_the_paths_and_handles(
        self, two_txt_files: tuple[str, str]
    ) -> None:
        path_a, path_b = two_txt_files
        text_group = load_document_group("text_fg", "TextFG")
        dac = DataAccessCollection(files={"notes_a": path_a, "notes_b": path_b})
        with pytest.raises(ValueError) as excinfo:
            resolve_or_raise(Feature("TextFG"), {text_group: {PyArrowTable}}, None, dac)
        msg = str(excinfo.value)
        assert os.path.abspath(path_a) in msg
        assert os.path.abspath(path_b) in msg
        assert "notes_a" in msg
        assert "notes_b" in msg

    def test_hint_at_foreign_file_declines_instead_of_rescanning(self, csv_and_txt_files: tuple[str, str]) -> None:
        """A handle naming a .csv file makes TextFG decline, not rescan and bind the .txt file nobody named."""
        csv_path, txt_path = csv_and_txt_files
        text_group = load_document_group("text_fg", "TextFG")
        dac = DataAccessCollection(files={"notes": txt_path, "data": csv_path})
        feature = Feature("TextFG", Options(context={"data_access_handle": "data"}))
        result = IdentifyFeatureGroupClass.evaluate(feature, {text_group: {PyArrowTable}}, None, dac)
        assert text_group not in result.identified


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
