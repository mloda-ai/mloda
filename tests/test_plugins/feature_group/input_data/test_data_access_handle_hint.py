"""Contract tests for the ``data_access_handle`` Options key flowing through
the file, document, and DB consumers of ``DataAccessCollection``.

In each case, multi-entry without a hint must raise ``ValueError`` listing the
candidate handles, the hint must disambiguate, and single-entry behavior is
preserved. See ``docs/docs/in_depth/named-data-access-handles.md``.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.user import Options
from mloda_plugins.feature_group.input_data.read_db import ReadDB
from mloda_plugins.feature_group.input_data.read_document import ReadDocument
from mloda_plugins.feature_group.input_data.read_file import ReadFile


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
def two_sqlite_dbs(tmp_path: Path) -> tuple[Path, Path]:
    """Two distinct, valid SQLite database files."""
    db_a = tmp_path / "warehouse.sqlite"
    db_b = tmp_path / "analytics.sqlite"
    for db in (db_a, db_b):
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()
    return db_a, db_b


# ----------------------------------------------------------------------------
# Concrete reader subclasses used only by these tests.
# ----------------------------------------------------------------------------


class _CsvLikeReader(ReadFile):
    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".csv",)

    @classmethod
    def get_column_names(cls, file_name: str) -> list[str]:
        raise NotImplementedError


class _TxtDocReader(ReadDocument):
    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".txt",)


# ----------------------------------------------------------------------------
# ReadFile: multi-file ambiguity raises, data_access_handle disambiguates
# ----------------------------------------------------------------------------


class TestReadFileHint:
    def test_multiple_files_without_hint_raises(self, two_csv_files: tuple[str, str]) -> None:
        path_a, path_b = two_csv_files
        dac = DataAccessCollection(files={"transactions": path_a, "users": path_b})
        with pytest.raises(ValueError) as excinfo:
            _CsvLikeReader.match_subclass_data_access(dac, feature_names=["id"], options=Options())
        msg = str(excinfo.value)
        assert "transactions" in msg
        assert "users" in msg

    def test_hint_disambiguates_to_named_file(self, two_csv_files: tuple[str, str]) -> None:
        path_a, path_b = two_csv_files
        dac = DataAccessCollection(files={"transactions": path_a, "users": path_b})
        options = Options(context={"data_access_handle": "users"})
        resolved = _CsvLikeReader.match_subclass_data_access(dac, feature_names=["id"], options=options)
        assert resolved == path_b

    def test_single_file_no_hint_resolves(self, two_csv_files: tuple[str, str]) -> None:
        path_a, _ = two_csv_files
        dac = DataAccessCollection(files={"transactions": path_a})
        resolved = _CsvLikeReader.match_subclass_data_access(dac, feature_names=["id"], options=Options())
        assert resolved == path_a


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


# ----------------------------------------------------------------------------
# ReadDB: multi-credentials ambiguity raises, data_access_handle disambiguates,
# single-credentials behaves like today.
# ----------------------------------------------------------------------------


class _AlwaysValidCredsDB(ReadDB):
    """Minimal DB reader whose credentials are any dict; feature presence is implicit."""

    @classmethod
    def is_valid_credentials(cls, credentials: dict[str, Any]) -> bool:
        return isinstance(credentials, dict) and "db_path" in credentials

    @classmethod
    def check_feature_in_data_access(cls, feature_name: str, data_access: Any) -> bool:
        return True


class TestReadDBHint:
    def test_multiple_credentials_without_hint_raises(self, two_sqlite_dbs: tuple[Path, Path]) -> None:
        db_a, db_b = two_sqlite_dbs
        dac = DataAccessCollection(
            credentials={
                "warehouse": {"db_path": str(db_a)},
                "analytics": {"db_path": str(db_b)},
            }
        )
        with pytest.raises(ValueError) as excinfo:
            _AlwaysValidCredsDB.match_subclass_data_access(dac, feature_names=["any"], options=Options())
        msg = str(excinfo.value)
        assert "warehouse" in msg
        assert "analytics" in msg

    def test_hint_disambiguates_to_named_credentials(self, two_sqlite_dbs: tuple[Path, Path]) -> None:
        db_a, db_b = two_sqlite_dbs
        dac = DataAccessCollection(
            credentials={
                "warehouse": {"db_path": str(db_a)},
                "analytics": {"db_path": str(db_b)},
            }
        )
        options = Options(context={"data_access_handle": "analytics"})
        resolved = _AlwaysValidCredsDB.match_subclass_data_access(dac, feature_names=["any"], options=options)
        assert isinstance(resolved, dict)
        assert resolved.get("db_path") == str(db_b)

    def test_single_credentials_no_hint_resolves(self, two_sqlite_dbs: tuple[Path, Path]) -> None:
        db_a, _ = two_sqlite_dbs
        dac = DataAccessCollection(credentials={"warehouse": {"db_path": str(db_a)}})
        resolved = _AlwaysValidCredsDB.match_subclass_data_access(dac, feature_names=["any"], options=Options())
        assert isinstance(resolved, dict)
        assert resolved.get("db_path") == str(db_a)


# Sanity check that fixtures are isolated (parallel-safety smoke).
def test_tmp_files_are_per_test(tmp_path: Path) -> None:
    assert tmp_path.exists()
    assert not os.listdir(tmp_path)
