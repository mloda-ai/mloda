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

from mloda.core.abstract_plugins.components.credential import Credential
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.user import Options
from mloda_plugins.feature_group.input_data.read_db import ReadDB
from mloda_plugins.feature_group.input_data.read_dbs.sqlite import SQLITEReader
from mloda_plugins.feature_group.input_data.read_document import ReadDocument
from mloda_plugins.feature_group.input_data.read_file import ReadFile
from mloda_plugins.feature_group.input_data.read_files.csv import CsvReader


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

    def test_single_file_set_form_no_handle_needed(self, two_csv_files: tuple[str, str]) -> None:
        """Bare set form with a single file resolves cleanly without a hint."""
        path_a, _ = two_csv_files
        dac = DataAccessCollection(files={path_a})
        resolved = _CsvLikeReader.match_subclass_data_access(dac, feature_names=["id"], options=Options())
        assert resolved == path_a

    def test_hint_at_foreign_file_declines_instead_of_rescanning(self, csv_and_txt_files: tuple[str, str]) -> None:
        """Issue #1170: a hint naming a "file" handle this reader's own predicate rejects
        must make the reader decline (None), not fall back to an unhinted rescan that
        silently binds a different file the caller never named.
        """
        csv_path, txt_path = csv_and_txt_files
        dac = DataAccessCollection(files={"data": csv_path, "notes": txt_path})
        options = Options(context={"data_access_handle": "notes"})
        resolved = _CsvLikeReader.match_subclass_data_access(dac, feature_names=["id"], options=options)
        # Crux of the bug: today this rescans the collection and wrongly returns csv_path
        # (the OTHER file, which the caller never hinted at) instead of declining.
        assert resolved != csv_path
        assert resolved is None


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
        """Issue #1170: a hint naming a "file" handle this reader's own predicate rejects
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

    def test_single_credentials_list_form_no_handle_needed(self, two_sqlite_dbs: tuple[Path, Path]) -> None:
        """Bare list form with a single credentials entry resolves without a hint."""
        db_a, _ = two_sqlite_dbs
        dac = DataAccessCollection(credentials=[{"db_path": str(db_a)}])
        resolved = _AlwaysValidCredsDB.match_subclass_data_access(dac, feature_names=["any"], options=Options())
        assert isinstance(resolved, dict)
        assert resolved.get("db_path") == str(db_a)


# ----------------------------------------------------------------------------
# ReadDB: resolve() must filter candidates through *this* reader's
# is_valid_credentials before flagging ambiguity, not treat every registered
# credentials entry as a match.
# ----------------------------------------------------------------------------


class _WarehouseCredsDB(ReadDB):
    """DB reader that only accepts credentials carrying its own connector key, not any dict."""

    @classmethod
    def is_valid_credentials(cls, credentials: dict[str, Any]) -> bool:
        return isinstance(credentials, dict) and "warehouse_dsn" in credentials

    @classmethod
    def check_feature_in_data_access(cls, feature_name: str, data_access: Any) -> bool:
        return True


class _AnalyticsCredsDB(ReadDB):
    """Sibling reader accepting a different connector key than _WarehouseCredsDB."""

    @classmethod
    def is_valid_credentials(cls, credentials: dict[str, Any]) -> bool:
        return isinstance(credentials, dict) and "analytics_dsn" in credentials

    @classmethod
    def check_feature_in_data_access(cls, feature_name: str, data_access: Any) -> bool:
        return True


class TestReadDBCredentialsPredicateFiltering:
    """resolve() must filter candidates through this reader's is_valid_credentials before flagging ambiguity."""

    def test_only_matching_entry_resolves_without_hint(self) -> None:
        dac = DataAccessCollection(
            credentials={
                "warehouse": {"warehouse_dsn": "warehouse://a"},
                "analytics": {"analytics_dsn": "analytics://b"},
            }
        )
        resolved = _WarehouseCredsDB.match_subclass_data_access(dac, feature_names=["any"], options=Options())
        assert resolved == {"warehouse_dsn": "warehouse://a"}

    def test_no_matching_entry_resolves_to_none(self) -> None:
        dac = DataAccessCollection(
            credentials={
                "analytics_one": {"analytics_dsn": "analytics://a"},
                "analytics_two": {"analytics_dsn": "analytics://b"},
            }
        )
        resolved = _WarehouseCredsDB.match_subclass_data_access(dac, feature_names=["any"], options=Options())
        assert resolved is None

    def test_sibling_readers_each_resolve_their_own_entry(self) -> None:
        dac = DataAccessCollection(
            credentials={
                "warehouse": {"warehouse_dsn": "warehouse://a"},
                "analytics": {"analytics_dsn": "analytics://b"},
            }
        )
        warehouse = _WarehouseCredsDB.match_subclass_data_access(dac, feature_names=["any"], options=Options())
        analytics = _AnalyticsCredsDB.match_subclass_data_access(dac, feature_names=["any"], options=Options())
        assert warehouse == {"warehouse_dsn": "warehouse://a"}
        assert analytics == {"analytics_dsn": "analytics://b"}

    def test_hint_at_foreign_credentials_declines_instead_of_rescanning(self) -> None:
        """A reader whose predicate rejects the hinted entry must decline, not rescan for its own unrelated entry."""
        dac = DataAccessCollection(
            credentials={
                "warehouse": {"warehouse_dsn": "warehouse://a"},
                "analytics": {"analytics_dsn": "analytics://b"},
            }
        )
        options = Options(context={"data_access_handle": "analytics"})
        resolved = _WarehouseCredsDB.match_subclass_data_access(dac, feature_names=["any"], options=options)
        assert resolved is None

    def test_hint_at_owned_credentials_still_resolves(self) -> None:
        """Sanity check: the correct owner still resolves via the same hint."""
        dac = DataAccessCollection(
            credentials={
                "warehouse": {"warehouse_dsn": "warehouse://a"},
                "analytics": {"analytics_dsn": "analytics://b"},
            }
        )
        options = Options(context={"data_access_handle": "analytics"})
        resolved = _AnalyticsCredsDB.match_subclass_data_access(dac, feature_names=["any"], options=options)
        assert resolved == {"analytics_dsn": "analytics://b"}

    def test_not_implemented_credentials_check_resolves_to_none_via_predicate(self) -> None:
        """Base ReadDB.is_valid_credentials always raises NotImplementedError; the predicate treats it as no match."""
        dac = DataAccessCollection(credentials={"only": {"whatever": "value"}})
        resolved = ReadDB.match_subclass_data_access(dac, feature_names=["any"], options=Options())
        assert resolved is None


# ----------------------------------------------------------------------------
# Typed Credential through the real SQLITEReader matcher (issue #511 pin).
# ----------------------------------------------------------------------------


class TestSqliteReaderMatchesTypedCredential:
    """Regression pin: a typed ``Credential`` flows end-to-end through the
    ReadDB matcher path and is matched as a plain credentials dict, never None.
    """

    def test_credential_kwarg_form_is_matched_by_sqlite_reader(self, two_sqlite_dbs: tuple[Path, Path]) -> None:
        db_a, _ = two_sqlite_dbs
        dac = DataAccessCollection(credentials=Credential(sqlite=str(db_a)))
        resolved = SQLITEReader.match_subclass_data_access(dac, feature_names=["id"], options=Options())
        assert resolved is not None
        assert isinstance(resolved, dict)
        assert resolved[SQLITEReader.db_path()] == str(db_a)


class TestDataAccessHandleRejectsCollectionsOutright:
    """A collection-shaped data_access_handle must not reach dict.get() as an unhashable key (#1165)."""

    def test_file_reader_rejects_a_list_handle_instead_of_crashing(self, two_csv_files: tuple[str, str]) -> None:
        path_a, path_b = two_csv_files
        dac = DataAccessCollection(files={"transactions": path_a, "users": path_b})
        options = Options(context={"data_access_handle": ["users", "transactions"]})
        assert CsvReader.match_data_access(["id"], dac, options=options) == (None, None)

    def test_db_reader_rejects_a_list_handle_instead_of_crashing(self, two_sqlite_dbs: tuple[Path, Path]) -> None:
        db_a, db_b = two_sqlite_dbs
        dac = DataAccessCollection(
            credentials={"warehouse": {"db_path": str(db_a)}, "analytics": {"db_path": str(db_b)}}
        )
        options = Options(context={"data_access_handle": ["warehouse", "analytics"]})
        assert SQLITEReader.match_data_access(["id"], dac, options=options) == (None, None)


# Sanity check that fixtures are isolated (parallel-safety smoke).
def test_tmp_files_are_per_test(tmp_path: Path) -> None:
    assert tmp_path.exists()
    assert not os.listdir(tmp_path)
