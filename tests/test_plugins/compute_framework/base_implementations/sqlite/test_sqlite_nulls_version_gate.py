"""Runtime SQLite-version guard for NULLS FIRST/LAST in window order_by.

NULLS FIRST/LAST was introduced in SQLite 3.30.0 (2019-10-04). Calls that ask for
explicit NULLS placement must raise a clear Python ``ValueError`` when the runtime
SQLite is older than that, instead of producing an opaque syntax error from the
underlying engine. The guard must trigger only when an ``OrderBy(nulls=...)`` item
is present.
"""

import sqlite3
from typing import Any

import pyarrow as pa
import pytest

from mloda_plugins.compute_framework.base_implementations.sql.sql_window import OrderBy
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import (
    SqliteRelation,
)
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import _regexp


@pytest.fixture
def connection() -> Any:
    conn = sqlite3.connect(":memory:")
    conn.create_function("REGEXP", 2, _regexp, deterministic=True)
    yield conn
    conn.close()


@pytest.fixture
def sample_relation(connection: sqlite3.Connection) -> SqliteRelation:
    arrow = pa.Table.from_pydict(
        {
            "id": [1, 2, 3, 4, 5],
            "age": [25, 30, 35, 40, 45],
            "category": ["A", "B", "A", "C", "B"],
        }
    )
    return SqliteRelation.from_arrow(connection, arrow)


class TestSqliteNullsVersionGate:
    def test_window_with_nulls_last_on_old_sqlite_raises(
        self, sample_relation: SqliteRelation, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SqliteRelation.window must raise ValueError when nulls=... is requested on SQLite < 3.30."""
        monkeypatch.setattr("sqlite3.sqlite_version_info", (3, 28, 0))
        monkeypatch.setattr("sqlite3.sqlite_version", "3.28.0")
        with pytest.raises(ValueError) as excinfo:
            sample_relation.window("ROW_NUMBER()", "rn", order_by=(OrderBy("age", nulls="last"),))
        message = str(excinfo.value)
        assert "NULLS" in message
        assert "3.30" in message
        # The actual runtime sqlite version must appear in the error to aid debugging.
        assert "3.28.0" in message

    def test_with_row_number_with_nulls_first_on_old_sqlite_raises(
        self, sample_relation: SqliteRelation, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SqliteRelation.with_row_number must raise ValueError when nulls=... is requested on SQLite < 3.30."""
        monkeypatch.setattr("sqlite3.sqlite_version_info", (3, 28, 0))
        monkeypatch.setattr("sqlite3.sqlite_version", "3.28.0")
        with pytest.raises(ValueError) as excinfo:
            sample_relation.with_row_number("rn", order_by=(OrderBy("age", nulls="first"),))
        message = str(excinfo.value)
        assert "NULLS" in message
        assert "3.30" in message
        assert "3.28.0" in message

    def test_window_with_nulls_last_on_exact_floor_version_succeeds(
        self, sample_relation: SqliteRelation, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """At exactly SQLite 3.30.0 the gate must not fire: NULLS FIRST/LAST is supported (gate is ``<``, not ``<=``).

        The runtime sqlite is >= 3.30 so the underlying execute will succeed; we are only
        asserting that the Python-level guard does not raise here.
        """
        monkeypatch.setattr("sqlite3.sqlite_version_info", (3, 30, 0))
        monkeypatch.setattr("sqlite3.sqlite_version", "3.30.0")
        result = sample_relation.window("ROW_NUMBER()", "rn", order_by=(OrderBy("age", nulls="last"),))
        assert "rn" in result.columns
        assert len(result) == 5

    def test_window_without_nulls_on_old_sqlite_succeeds_bare_string(
        self, sample_relation: SqliteRelation, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Old SQLite is fine when no OrderBy(nulls=...) is requested: a bare string order_by must not trigger the gate."""
        monkeypatch.setattr("sqlite3.sqlite_version_info", (3, 28, 0))
        monkeypatch.setattr("sqlite3.sqlite_version", "3.28.0")
        result = sample_relation.window("ROW_NUMBER()", "rn", order_by=("id",))
        assert "rn" in result.columns
        assert len(result) == 5

    def test_window_without_nulls_on_old_sqlite_succeeds_orderby_descending_only(
        self, sample_relation: SqliteRelation, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Old SQLite is fine when OrderBy is used without nulls placement: only descending must not trigger the gate."""
        monkeypatch.setattr("sqlite3.sqlite_version_info", (3, 28, 0))
        monkeypatch.setattr("sqlite3.sqlite_version", "3.28.0")
        result = sample_relation.window("ROW_NUMBER()", "rn", order_by=(OrderBy("id", descending=True),))
        assert "rn" in result.columns
        assert len(result) == 5
