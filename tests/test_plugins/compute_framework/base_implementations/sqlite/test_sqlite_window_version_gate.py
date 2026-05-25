"""Runtime SQLite-version guard for the window-function helpers.

SQL window functions were introduced in SQLite 3.25.0, but the OVER-clause flavor
mloda relies on (named windows, frame clauses) only became reliably available in
3.28.0. Below that floor the helpers must raise a clear Python ``ValueError`` that
includes the runtime ``sqlite3.sqlite_version`` instead of an opaque syntax error
from the underlying engine.
"""

import sqlite3
from typing import Any

import pyarrow as pa
import pytest

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


class TestSqliteWindowVersionGate:
    def test_window_on_pre_3_28_sqlite_raises(
        self, sample_relation: SqliteRelation, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SqliteRelation.window must raise ValueError on SQLite < 3.28.0."""
        monkeypatch.setattr("sqlite3.sqlite_version_info", (3, 27, 0))
        monkeypatch.setattr("sqlite3.sqlite_version", "3.27.0")
        with pytest.raises(ValueError) as excinfo:
            sample_relation.window("COUNT(*)", "n")
        message = str(excinfo.value)
        assert "3.28" in message
        assert "3.27.0" in message

    def test_with_row_number_on_pre_3_28_sqlite_raises(
        self, sample_relation: SqliteRelation, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SqliteRelation.with_row_number must raise ValueError on SQLite < 3.28.0."""
        monkeypatch.setattr("sqlite3.sqlite_version_info", (3, 27, 0))
        monkeypatch.setattr("sqlite3.sqlite_version", "3.27.0")
        with pytest.raises(ValueError) as excinfo:
            sample_relation.with_row_number("rn")
        message = str(excinfo.value)
        assert "3.28" in message
        assert "3.27.0" in message
