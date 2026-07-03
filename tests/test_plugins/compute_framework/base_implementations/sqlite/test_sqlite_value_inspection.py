"""Failing tests for sqlite ISO-TEXT datetime value-inspection (epic #518, follow-up).

sqlite stores datetimes as ISO-8601 TEXT, so schema-only introspection reports
such columns as plain (non-temporal) strings. This lets a tz-aware-vs-tz-naive
mismatch slip through the cross-engine comparison contract. The fix is for the
sqlite filter/merge engines' ``_column_semantics`` to sample actual values from
the relation and classify ISO-TEXT datetime columns as temporal with the correct
timezone awareness.

These tests build a ``SqliteRelation`` whose datetime column is stored as TEXT
(ISO strings) and assert the classification. They fail until Green wires value
inspection into ``_column_semantics``.
"""

import logging
import sqlite3
from typing import Any

import pytest

from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_filter_engine import SqliteFilterEngine
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_merge_engine import SqliteMergeEngine
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

logger = logging.getLogger(__name__)

try:
    import pyarrow as pa
except ImportError:
    logger.warning("PyArrow is not installed. Some tests will be skipped.")
    pa = None  # type: ignore[assignment]


@pytest.mark.skipif(pa is None, reason="PyArrow is not installed. Skipping this test.")
class TestSqliteFilterEngineValueInspection:
    """SqliteFilterEngine._column_semantics samples ISO-TEXT datetime values."""

    def test_naive_iso_text_column_is_temporal(self, connection: sqlite3.Connection) -> None:
        rel = SqliteRelation.from_dict(
            connection,
            {"ts": ["2024-01-01T00:00:00", "2024-06-01T12:30:00"]},
        )
        sem = SqliteFilterEngine._column_semantics(rel, "ts")
        assert sem.is_temporal is True
        assert sem.is_tz_aware is False

    def test_aware_iso_text_column_is_tz_aware(self, connection: sqlite3.Connection) -> None:
        rel = SqliteRelation.from_dict(
            connection,
            {"ts": ["2024-01-01T00:00:00+00:00", "2024-06-01T12:30:00+00:00"]},
        )
        sem = SqliteFilterEngine._column_semantics(rel, "ts")
        assert sem.is_temporal is True
        assert sem.is_tz_aware is True

    def test_z_suffix_iso_text_column_is_tz_aware(self, connection: sqlite3.Connection) -> None:
        rel = SqliteRelation.from_dict(
            connection,
            {"ts": ["2024-01-01T00:00:00Z", "2024-06-01T12:30:00Z"]},
        )
        sem = SqliteFilterEngine._column_semantics(rel, "ts")
        assert sem.is_temporal is True
        assert sem.is_tz_aware is True

    def test_plain_text_column_stays_non_temporal(self, connection: sqlite3.Connection) -> None:
        rel = SqliteRelation.from_dict(connection, {"name": ["Alice", "Bob"]})
        sem = SqliteFilterEngine._column_semantics(rel, "name")
        assert sem.is_temporal is False
        assert sem.is_tz_aware is False


@pytest.mark.skipif(pa is None, reason="PyArrow is not installed. Skipping this test.")
class TestSqliteMergeEngineValueInspection:
    """SqliteMergeEngine._column_semantics samples ISO-TEXT datetime values."""

    def _engine(self) -> Any:
        return SqliteMergeEngine.__new__(SqliteMergeEngine)

    def test_naive_iso_text_column_is_temporal(self, connection: sqlite3.Connection) -> None:
        rel = SqliteRelation.from_dict(
            connection,
            {"ts": ["2024-01-01T00:00:00", "2024-06-01T12:30:00"]},
        )
        sem = self._engine()._column_semantics(rel, "ts")
        assert sem.is_temporal is True
        assert sem.is_tz_aware is False

    def test_aware_iso_text_column_is_tz_aware(self, connection: sqlite3.Connection) -> None:
        rel = SqliteRelation.from_dict(
            connection,
            {"ts": ["2024-01-01T00:00:00+00:00", "2024-06-01T12:30:00+00:00"]},
        )
        sem = self._engine()._column_semantics(rel, "ts")
        assert sem.is_temporal is True
        assert sem.is_tz_aware is True

    def test_plain_text_column_stays_non_temporal(self, connection: sqlite3.Connection) -> None:
        rel = SqliteRelation.from_dict(connection, {"name": ["Alice", "Bob"]})
        sem = self._engine()._column_semantics(rel, "name")
        assert sem.is_temporal is False
        assert sem.is_tz_aware is False
