"""Lazy sqlite column-semantics seam (issue #594)."""

import logging
import sqlite3
from typing import Any

import pytest

import mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_merge_engine as sqlite_merge_engine_module
from mloda.user import Index, JoinType
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_merge_engine import SqliteMergeEngine
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_value_sample import (
    sample_string_values as _real_sample_string_values,
)
from tests.test_plugins.compute_framework.test_tooling.merge_link import make_merge_link

logger = logging.getLogger(__name__)

try:
    import pyarrow as pa
except ImportError:
    logger.warning("PyArrow is not installed. Some tests will be skipped.")
    pa = None  # type: ignore[assignment]


def _spy_on_sample_string_values(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Count sample_string_values calls by column, delegating to the real implementation."""
    sampled_columns: list[str] = []

    def spy(data: Any, column: str) -> list[Any]:
        sampled_columns.append(column)
        return _real_sample_string_values(data, column)

    monkeypatch.setattr(sqlite_merge_engine_module, "sample_string_values", spy)
    return sampled_columns


@pytest.mark.skipif(pa is None, reason="PyArrow is not installed. Skipping this test.")
class TestSqliteLazyColumnSemantics:
    def test_string_vs_integer_equi_join_does_zero_sampling(
        self, connection: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        left = SqliteRelation.from_dict(connection, {"lk": ["a", "b"], "lv": [1, 2]})
        right = SqliteRelation.from_dict(connection, {"rk": [1, 2], "rv": [10, 20]})
        sampled_columns = _spy_on_sample_string_values(monkeypatch)

        engine = SqliteMergeEngine(connection)
        link = make_merge_link(JoinType.INNER, Index(("lk",)), Index(("rk",)))
        result = engine.merge(left, right, link)

        assert result.df() is not None
        assert sampled_columns == []

    def test_string_vs_string_id_equi_join_samples_left_only(
        self, connection: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        left = SqliteRelation.from_dict(connection, {"k": ["id_a", "id_b"], "lv": [1, 2]})
        right = SqliteRelation.from_dict(connection, {"k": ["id_a", "id_c"], "rv": [10, 20]})
        sampled_columns = _spy_on_sample_string_values(monkeypatch)

        engine = SqliteMergeEngine(connection)
        link = make_merge_link(JoinType.INNER, Index(("k",)), Index(("k",)))
        result = engine.merge(left, right, link)

        result_df = result.df()
        assert len(result_df) == 1
        assert result_df["k"].tolist() == ["id_a"]
        assert len(sampled_columns) == 1

    def test_iso_datetime_text_tz_mismatch_still_raises(self, connection: sqlite3.Connection) -> None:
        left = SqliteRelation.from_dict(
            connection,
            {"k": ["2024-01-01T00:00:00+00:00", "2024-06-01T12:30:00+00:00"], "lv": [1, 2]},
        )
        right = SqliteRelation.from_dict(
            connection,
            {"k": ["2024-01-01T00:00:00", "2024-06-01T12:30:00"], "rv": [10, 20]},
        )

        engine = SqliteMergeEngine(connection)
        link = make_merge_link(JoinType.INNER, Index(("k",)), Index(("k",)))
        with pytest.raises(ValueError, match=r"(?i)time[ -]?zone"):
            engine.merge(left, right, link)
